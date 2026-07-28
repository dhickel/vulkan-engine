//! Package resolver: normalizes, validates, and reads package-relative
//! paths under a trusted [`PackageRoot`].
//!
//! The resolver is the single shared path authority for both runtime and
//! tooling. All path confinement, symlink detection, device rejection,
//! and budget enforcement passes through this module.
//!
//! # Pipeline
//!
//! 1. Classify raw manifest reference → reject NUL/root/drive/UNC/archive-member/unsupported scheme
//! 2. Percent-decode exactly once where allowed → reject malformed or decoded traversal
//! 3. Normalize separators and components → reject `.`/`..` escapes
//! 4. Walk with symlink-aware metadata → reject symlinks/non-regular files
//! 5. Canonicalize root prefix → verify containment
//! 6. Check cumulative budget without mutation
//! 7. Read/hash → fail if metadata drifts
//! 8. Commit budget reservation only after read and optional hash verification succeed

use super::budget::BudgetLedger;
use super::{
    AuthorizedBytes, ConfinedResource, ContentIdentity, DiagnosticCode, LogicalResourceId,
    PackageIoError, PackageRoot, ResourceKind,
};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

// ---------------------------------------------------------------------------
// PackageResolver
// ---------------------------------------------------------------------------

/// The shared package resolver that loads resources under a [`PackageRoot`].
///
/// All path validation, symlink detection, budget reservation, reading,
/// and hashing is mediated through this type.
#[derive(Debug)]
pub struct PackageResolver {
    root: PackageRoot,
    ledger: BudgetLedger,
}

impl PackageResolver {
    /// Create a new package resolver from a trusted package root.
    pub fn new(root: PackageRoot, ledger: BudgetLedger) -> Self {
        PackageResolver { root, ledger }
    }

    /// Return a reference to the package root.
    pub fn root(&self) -> &PackageRoot {
        &self.root
    }

    /// Return a snapshot of the current budget ledger.
    pub fn budget_snapshot(&self) -> super::budget::BudgetSnapshot {
        self.ledger.snapshot()
    }

    /// Resolve a logical resource: validate the path, check containment,
    /// check/read/hash, atomically reserve budget, and return a [`ConfinedResource`].
    ///
    /// This is the primary entry point for loading package resources.
    pub fn resolve(
        &mut self,
        relative_path: &str,
        kind: ResourceKind,
    ) -> Result<ConfinedResource, PackageIoError> {
        self.resolve_inner(relative_path, kind, None)
    }

    /// Resolve a resource and verify its content hash matches an expected value.
    pub fn resolve_with_hash_check(
        &mut self,
        relative_path: &str,
        kind: ResourceKind,
        expected_hash: &ContentIdentity,
    ) -> Result<ConfinedResource, PackageIoError> {
        self.resolve_inner(relative_path, kind, Some(expected_hash))
    }

    fn resolve_inner(
        &mut self,
        relative_path: &str,
        kind: ResourceKind,
        expected_hash: Option<&ContentIdentity>,
    ) -> Result<ConfinedResource, PackageIoError> {
        let id = LogicalResourceId::new(relative_path, kind)?;

        // Walk each component of the normalized path, checking for
        // symlinks, device files, and non-regular files.
        let (file_path, metadata) = self.walk_and_verify(&id)?;

        // Verify it's a regular file
        if !metadata.is_file() {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoNotARegularFile,
                format!("resource is not a regular file: '{}'", file_path.display()),
            ));
        }

        let file_size = metadata.len();
        self.ledger.check_file_and_source_bytes(1, file_size)?;

        let data = read_file_exact_len(&file_path, file_size)?;

        // Verify file size hasn't changed since we inspected it.
        let post_read_meta = file_path.metadata().map_err(|e| {
            PackageIoError::io(DiagnosticCode::PackageIoMetadataFailed, &file_path, e)
        })?;
        if post_read_meta.len() != file_size {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoMetadataDrifted,
                format!(
                    "file size changed during read: was {}, now {}",
                    file_size,
                    post_read_meta.len()
                ),
            ));
        }

        let identity = ContentIdentity::from_bytes(&data);
        if let Some(expected_hash) = expected_hash {
            if identity != *expected_hash {
                return Err(PackageIoError::new(
                    DiagnosticCode::PackageIoHashMismatch,
                    format!(
                        "hash mismatch for '{}': expected {}, got {}",
                        relative_path,
                        expected_hash.hex(),
                        identity.hex()
                    ),
                ));
            }
        }

        self.ledger.reserve_file_and_source_bytes(1, file_size)?;

        Ok(ConfinedResource {
            id,
            bytes: AuthorizedBytes::new(data),
            identity,
        })
    }

    /// Walk the logical path from the package root, verifying every component.
    ///
    /// Returns the canonical file path and its metadata on success.
    /// Rejects symlinks, device files, sockets, and path escapes at every step.
    fn walk_and_verify(
        &self,
        id: &LogicalResourceId,
    ) -> Result<(PathBuf, fs::Metadata), PackageIoError> {
        let root = self.root.canonical_path();

        // Check for non-existent path early
        let mut current = root.to_path_buf();
        for part in id.as_str().split('/') {
            if part.is_empty() {
                return Err(PackageIoError::new(
                    DiagnosticCode::PackageIoEmptyPath,
                    "logical path has empty component",
                ));
            }
            current.push(part);

            // Check with symlink_metadata to reject symlinks
            let meta = match fs::symlink_metadata(&current) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(PackageIoError::io(
                        DiagnosticCode::PackageIoNotFound,
                        &current,
                        e,
                    ));
                }
                Err(e) => {
                    return Err(PackageIoError::io(
                        DiagnosticCode::PackageIoMetadataFailed,
                        &current,
                        e,
                    ));
                }
            };

            let ft = meta.file_type();

            // Reject symlinks at every component
            if ft.is_symlink() {
                return Err(PackageIoError::new(
                    DiagnosticCode::PackageIoSymlinkRejected,
                    format!("symlink component in path: '{}'", current.display()),
                ));
            }

            // For intermediate components, require directory
            // For the final component, it can be a file or directory
            let is_final = current.ends_with(id.as_str());
            if !is_final && !ft.is_dir() {
                return Err(PackageIoError::new(
                    DiagnosticCode::PackageIoNotADirectory,
                    format!(
                        "intermediate path component is not a directory: '{}'",
                        current.display()
                    ),
                ));
            }

            // Reject device files, FIFOs, sockets
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileTypeExt;
                if ft.is_block_device() || ft.is_char_device() {
                    return Err(PackageIoError::new(
                        DiagnosticCode::PackageIoDeviceFile,
                        format!("device file in path: '{}'", current.display()),
                    ));
                }
                if ft.is_fifo() {
                    return Err(PackageIoError::new(
                        DiagnosticCode::PackageIoDeviceFile,
                        format!("FIFO in path: '{}'", current.display()),
                    ));
                }
                if ft.is_socket() {
                    return Err(PackageIoError::new(
                        DiagnosticCode::PackageIoDeviceFile,
                        format!("socket in path: '{}'", current.display()),
                    ));
                }
            }
        }

        // Final containment check: canonicalize the resolved path and verify
        // it stays within the root.
        let canonical = current.canonicalize().map_err(|e| {
            PackageIoError::io(DiagnosticCode::PackageIoCanonicalizeFailed, &current, e)
        })?;

        if !canonical.starts_with(root) {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoPathEscape,
                format!(
                    "resolved path '{}' escapes root '{}'",
                    canonical.display(),
                    root.display()
                ),
            ));
        }

        let metadata = fs::metadata(&canonical).map_err(|e| {
            PackageIoError::io(DiagnosticCode::PackageIoMetadataFailed, &canonical, e)
        })?;

        Ok((canonical, metadata))
    }
}

fn read_file_exact_len(path: &Path, expected_len: u64) -> Result<Vec<u8>, PackageIoError> {
    let read_limit = expected_len.checked_add(1).ok_or_else(|| {
        PackageIoError::new(
            DiagnosticCode::PackageIoBudgetOverflow,
            "file length overflow",
        )
    })?;
    let file = fs::File::open(path)
        .map_err(|e| PackageIoError::io(DiagnosticCode::PackageIoReadFailed, path, e))?;
    let mut reader = file.take(read_limit);
    let mut data = Vec::new();
    reader
        .read_to_end(&mut data)
        .map_err(|e| PackageIoError::io(DiagnosticCode::PackageIoReadFailed, path, e))?;
    if data.len() as u64 != expected_len {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoMetadataDrifted,
            format!(
                "file size changed during read: was {}, read {}",
                expected_len,
                data.len()
            ),
        ));
    }
    Ok(data)
}

// ---------------------------------------------------------------------------
// Path normalization
// ---------------------------------------------------------------------------

/// Normalize a raw package-relative path string into a canonical logical form.
///
/// Steps:
/// 1. Reject empty strings
/// 2. Reject NUL bytes
/// 3. Reject absolute paths (starts with `/`, `\`, or has a prefix)
/// 4. Reject `data:` URI scheme and archive member syntax (`!` or `#` inside)
/// 5. Percent-decode exactly once where percent sequences appear
/// 6. Reject malformed percent sequences (`%` not followed by two hex digits)
/// 7. Reject decoded traversal (parent traversal or NUL after decode)
/// 8. Normalize separators to `/`
/// 9. Resolve `.` and `..` components lexically
/// 10. Reject resulting empty path or escape
pub fn normalize_logical_path(raw: &str) -> Result<String, PackageIoError> {
    // 1. Reject empty
    if raw.is_empty() {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoEmptyPath,
            "path must not be empty",
        ));
    }

    // 2. Reject NUL bytes
    if raw.contains('\0') {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoNullByte,
            "path must not contain NUL bytes",
        ));
    }

    // 3. Reject absolute paths
    if raw.starts_with('/') || raw.starts_with('\\') {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoAbsolutePath,
            format!("path must be relative, got '{raw}'"),
        ));
    }

    // Check for Windows absolute: C:\ or \\server\share
    if has_prefix_or_root(raw) {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoPrefixComponent,
            format!("path has root or prefix component: '{raw}'"),
        ));
    }

    // 4. Reject unsupported URI schemes and archive member syntax
    if raw.starts_with("data:") || raw.contains("data:") {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoDataUri,
            "data URIs are not supported in package paths",
        ));
    }
    if raw.contains('!') || raw.contains('#') {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoArchiveMember,
            "archive member and fragment references are not supported",
        ));
    }
    if raw.contains("://") {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoUnsupportedUriScheme,
            "URI schemes are not supported in package paths",
        ));
    }

    // 5. Percent-decode exactly once
    let decoded = percent_decode(raw)?;

    // 6. Reject decoded NUL
    if decoded.contains('\0') {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoNullByte,
            "decoded path contains NUL byte",
        ));
    }

    // 7. If percent-decoding actually changed the string, reject any
    //    `..` components that were hidden via encoding.
    //    Explicit literal `..` in the raw string is handled by
    //    normal lexical resolution below — only encoded escapes are
    //    rejected here.
    if raw != decoded {
        for component in decoded.split(&['/', '\\'][..]) {
            if component == ".." {
                return Err(PackageIoError::new(
                    DiagnosticCode::PackageIoParentTraversal,
                    format!("decoded path contains percent-encoded parent traversal: '{raw}' -> '{decoded}'"),
                ));
            }
        }
    }

    // 8. Reject decoded absolute path
    if decoded.starts_with('/') || decoded.starts_with('\\') {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoAbsolutePath,
            format!("decoded path became absolute: '{decoded}'"),
        ));
    }
    if has_prefix_or_root(&decoded) {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoPrefixComponent,
            format!("decoded path has root/prefix component: '{decoded}'"),
        ));
    }

    // 9. Also reject Windows drive-letter patterns that Rust doesn't
    // recognize as a Prefix on non-Windows platforms (e.g., "C:/").
    if has_windows_drive_like(&decoded) {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoPrefixComponent,
            format!("path has Windows drive-like component: '{decoded}'"),
        ));
    }

    // 10. Normalize separators to `/` and split
    let normalized_seps = decoded.replace('\\', "/");

    // 11. Resolve `.` and `..` lexically (genuine non-encoded `..`)
    let mut parts: Vec<&str> = Vec::new();
    for component in normalized_seps.split('/') {
        match component {
            "" | "." => {} // skip empty (multiple slashes) and current-dir
            ".." => {
                if parts.is_empty() {
                    return Err(PackageIoError::new(
                        DiagnosticCode::PackageIoParentTraversal,
                        format!("path escapes root: '{raw}' -> '{decoded}'"),
                    ));
                }
                parts.pop();
            }
            other => parts.push(other),
        }
    }

    // 10. Reject resulting empty path
    if parts.is_empty() {
        return Err(PackageIoError::new(
            DiagnosticCode::PackageIoEmptyPath,
            format!("normalized path is empty: '{raw}'"),
        ));
    }

    Ok(parts.join("/"))
}

/// Check if a path string has a Windows-style prefix or root component.
fn has_prefix_or_root(path: &str) -> bool {
    let p = Path::new(path);
    for component in p.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => return true,
            _ => {}
        }
    }
    false
}

/// Check for Windows drive-letter patterns that Rust on Linux doesn't
/// recognize as a `Prefix` component (e.g., `"C:file"`, `"C:/path"`).
fn has_windows_drive_like(path: &str) -> bool {
    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return true;
    }
    false
}

/// Percent-decode a path string exactly once.
///
/// `%XX` sequences are decoded to the corresponding byte.
/// Malformed sequences (`%` not followed by two hex digits) are rejected.
/// `%00` (NUL) is NOT rejected here — the caller checks for NUL after decode.
fn percent_decode(input: &str) -> Result<String, PackageIoError> {
    if !input.contains('%') {
        return Ok(input.to_string());
    }

    let bytes = input.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(PackageIoError::new(
                    DiagnosticCode::PackageIoAbsolutePath,
                    format!("malformed percent sequence at position {i} in '{input}'"),
                ));
            }
            let hex = &bytes[i + 1..i + 3];
            let decoded = hex_pair_to_byte(hex[0], hex[1]).ok_or_else(|| {
                PackageIoError::new(
                    DiagnosticCode::PackageIoAbsolutePath,
                    format!(
                        "invalid percent sequence '%{}' at position {} in '{}'",
                        String::from_utf8_lossy(hex),
                        i,
                        input
                    ),
                )
            })?;
            result.push(decoded);
            i += 3;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }

    String::from_utf8(result).map_err(|_| {
        PackageIoError::new(
            DiagnosticCode::PackageIoNullByte,
            "decoded path is not valid UTF-8",
        )
    })
}

fn hex_pair_to_byte(hi: u8, lo: u8) -> Option<u8> {
    let hi_nibble = hex_digit(hi)?;
    let lo_nibble = hex_digit(lo)?;
    Some(hi_nibble << 4 | lo_nibble)
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_simple_path() {
        assert_eq!(
            normalize_logical_path("maps/test.bsp").unwrap(),
            "maps/test.bsp"
        );
    }

    #[test]
    fn normalize_rejects_empty() {
        let err = normalize_logical_path("").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoEmptyPath);
    }

    #[test]
    fn normalize_rejects_null() {
        let err = normalize_logical_path("a\0b").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoNullByte);
    }

    #[test]
    fn normalize_rejects_absolute() {
        let err = normalize_logical_path("/etc/passwd").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoAbsolutePath);
    }

    #[test]
    fn normalize_rejects_backslash_absolute() {
        let err = normalize_logical_path("\\Windows\\system32").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoAbsolutePath);
    }

    #[test]
    fn normalize_rejects_parent_traversal() {
        let err = normalize_logical_path("../escape").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoParentTraversal);
    }

    #[test]
    fn normalize_rejects_double_parent_traversal() {
        let err = normalize_logical_path("a/../../escape").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoParentTraversal);
    }

    #[test]
    fn normalize_resolves_dot_components() {
        assert_eq!(normalize_logical_path("a/./b/./c").unwrap(), "a/b/c");
    }

    #[test]
    fn normalize_resolves_lexical_parent() {
        assert_eq!(normalize_logical_path("a/b/../c").unwrap(), "a/c");
    }

    #[test]
    fn normalize_resolves_multiple_parents() {
        assert_eq!(normalize_logical_path("a/b/c/../../d").unwrap(), "a/d");
    }

    #[test]
    fn normalize_rejects_escape_via_percent_decode() {
        // %2e%2e%2f = "../"
        let err = normalize_logical_path("a/%2e%2e%2fescape").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoParentTraversal);
    }

    #[test]
    fn normalize_rejects_percent_decoded_slash() {
        // %2f = "/"   => a/../.. has parent traversal after decode
        let err = normalize_logical_path("a%2f..%2f..").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoParentTraversal);
    }

    #[test]
    fn normalize_rejects_data_uri() {
        let err = normalize_logical_path("data:text/plain,hello").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoDataUri);
    }

    #[test]
    fn normalize_rejects_uri_scheme() {
        let err = normalize_logical_path("http://evil.com/payload").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoUnsupportedUriScheme);
    }

    #[test]
    fn normalize_rejects_archive_member() {
        let err = normalize_logical_path("archive.zip!member").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoArchiveMember);
    }

    #[test]
    fn normalize_percent_decode_valid() {
        assert_eq!(
            normalize_logical_path("file%20name.bsp").unwrap(),
            "file name.bsp"
        );
    }

    #[test]
    fn normalize_rejects_malformed_percent() {
        let err = normalize_logical_path("bad%GG").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoAbsolutePath);
    }

    #[test]
    fn normalize_rejects_truncated_percent() {
        let err = normalize_logical_path("bad%").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoAbsolutePath);
    }

    #[test]
    fn normalize_backslash_separator() {
        assert_eq!(normalize_logical_path(r"a\b\c").unwrap(), "a/b/c");
    }

    #[test]
    fn normalize_rejects_windows_absolute() {
        let err = normalize_logical_path(r"C:\Windows\system32").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoPrefixComponent);
    }

    #[test]
    fn normalize_rejects_empty_after_normalization() {
        let err = normalize_logical_path(".").unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoEmptyPath);
    }

    #[test]
    fn hex_digit_valid() {
        assert_eq!(hex_digit(b'0'), Some(0));
        assert_eq!(hex_digit(b'9'), Some(9));
        assert_eq!(hex_digit(b'A'), Some(10));
        assert_eq!(hex_digit(b'F'), Some(15));
        assert_eq!(hex_digit(b'a'), Some(10));
        assert_eq!(hex_digit(b'f'), Some(15));
    }

    #[test]
    fn hex_digit_invalid() {
        assert_eq!(hex_digit(b'G'), None);
        assert_eq!(hex_digit(b'g'), None);
    }

    #[test]
    fn hex_pair_to_byte_valid() {
        assert_eq!(hex_pair_to_byte(b'2', b'F'), Some(0x2F));
        assert_eq!(hex_pair_to_byte(b'0', b'0'), Some(0x00));
        assert_eq!(hex_pair_to_byte(b'F', b'F'), Some(0xFF));
    }
}
