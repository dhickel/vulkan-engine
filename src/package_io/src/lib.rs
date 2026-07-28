//! Canonical package trust boundary: confined loading, budget enforcement,
//! shared resolver for runtime and tooling.
//!
//! # Architecture
//!
//! - [`PackageRoot`] is a trusted, canonicalized absolute path that serves as
//!   the confinement root for all resource resolution.
//! - [`BudgetLedger`] tracks cumulative resource reservations and enforces
//!   hard limits. Every reservation is atomic — overflow or breach leaves the
//!   ledger unchanged.
//! - [`PackageResolver`] normalizes, validates, and reads package-relative
//!   paths under a [`PackageRoot`]. It is the single resolver shared by
//!   runtime and tooling.
//! - [`ConfinedResource`] is an authorized resource returned by the resolver,
//!   carrying content identity and typed bytes.
//!
//! # Crate Dependency Rules
//!
//! Zero dependencies beyond `std`. No renderer, Vulkan, BSP, physics, app,
//! windowing, async, or filesystem-watcher dependencies.

pub mod budget;
pub mod resolver;

use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// PackageRoot
// ---------------------------------------------------------------------------

/// A trusted, canonicalized absolute directory that serves as the confinement
/// root for all resource resolution.
///
/// Created from an absolute path that is canonicalized and verified to be a
/// directory. Once constructed, all resource access is mediated through
/// [`PackageResolver`], which enforces containment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageRoot {
    canonical_path: PathBuf,
}

impl PackageRoot {
    /// Create a new package root from an absolute path.
    ///
    /// The path is canonicalized (symlinks resolved) and verified to be
    /// a directory. Returns an error if the path is not absolute, cannot
    /// be canonicalized, is not a directory, or is a symlink.
    pub fn new(path: &Path) -> Result<Self, PackageIoError> {
        if !path.is_absolute() {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoInvalidRoot,
                format!("package root path must be absolute: '{}'", path.display()),
            ));
        }

        // Check that the path itself is not a symlink before canonicalization
        let meta = path
            .symlink_metadata()
            .map_err(|e| PackageIoError::io(DiagnosticCode::PackageIoMetadataFailed, path, e))?;
        if meta.file_type().is_symlink() {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoSymlinkRejected,
                format!("package root must not be a symlink: '{}'", path.display()),
            ));
        }

        let canonical = path.canonicalize().map_err(|e| {
            PackageIoError::io(DiagnosticCode::PackageIoCanonicalizeFailed, path, e)
        })?;

        let canonical_meta = canonical.metadata().map_err(|e| {
            PackageIoError::io(DiagnosticCode::PackageIoMetadataFailed, &canonical, e)
        })?;
        if !canonical_meta.is_dir() {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoNotADirectory,
                format!(
                    "package root must be a directory: '{}'",
                    canonical.display()
                ),
            ));
        }

        Ok(PackageRoot {
            canonical_path: canonical,
        })
    }

    /// The canonical absolute path of the package root.
    pub fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }
}

// ---------------------------------------------------------------------------
// LogicalResourceId
// ---------------------------------------------------------------------------

/// A package-relative logical resource identifier.
///
/// Normalized to use `/` separators with no `.`, `..`, NUL, or absolute
/// components. This is the key by which resources are addressed within
/// a package.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LogicalResourceId {
    /// Normalized package-relative path (canonical components, `/` separated).
    path: String,
    /// Resource classification.
    kind: ResourceKind,
}

impl LogicalResourceId {
    /// Create a validated logical resource ID from a package-relative path.
    ///
    /// The path must:
    /// - Be non-empty
    /// - Not contain NUL bytes
    /// - Not be absolute or have a root/prefix component
    /// - Not contain `..` parent traversal after normalization
    /// - Normalize to at least one component
    ///
    /// Returns the validated ID on success.
    pub fn new(relative_path: &str, kind: ResourceKind) -> Result<Self, PackageIoError> {
        let normalized = resolver::normalize_logical_path(relative_path)?;
        Ok(LogicalResourceId {
            path: normalized,
            kind,
        })
    }

    /// The normalized logical path (package-relative, `/` separated).
    pub fn as_str(&self) -> &str {
        &self.path
    }

    /// The resource classification.
    pub fn kind(&self) -> ResourceKind {
        self.kind
    }
}

impl fmt::Display for LogicalResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind.tag(), self.path)
    }
}

// ---------------------------------------------------------------------------
// ResourceKind
// ---------------------------------------------------------------------------

/// Resource classification for budget tracking and type-specific validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// A BSP map file.
    Bsp,
    /// A palette file (768 bytes).
    Palette,
    /// Colored light data file.
    Lit,
    /// A WAD2 texture archive.
    Wad,
    /// A loose replacement texture.
    Texture,
    /// A package manifest (TOML).
    Manifest,
    /// A model asset (glTF, etc.).
    Model,
    /// A generic binary asset.
    Generic,
}

impl ResourceKind {
    /// Human-readable tag for diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            ResourceKind::Bsp => "bsp",
            ResourceKind::Palette => "palette",
            ResourceKind::Lit => "lit",
            ResourceKind::Wad => "wad",
            ResourceKind::Texture => "texture",
            ResourceKind::Manifest => "manifest",
            ResourceKind::Model => "model",
            ResourceKind::Generic => "asset",
        }
    }
}

// ---------------------------------------------------------------------------
// AuthorizedBytes
// ---------------------------------------------------------------------------

/// Byte content that has passed path-confinement, budget-reservation,
/// and read-integrity checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedBytes(Vec<u8>);

impl AuthorizedBytes {
    /// Create authorized bytes (constructor only visible within the crate).
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        AuthorizedBytes(bytes)
    }

    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Consume and return the raw bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the content is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl AsRef<[u8]> for AuthorizedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// ConfinedResource
// ---------------------------------------------------------------------------

/// A resource that has been authorized by the package resolver.
///
/// Carries the logical identity, the authorized bytes, and a content hash.
#[derive(Debug, Clone)]
pub struct ConfinedResource {
    /// Logical resource identity.
    pub id: LogicalResourceId,
    /// Authorized byte content.
    pub bytes: AuthorizedBytes,
    /// Content identity hash.
    pub identity: ContentIdentity,
}

// ---------------------------------------------------------------------------
// ContentIdentity
// ---------------------------------------------------------------------------

/// A deterministic content hash computed from resource bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentIdentity([u8; 32]);

impl ContentIdentity {
    /// Compute a SHA-256 content identity from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        ContentIdentity(sha256(data))
    }

    /// Parse a SHA-256 hex digest into a content identity.
    pub fn from_sha256_hex(hex: &str) -> Result<Self, PackageIoError> {
        if hex.len() != 64 {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoInvalidHash,
                "SHA-256 digest must be 64 hex characters",
            ));
        }

        let mut bytes = [0u8; 32];
        for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
            bytes[i] = hex_pair_to_byte(chunk[0], chunk[1]).ok_or_else(|| {
                PackageIoError::new(
                    DiagnosticCode::PackageIoInvalidHash,
                    "SHA-256 digest contains a non-hex character",
                )
            })?;
        }
        Ok(ContentIdentity(bytes))
    }

    /// The raw SHA-256 hash bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Hex representation of the SHA-256 hash.
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Display for ContentIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hex())
    }
}

// ---------------------------------------------------------------------------
// DiagnosticCode
// ---------------------------------------------------------------------------

/// Stable machine-readable diagnostic codes for package I/O operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DiagnosticCode {
    // Root validation
    PackageIoInvalidRoot,
    PackageIoNotADirectory,
    PackageIoCanonicalizeFailed,
    PackageIoMetadataFailed,

    // Path normalization
    PackageIoEmptyPath,
    PackageIoNullByte,
    PackageIoAbsolutePath,
    PackageIoRootComponent,
    PackageIoPrefixComponent,
    PackageIoParentTraversal,
    PackageIoPathEscape,
    PackageIoUnsupportedUriScheme,
    PackageIoArchiveMember,
    PackageIoDataUri,

    // Filesystem
    PackageIoNotFound,
    PackageIoSymlinkRejected,
    PackageIoDeviceFile,
    PackageIoNotARegularFile,
    PackageIoMetadataDrifted,
    PackageIoReadFailed,

    // Budget
    PackageIoBudgetFileCount,
    PackageIoBudgetSourceBytes,
    PackageIoBudgetDecompressedBytes,
    PackageIoBudgetImagePixels,
    PackageIoBudgetImageDimensions,
    PackageIoBudgetDataUriDecode,
    PackageIoBudgetNestingDepth,
    PackageIoBudgetExternalModel,
    PackageIoBudgetAggregateExceeded,
    PackageIoBudgetOverflow,

    // Decompression
    PackageIoDecompressionRejected,

    // Hash
    PackageIoHashMismatch,
    PackageIoInvalidHash,
}

impl DiagnosticCode {
    pub fn as_str(self) -> &'static str {
        match self {
            DiagnosticCode::PackageIoInvalidRoot => "PKG-IO-INVALID-ROOT",
            DiagnosticCode::PackageIoNotADirectory => "PKG-IO-NOT-A-DIRECTORY",
            DiagnosticCode::PackageIoCanonicalizeFailed => "PKG-IO-CANONICALIZE-FAILED",
            DiagnosticCode::PackageIoMetadataFailed => "PKG-IO-METADATA-FAILED",
            DiagnosticCode::PackageIoEmptyPath => "PKG-IO-EMPTY-PATH",
            DiagnosticCode::PackageIoNullByte => "PKG-IO-NULL-BYTE",
            DiagnosticCode::PackageIoAbsolutePath => "PKG-IO-ABSOLUTE-PATH",
            DiagnosticCode::PackageIoRootComponent => "PKG-IO-ROOT-COMPONENT",
            DiagnosticCode::PackageIoPrefixComponent => "PKG-IO-PREFIX-COMPONENT",
            DiagnosticCode::PackageIoParentTraversal => "PKG-IO-PARENT-TRAVERSAL",
            DiagnosticCode::PackageIoPathEscape => "PKG-IO-PATH-ESCAPE",
            DiagnosticCode::PackageIoUnsupportedUriScheme => "PKG-IO-UNSUPPORTED-URI-SCHEME",
            DiagnosticCode::PackageIoArchiveMember => "PKG-IO-ARCHIVE-MEMBER",
            DiagnosticCode::PackageIoDataUri => "PKG-IO-DATA-URI",
            DiagnosticCode::PackageIoNotFound => "PKG-IO-NOT-FOUND",
            DiagnosticCode::PackageIoSymlinkRejected => "PKG-IO-SYMLINK-REJECTED",
            DiagnosticCode::PackageIoDeviceFile => "PKG-IO-DEVICE-FILE",
            DiagnosticCode::PackageIoNotARegularFile => "PKG-IO-NOT-A-REGULAR-FILE",
            DiagnosticCode::PackageIoMetadataDrifted => "PKG-IO-METADATA-DRIFTED",
            DiagnosticCode::PackageIoReadFailed => "PKG-IO-READ-FAILED",
            DiagnosticCode::PackageIoBudgetFileCount => "PKG-IO-BUDGET-FILE-COUNT",
            DiagnosticCode::PackageIoBudgetSourceBytes => "PKG-IO-BUDGET-SOURCE-BYTES",
            DiagnosticCode::PackageIoBudgetDecompressedBytes => "PKG-IO-BUDGET-DECOMPRESSED-BYTES",
            DiagnosticCode::PackageIoBudgetImagePixels => "PKG-IO-BUDGET-IMAGE-PIXELS",
            DiagnosticCode::PackageIoBudgetImageDimensions => "PKG-IO-BUDGET-IMAGE-DIMENSIONS",
            DiagnosticCode::PackageIoBudgetDataUriDecode => "PKG-IO-BUDGET-DATA-URI-DECODE",
            DiagnosticCode::PackageIoBudgetNestingDepth => "PKG-IO-BUDGET-NESTING-DEPTH",
            DiagnosticCode::PackageIoBudgetExternalModel => "PKG-IO-BUDGET-EXTERNAL-MODEL",
            DiagnosticCode::PackageIoBudgetAggregateExceeded => "PKG-IO-BUDGET-AGGREGATE-EXCEEDED",
            DiagnosticCode::PackageIoBudgetOverflow => "PKG-IO-BUDGET-OVERFLOW",
            DiagnosticCode::PackageIoDecompressionRejected => "PKG-IO-DECOMPRESSION-REJECTED",
            DiagnosticCode::PackageIoHashMismatch => "PKG-IO-HASH-MISMATCH",
            DiagnosticCode::PackageIoInvalidHash => "PKG-IO-INVALID-HASH",
        }
    }
}

// ---------------------------------------------------------------------------
// PackageIoError
// ---------------------------------------------------------------------------

/// A stable, fail-closed error for all package I/O operations.
#[derive(Debug)]
pub struct PackageIoError {
    /// Stable machine-readable diagnostic code.
    pub code: DiagnosticCode,
    /// Human-readable message.
    pub message: String,
    /// Optional underlying I/O error.
    source: Option<std::io::Error>,
}

impl PackageIoError {
    pub fn new(code: DiagnosticCode, message: impl Into<String>) -> Self {
        PackageIoError {
            code,
            message: message.into(),
            source: None,
        }
    }

    pub fn io(code: DiagnosticCode, path: &Path, source: std::io::Error) -> Self {
        PackageIoError {
            code,
            message: format!("{}: '{}': {}", code.as_str(), path.display(), source),
            source: Some(source),
        }
    }
}

impl fmt::Display for PackageIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for PackageIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

fn hex_pair_to_byte(hi: u8, lo: u8) -> Option<u8> {
    let hi = hex_digit(hi)?;
    let lo = hex_digit(lo)?;
    Some((hi << 4) | lo)
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.total_len += data.len() as u64;
        let mut offset = 0usize;
        while offset < data.len() {
            let space = 64 - self.buf_len;
            let copy = (data.len() - offset).min(space);
            self.buf[self.buf_len..self.buf_len + copy]
                .copy_from_slice(&data[offset..offset + copy]);
            self.buf_len += copy;
            offset += copy;
            if self.buf_len == 64 {
                self.process_block();
                self.buf_len = 0;
            }
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let total_bits = self.total_len * 8;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            for i in self.buf_len..64 {
                self.buf[i] = 0;
            }
            self.process_block();
            self.buf_len = 0;
        }
        for i in self.buf_len..56 {
            self.buf[i] = 0;
        }
        self.buf[56..64].copy_from_slice(&total_bits.to_be_bytes());
        self.process_block();

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn process_block(&mut self) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let base = i * 4;
            *word = u32::from_be_bytes([
                self.buf[base],
                self.buf[base + 1],
                self.buf[base + 2],
                self.buf[base + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}
