//! Private scene-file publication primitives.
//!
//! Scene saves stage bytes into a same-directory sibling, keep the reserved
//! file handle through write/fsync, and publish with a same-parent rename.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub(crate) fn save_scene_file(target: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut staged = StagedSceneFile::reserve(target)?;
    if let Err(err) = staged.write_all(bytes).and_then(|_| staged.sync_file()) {
        let cleanup = staged.cleanup();
        return Err(merge_cleanup_error(err, cleanup));
    }

    if let Err(err) = staged.publish() {
        let cleanup = staged.cleanup();
        return Err(merge_cleanup_error(err, cleanup));
    }

    Ok(())
}

fn merge_cleanup_error(primary: io::Error, cleanup: io::Result<()>) -> io::Error {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup_err) => io::Error::new(
            primary.kind(),
            format!("{primary}; cleanup of staged scene file also failed: {cleanup_err}"),
        ),
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    use std::ffi::CString;
    use std::fs::{File, OpenOptions};
    use std::mem::MaybeUninit;
    use std::os::fd::{AsRawFd, FromRawFd, RawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    use std::sync::atomic::{AtomicU64, Ordering};

    static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FileIdentity {
        dev: u64,
        ino: u64,
    }

    impl FileIdentity {
        fn from_metadata(metadata: &std::fs::Metadata) -> Self {
            Self {
                dev: metadata.dev(),
                ino: metadata.ino(),
            }
        }

        fn from_stat(stat: &libc::stat) -> Self {
            Self {
                dev: stat.st_dev as u64,
                ino: stat.st_ino as u64,
            }
        }
    }

    pub(super) struct StagedSceneFile {
        parent: File,
        file: File,
        target_name: CString,
        staged_name: CString,
        staged_path: PathBuf,
        staged_identity: FileIdentity,
        published: bool,
    }

    impl StagedSceneFile {
        pub(super) fn reserve(target: &Path) -> io::Result<Self> {
            let parent_path = target.parent().unwrap_or_else(|| Path::new("."));
            let target_name = path_file_name_cstring(target)?;
            let parent = open_parent_no_follow(parent_path)?;
            let parent_fd = parent.as_raw_fd();

            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let pid = std::process::id();

            for attempt in 0..1024u64 {
                let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
                let staged_label = format!(".scene-save.{pid}.{nanos}.{counter}.{attempt}.tmp");
                let staged_name = CString::new(staged_label.as_bytes()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "staged name contains NUL")
                })?;

                match openat_new_no_follow(parent_fd, &staged_name) {
                    Ok(file) => {
                        let metadata = file.metadata()?;
                        return Ok(Self {
                            parent,
                            file,
                            target_name,
                            staged_path: parent_path.join(staged_label),
                            staged_name,
                            staged_identity: FileIdentity::from_metadata(&metadata),
                            published: false,
                        });
                    }
                    Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(err) => return Err(err),
                }
            }

            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not reserve unique staged scene file",
            ))
        }

        pub(super) fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.file.write_all(bytes)
        }

        pub(super) fn sync_file(&mut self) -> io::Result<()> {
            self.file.flush()?;
            self.file.sync_all()
        }

        pub(super) fn publish(&mut self) -> io::Result<()> {
            self.verify_staged_identity()?;
            self.verify_target_absent_or_regular()?;

            let rc = unsafe {
                libc::renameat(
                    self.parent.as_raw_fd(),
                    self.staged_name.as_ptr(),
                    self.parent.as_raw_fd(),
                    self.target_name.as_ptr(),
                )
            };
            if rc != 0 {
                return Err(io::Error::last_os_error());
            }
            self.parent.sync_all()?;
            self.published = true;
            Ok(())
        }

        pub(super) fn cleanup(&mut self) -> io::Result<()> {
            if self.published {
                return Ok(());
            }
            self.verify_staged_identity()?;
            let rc =
                unsafe { libc::unlinkat(self.parent.as_raw_fd(), self.staged_name.as_ptr(), 0) };
            if rc == 0 {
                self.published = true;
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }

        fn verify_staged_identity(&self) -> io::Result<()> {
            let stat = fstatat_no_follow(self.parent.as_raw_fd(), &self.staged_name)?;
            let observed = FileIdentity::from_stat(&stat);
            if observed == self.staged_identity {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!(
                        "staged scene identity changed before publication/cleanup: '{}'",
                        self.staged_path.display()
                    ),
                ))
            }
        }

        fn verify_target_absent_or_regular(&self) -> io::Result<()> {
            match fstatat_no_follow(self.parent.as_raw_fd(), &self.target_name) {
                Ok(stat) if (stat.st_mode & libc::S_IFMT) == libc::S_IFREG => Ok(()),
                Ok(_) => Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "scene save target exists but is not a regular file",
                )),
                Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(err) => Err(err),
            }
        }
    }

    impl Drop for StagedSceneFile {
        fn drop(&mut self) {
            let _ = self.cleanup();
        }
    }

    fn open_parent_no_follow(path: &Path) -> io::Result<File> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        let metadata = file.metadata()?;
        if metadata.file_type().is_dir() {
            Ok(file)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "scene save parent is not a directory",
            ))
        }
    }

    fn path_file_name_cstring(path: &Path) -> io::Result<CString> {
        let name = path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "scene target has no file name")
        })?;
        CString::new(name.as_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "scene target file name contains an interior NUL byte",
            )
        })
    }

    fn openat_new_no_follow(parent_fd: RawFd, name: &CString) -> io::Result<File> {
        let fd = unsafe {
            libc::openat(
                parent_fd,
                name.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(unsafe { File::from_raw_fd(fd) })
        }
    }

    fn fstatat_no_follow(parent_fd: RawFd, name: &CString) -> io::Result<libc::stat> {
        let mut stat = MaybeUninit::<libc::stat>::uninit();
        let rc = unsafe {
            libc::fstatat(
                parent_fd,
                name.as_ptr(),
                stat.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if rc == 0 {
            Ok(unsafe { stat.assume_init() })
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

#[cfg(not(unix))]
mod platform {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::sync::atomic::{AtomicU64, Ordering};

    static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

    pub(super) struct StagedSceneFile {
        target: PathBuf,
        staged: PathBuf,
        file: File,
        published: bool,
    }

    impl StagedSceneFile {
        pub(super) fn reserve(target: &Path) -> io::Result<Self> {
            let parent = target.parent().unwrap_or_else(|| Path::new("."));
            let _ = target.file_name().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "scene target has no file name")
            })?;
            let pid = std::process::id();
            for attempt in 0..1024u64 {
                let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
                let staged = parent.join(format!(".scene-save.{pid}.{counter}.{attempt}.tmp"));
                match OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&staged)
                {
                    Ok(file) => {
                        return Ok(Self {
                            target: target.to_path_buf(),
                            staged,
                            file,
                            published: false,
                        })
                    }
                    Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(err) => return Err(err),
                }
            }
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not reserve unique staged scene file",
            ))
        }

        pub(super) fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.file.write_all(bytes)
        }

        pub(super) fn sync_file(&mut self) -> io::Result<()> {
            self.file.flush()?;
            self.file.sync_all()
        }

        pub(super) fn publish(&mut self) -> io::Result<()> {
            std::fs::rename(&self.staged, &self.target)?;
            self.published = true;
            Ok(())
        }

        pub(super) fn cleanup(&mut self) -> io::Result<()> {
            if self.published {
                Ok(())
            } else {
                match std::fs::remove_file(&self.staged) {
                    Ok(()) => {
                        self.published = true;
                        Ok(())
                    }
                    Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
                    Err(err) => Err(err),
                }
            }
        }
    }

    impl Drop for StagedSceneFile {
        fn drop(&mut self) {
            let _ = self.cleanup();
        }
    }
}

use platform::StagedSceneFile;
