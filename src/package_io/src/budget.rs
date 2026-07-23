//! Checked resource budget reservations: file count, source bytes,
//! decompressed bytes, image pixels/dimensions, data-URI decoded bytes,
//! nesting/recursion depth, external-model buffers/images, and aggregate
//! package/mount totals.
//!
//! Every reservation is atomic — overflow or limit breach leaves the
//! [`BudgetLedger`] unchanged. The ledger exposes [`BudgetSnapshot`]
//! for inspection without mutable access.

use super::{DiagnosticCode, PackageIoError};

// ---------------------------------------------------------------------------
// Default budget limits
// ---------------------------------------------------------------------------

/// Default maximum file count per package.
pub const DEFAULT_MAX_FILE_COUNT: u64 = 1_000;
/// Default maximum total source bytes per package.
pub const DEFAULT_MAX_SOURCE_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB
/// Default maximum decompressed bytes per package.
pub const DEFAULT_MAX_DECOMPRESSED_BYTES: u64 = 512 * 1024 * 1024; // 512 MiB
/// Default maximum total image pixels per package.
pub const DEFAULT_MAX_IMAGE_PIXELS: u64 = 64 * 1024 * 1024; // 64 Mpix
/// Default maximum image dimension (width or height).
pub const DEFAULT_MAX_IMAGE_DIMENSION: u32 = 16_384;
/// Default maximum data-URI decoded bytes.
pub const DEFAULT_MAX_DATA_URI_BYTES: u64 = 4 * 1024 * 1024; // 4 MiB
/// Default maximum nesting/recursion depth for resource resolution.
pub const DEFAULT_MAX_NESTING_DEPTH: u32 = 16;
/// Default maximum external model buffer bytes.
pub const DEFAULT_MAX_EXTERNAL_MODEL_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB
/// Default maximum external model image bytes.
pub const DEFAULT_MAX_EXTERNAL_MODEL_IMAGE_BYTES: u64 = 32 * 1024 * 1024; // 32 MiB
/// Default aggregate package total (sum of all categories).
pub const DEFAULT_AGGREGATE_PACKAGE_BYTES: u64 = 384 * 1024 * 1024; // 384 MiB
/// Default aggregate mount total.
pub const DEFAULT_AGGREGATE_MOUNT_BYTES: u64 = 512 * 1024 * 1024; // 512 MiB

// ---------------------------------------------------------------------------
// ResourceBudget
// ---------------------------------------------------------------------------

/// Hard budget limits for a package or mount.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBudget {
    /// Maximum number of files.
    pub max_file_count: u64,
    /// Maximum total source bytes.
    pub max_source_bytes: u64,
    /// Maximum total decompressed bytes.
    pub max_decompressed_bytes: u64,
    /// Maximum total image pixels.
    pub max_image_pixels: u64,
    /// Maximum image dimension (width or height).
    pub max_image_dimension: u32,
    /// Maximum data-URI decoded bytes.
    pub max_data_uri_bytes: u64,
    /// Maximum nesting/recursion depth.
    pub max_nesting_depth: u32,
    /// Maximum total external model buffer bytes.
    pub max_external_model_bytes: u64,
    /// Maximum total external model image bytes.
    pub max_external_model_image_bytes: u64,
    /// Aggregate package total (sum of all categories).
    pub aggregate_package_bytes: u64,
    /// Aggregate mount total.
    pub aggregate_mount_bytes: u64,
}

impl Default for ResourceBudget {
    fn default() -> Self {
        ResourceBudget {
            max_file_count: DEFAULT_MAX_FILE_COUNT,
            max_source_bytes: DEFAULT_MAX_SOURCE_BYTES,
            max_decompressed_bytes: DEFAULT_MAX_DECOMPRESSED_BYTES,
            max_image_pixels: DEFAULT_MAX_IMAGE_PIXELS,
            max_image_dimension: DEFAULT_MAX_IMAGE_DIMENSION,
            max_data_uri_bytes: DEFAULT_MAX_DATA_URI_BYTES,
            max_nesting_depth: DEFAULT_MAX_NESTING_DEPTH,
            max_external_model_bytes: DEFAULT_MAX_EXTERNAL_MODEL_BYTES,
            max_external_model_image_bytes: DEFAULT_MAX_EXTERNAL_MODEL_IMAGE_BYTES,
            aggregate_package_bytes: DEFAULT_AGGREGATE_PACKAGE_BYTES,
            aggregate_mount_bytes: DEFAULT_AGGREGATE_MOUNT_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// BudgetSnapshot
// ---------------------------------------------------------------------------

/// A read-only snapshot of cumulative budget state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub file_count: u64,
    pub source_bytes: u64,
    pub decompressed_bytes: u64,
    pub image_pixels: u64,
    pub data_uri_bytes: u64,
    pub nesting_depth: u32,
    pub external_model_buffer_bytes: u64,
    pub external_model_image_bytes: u64,
}

// ---------------------------------------------------------------------------
// BudgetLedger
// ---------------------------------------------------------------------------

/// A cumulative resource budget ledger with checked atomic reservations.
///
/// Every reservation attempt that would overflow a u64 accumulator or exceed
/// a hard limit leaves the ledger completely unchanged — no partial mutations.
/// The ledger exposes a [`BudgetSnapshot`] for read-only inspection.
#[derive(Debug, Clone)]
pub struct BudgetLedger {
    budget: ResourceBudget,
    snapshot: BudgetSnapshot,
}

impl BudgetLedger {
    /// Create a new ledger with the given budget limits.
    pub fn new(budget: ResourceBudget) -> Self {
        BudgetLedger {
            budget,
            snapshot: BudgetSnapshot {
                file_count: 0,
                source_bytes: 0,
                decompressed_bytes: 0,
                image_pixels: 0,
                data_uri_bytes: 0,
                nesting_depth: 0,
                external_model_buffer_bytes: 0,
                external_model_image_bytes: 0,
            },
        }
    }

    /// Create a ledger with default budget limits.
    pub fn default_ledger() -> Self {
        Self::new(ResourceBudget::default())
    }

    /// Return a read-only snapshot of the current ledger state.
    pub fn snapshot(&self) -> BudgetSnapshot {
        self.snapshot
    }

    /// Check whether one logical file and its source bytes can be reserved
    /// without mutating the ledger.
    pub fn check_file_and_source_bytes(
        &self,
        file_count: u64,
        source_bytes: u64,
    ) -> Result<(), PackageIoError> {
        let new_file_count = self
            .snapshot
            .file_count
            .checked_add(file_count)
            .ok_or_else(|| overflow_err())?;
        if new_file_count > self.budget.max_file_count {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetFileCount,
                format!(
                    "file count {} would exceed limit {}",
                    new_file_count, self.budget.max_file_count
                ),
            ));
        }

        let new_source_bytes = self
            .snapshot
            .source_bytes
            .checked_add(source_bytes)
            .ok_or_else(|| overflow_err())?;
        if new_source_bytes > self.budget.max_source_bytes {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetSourceBytes,
                format!(
                    "source bytes {} would exceed limit {}",
                    new_source_bytes, self.budget.max_source_bytes
                ),
            ));
        }

        self.check_aggregate_tentative(source_bytes, AggregateField::SourceBytes)
    }

    /// Atomically reserve one logical file and its source bytes.
    pub fn reserve_file_and_source_bytes(
        &mut self,
        file_count: u64,
        source_bytes: u64,
    ) -> Result<(), PackageIoError> {
        self.check_file_and_source_bytes(file_count, source_bytes)?;
        self.snapshot.file_count = self
            .snapshot
            .file_count
            .checked_add(file_count)
            .ok_or_else(|| overflow_err())?;
        self.snapshot.source_bytes = self
            .snapshot
            .source_bytes
            .checked_add(source_bytes)
            .ok_or_else(|| overflow_err())?;
        Ok(())
    }

    // ── Aggregate check (called BEFORE mutation) ──────────────────────

    /// Compute the tentative aggregate total with a candidate delta applied
    /// to the given field selector. Returns the tentative total or an
    /// overflow/rejection error. DOES NOT mutate the ledger.
    fn check_aggregate_tentative(
        &self,
        delta: u64,
        field: AggregateField,
    ) -> Result<(), PackageIoError> {
        let snapshot = self.snapshot;
        // Sum all fields, using the tentative value for the one being updated
        let parts: [u64; 5] = match field {
            AggregateField::SourceBytes => [
                snapshot
                    .source_bytes
                    .checked_add(delta)
                    .ok_or_else(|| overflow_err())?,
                snapshot.decompressed_bytes,
                snapshot.data_uri_bytes,
                snapshot.external_model_buffer_bytes,
                snapshot.external_model_image_bytes,
            ],
            AggregateField::DecompressedBytes => [
                snapshot.source_bytes,
                snapshot
                    .decompressed_bytes
                    .checked_add(delta)
                    .ok_or_else(|| overflow_err())?,
                snapshot.data_uri_bytes,
                snapshot.external_model_buffer_bytes,
                snapshot.external_model_image_bytes,
            ],
            AggregateField::DataUriBytes => [
                snapshot.source_bytes,
                snapshot.decompressed_bytes,
                snapshot
                    .data_uri_bytes
                    .checked_add(delta)
                    .ok_or_else(|| overflow_err())?,
                snapshot.external_model_buffer_bytes,
                snapshot.external_model_image_bytes,
            ],
            AggregateField::ExternalModelBufferBytes => [
                snapshot.source_bytes,
                snapshot.decompressed_bytes,
                snapshot.data_uri_bytes,
                snapshot
                    .external_model_buffer_bytes
                    .checked_add(delta)
                    .ok_or_else(|| overflow_err())?,
                snapshot.external_model_image_bytes,
            ],
            AggregateField::ExternalModelImageBytes => [
                snapshot.source_bytes,
                snapshot.decompressed_bytes,
                snapshot.data_uri_bytes,
                snapshot.external_model_buffer_bytes,
                snapshot
                    .external_model_image_bytes
                    .checked_add(delta)
                    .ok_or_else(|| overflow_err())?,
            ],
        };

        let total: u64 = parts
            .iter()
            .try_fold(0u64, |acc, &p| acc.checked_add(p))
            .ok_or_else(|| overflow_err())?;

        if total > self.budget.aggregate_package_bytes {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetAggregateExceeded,
                format!(
                    "aggregate package total {} would exceed limit {}",
                    total, self.budget.aggregate_package_bytes
                ),
            ));
        }
        Ok(())
    }

    // ── Reservation methods ────────────────────────────────────────────

    /// Reserve one file. Fails if the count would exceed `max_file_count` or overflow.
    pub fn reserve_file_count(&mut self, count: u64) -> Result<(), PackageIoError> {
        let new_count = self
            .snapshot
            .file_count
            .checked_add(count)
            .ok_or_else(|| overflow_err())?;
        if new_count > self.budget.max_file_count {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetFileCount,
                format!(
                    "file count {} would exceed limit {}",
                    new_count, self.budget.max_file_count
                ),
            ));
        }
        // File count doesn't affect aggregate — commit
        self.snapshot.file_count = new_count;
        Ok(())
    }

    /// Reserve source bytes. Fails on overflow or if the cumulative total exceeds
    /// `max_source_bytes`.
    pub fn reserve_source_bytes(&mut self, bytes: u64) -> Result<(), PackageIoError> {
        let new_total = self
            .snapshot
            .source_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow_err())?;
        if new_total > self.budget.max_source_bytes {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetSourceBytes,
                format!(
                    "source bytes {} would exceed limit {}",
                    new_total, self.budget.max_source_bytes
                ),
            ));
        }
        // Check aggregate BEFORE committing
        self.check_aggregate_tentative(bytes, AggregateField::SourceBytes)?;
        self.snapshot.source_bytes = new_total;
        Ok(())
    }

    /// Reserve decompressed bytes. Fails on overflow or if the limit is exceeded.
    pub fn reserve_decompressed_bytes(&mut self, bytes: u64) -> Result<(), PackageIoError> {
        let new_total = self
            .snapshot
            .decompressed_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow_err())?;
        if new_total > self.budget.max_decompressed_bytes {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetDecompressedBytes,
                format!(
                    "decompressed bytes {} would exceed limit {}",
                    new_total, self.budget.max_decompressed_bytes
                ),
            ));
        }
        self.check_aggregate_tentative(bytes, AggregateField::DecompressedBytes)?;
        self.snapshot.decompressed_bytes = new_total;
        Ok(())
    }

    /// Reserve image pixels. Fails on overflow or if the limit is exceeded.
    /// Also validates that width and height are within `max_image_dimension`.
    pub fn reserve_image_pixels(&mut self, width: u32, height: u32) -> Result<(), PackageIoError> {
        if width > self.budget.max_image_dimension {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetImageDimensions,
                format!(
                    "image width {} exceeds dimension limit {}",
                    width, self.budget.max_image_dimension
                ),
            ));
        }
        if height > self.budget.max_image_dimension {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetImageDimensions,
                format!(
                    "image height {} exceeds dimension limit {}",
                    height, self.budget.max_image_dimension
                ),
            ));
        }
        let pixels = (width as u64)
            .checked_mul(height as u64)
            .ok_or_else(|| overflow_err())?;
        let new_total = self
            .snapshot
            .image_pixels
            .checked_add(pixels)
            .ok_or_else(|| overflow_err())?;
        if new_total > self.budget.max_image_pixels {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetImagePixels,
                format!(
                    "image pixels {} would exceed limit {}",
                    new_total, self.budget.max_image_pixels
                ),
            ));
        }
        // Image pixels don't affect aggregate byte total — commit
        self.snapshot.image_pixels = new_total;
        Ok(())
    }

    /// Reserve data-URI decoded bytes. Fails on overflow or if the limit is exceeded.
    pub fn reserve_data_uri_bytes(&mut self, bytes: u64) -> Result<(), PackageIoError> {
        let new_total = self
            .snapshot
            .data_uri_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow_err())?;
        if new_total > self.budget.max_data_uri_bytes {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetDataUriDecode,
                format!(
                    "data-URI bytes {} would exceed limit {}",
                    new_total, self.budget.max_data_uri_bytes
                ),
            ));
        }
        self.check_aggregate_tentative(bytes, AggregateField::DataUriBytes)?;
        self.snapshot.data_uri_bytes = new_total;
        Ok(())
    }

    /// Check nesting depth. Fails if `depth` exceeds `max_nesting_depth`.
    pub fn check_nesting_depth(&self, depth: u32) -> Result<(), PackageIoError> {
        if depth > self.budget.max_nesting_depth {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetNestingDepth,
                format!(
                    "nesting depth {} exceeds limit {}",
                    depth, self.budget.max_nesting_depth
                ),
            ));
        }
        Ok(())
    }

    /// Reserve external model buffer bytes. Fails on overflow or if the limit is exceeded.
    pub fn reserve_external_model_buffer_bytes(
        &mut self,
        bytes: u64,
    ) -> Result<(), PackageIoError> {
        let new_total = self
            .snapshot
            .external_model_buffer_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow_err())?;
        if new_total > self.budget.max_external_model_bytes {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetExternalModel,
                format!(
                    "external model buffer bytes {} would exceed limit {}",
                    new_total, self.budget.max_external_model_bytes
                ),
            ));
        }
        self.check_aggregate_tentative(bytes, AggregateField::ExternalModelBufferBytes)?;
        self.snapshot.external_model_buffer_bytes = new_total;
        Ok(())
    }

    /// Reserve external model image bytes. Fails on overflow or if the limit is exceeded.
    pub fn reserve_external_model_image_bytes(&mut self, bytes: u64) -> Result<(), PackageIoError> {
        let new_total = self
            .snapshot
            .external_model_image_bytes
            .checked_add(bytes)
            .ok_or_else(|| overflow_err())?;
        if new_total > self.budget.max_external_model_image_bytes {
            return Err(PackageIoError::new(
                DiagnosticCode::PackageIoBudgetExternalModel,
                format!(
                    "external model image bytes {} would exceed limit {}",
                    new_total, self.budget.max_external_model_image_bytes
                ),
            ));
        }
        self.check_aggregate_tentative(bytes, AggregateField::ExternalModelImageBytes)?;
        self.snapshot.external_model_image_bytes = new_total;
        Ok(())
    }
}

// ── Helper ────────────────────────────────────────────────────────────────

/// Which aggregate field is being tentatively incremented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateField {
    SourceBytes,
    DecompressedBytes,
    DataUriBytes,
    ExternalModelBufferBytes,
    ExternalModelImageBytes,
}

fn overflow_err() -> PackageIoError {
    PackageIoError::new(
        DiagnosticCode::PackageIoBudgetOverflow,
        "arithmetic overflow",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ledger_starts_empty() {
        let ledger = BudgetLedger::default_ledger();
        let snap = ledger.snapshot();
        assert_eq!(snap.file_count, 0);
        assert_eq!(snap.source_bytes, 0);
    }

    #[test]
    fn reserve_file_count_succeeds() {
        let mut ledger = BudgetLedger::default_ledger();
        ledger.reserve_file_count(5).unwrap();
        assert_eq!(ledger.snapshot().file_count, 5);
    }

    #[test]
    fn reserve_file_count_fails_on_exceed() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_file_count: 10,
            ..Default::default()
        });
        ledger.reserve_file_count(10).unwrap();
        let err = ledger.reserve_file_count(1).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetFileCount);
        // Verify ledger unchanged
        assert_eq!(ledger.snapshot().file_count, 10);
    }

    #[test]
    fn reserve_file_count_on_overflow() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_file_count: u64::MAX,
            ..Default::default()
        });
        ledger.reserve_file_count(1).unwrap();
        let err = ledger.reserve_file_count(u64::MAX).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetOverflow);
        assert_eq!(ledger.snapshot().file_count, 1);
    }

    #[test]
    fn reserve_source_bytes_succeeds() {
        let mut ledger = BudgetLedger::default_ledger();
        ledger.reserve_source_bytes(1024).unwrap();
        assert_eq!(ledger.snapshot().source_bytes, 1024);
    }

    #[test]
    fn reserve_file_and_source_bytes_is_atomic() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_file_count: 2,
            max_source_bytes: 10,
            aggregate_package_bytes: u64::MAX,
            ..Default::default()
        });
        ledger.reserve_file_and_source_bytes(1, 8).unwrap();
        let err = ledger.reserve_file_and_source_bytes(1, 3).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetSourceBytes);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.file_count, 1);
        assert_eq!(snapshot.source_bytes, 8);
    }

    #[test]
    fn reserve_source_bytes_unchanged_on_failure() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_source_bytes: 100,
            aggregate_package_bytes: u64::MAX,
            ..Default::default()
        });
        let err = ledger.reserve_source_bytes(200).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetSourceBytes);
        assert_eq!(ledger.snapshot().source_bytes, 0);
    }

    #[test]
    fn reserve_image_pixels_rejects_large_dimensions() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_image_dimension: 1024,
            ..Default::default()
        });
        let err = ledger.reserve_image_pixels(2048, 16).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetImageDimensions);
        assert_eq!(ledger.snapshot().image_pixels, 0);
    }

    #[test]
    fn reserve_image_pixels_exceed_total() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_image_dimension: u32::MAX,
            max_image_pixels: 100,
            ..Default::default()
        });
        // 256 * 256 = 65536 > 100 max
        let err = ledger.reserve_image_pixels(256, 256).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetImagePixels);
        assert_eq!(ledger.snapshot().image_pixels, 0);
    }

    #[test]
    fn reserve_data_uri_bytes_exceed() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_data_uri_bytes: 512,
            ..Default::default()
        });
        let err = ledger.reserve_data_uri_bytes(1024).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetDataUriDecode);
        assert_eq!(ledger.snapshot().data_uri_bytes, 0);
    }

    #[test]
    fn check_nesting_depth_rejects_exceed() {
        let ledger = BudgetLedger::default_ledger();
        assert!(ledger.check_nesting_depth(10).is_ok());
        let err = ledger.check_nesting_depth(20).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetNestingDepth);
    }

    #[test]
    fn aggregate_package_limit_decompressed_bytes_unchanged_on_failure() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_source_bytes: u64::MAX,
            max_decompressed_bytes: u64::MAX,
            aggregate_package_bytes: 1000,
            ..Default::default()
        });
        ledger.reserve_source_bytes(600).unwrap();
        let err = ledger.reserve_decompressed_bytes(500).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetAggregateExceeded);
        // decompressed_bytes unchanged — ledgers is atomic
        assert_eq!(ledger.snapshot().decompressed_bytes, 0);
    }

    #[test]
    fn aggregate_package_limit_source_bytes_unchanged_on_failure() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_source_bytes: u64::MAX,
            max_decompressed_bytes: u64::MAX,
            aggregate_package_bytes: 1000,
            ..Default::default()
        });
        // Reserve decompressed first
        ledger.reserve_decompressed_bytes(600).unwrap();
        let err = ledger.reserve_source_bytes(500).unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetAggregateExceeded);
        // source_bytes unchanged
        assert_eq!(ledger.snapshot().source_bytes, 0);
    }

    #[test]
    fn external_model_buffer_exceed() {
        let mut ledger = BudgetLedger::new(ResourceBudget {
            max_external_model_bytes: 1000,
            ..Default::default()
        });
        let err = ledger
            .reserve_external_model_buffer_bytes(2000)
            .unwrap_err();
        assert_eq!(err.code, DiagnosticCode::PackageIoBudgetExternalModel);
        assert_eq!(ledger.snapshot().external_model_buffer_bytes, 0);
    }

    #[test]
    fn snapshot_is_read_only() {
        let mut ledger = BudgetLedger::default_ledger();
        ledger.reserve_source_bytes(100).unwrap();
        let snap = ledger.snapshot();
        assert_eq!(snap.source_bytes, 100);
        // Further mutations don't affect prior snapshots
        ledger.reserve_source_bytes(50).unwrap();
        assert_eq!(snap.source_bytes, 100); // unchanged
    }
}
