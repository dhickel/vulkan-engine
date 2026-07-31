//! Stable diagnostic codes and report types for BSP parsing.
//!
//! Every diagnostic carries a stable machine-readable code (not only message text).
//! Codes include the `BSP-*` identifiers in `bsp-compatibility.md` §7 plus typed
//! limit/entity/resource diagnostics named by other compatibility tables.

use core::fmt;

/// Severity of a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
        }
    }
}

/// Stable diagnostic category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticCategory {
    UnsupportedCompatibility,
    StructuralCorruption,
    Security,
    MissingRequired,
    OptionalFallback,
    UnknownAppEntity,
    AuthoringQuality,
}

/// Machine-readable diagnostic code — the stable identifier for every diagnostic.
///
/// Variants cover the stable compatibility table and the typed diagnostics named by
/// limit/entity/resource rules. Test assertions must match on code and severity,
/// not on message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticCode {
    // ── Unsupported compatibility ──
    /// Magic/version not in approved profile.
    UnsupportedDialect,
    /// Unknown BSPX extension name.
    UnsupportedExtension,
    /// Conflicting valid extensions.
    AmbiguousExtension,
    /// Companion file version unsupported.
    CompanionVersion,
    /// Companion data mismatch (e.g., luxel count).
    CompanionContentMismatch,
    /// Companion file exists but hash does not match expected.
    StaleCompanion,

    // ── Structural corruption ──
    /// Lump offset/size invalid, overlap, or truncation.
    StructuralCorruptLump,
    /// Cross-lump index out of valid range.
    StructuralCorruptIndex,
    /// Cyclic tree/leaf graph.
    StructuralCorruptCycle,
    /// Integer overflow in count/layout arithmetic.
    StructuralCorruptOverflow,
    /// Required alignment violated.
    StructuralCorruptAlignment,
    /// Entity string structurally malformed.
    StructuralCorruptEntity,
    /// Face winding/plane/texinfo invalid.
    StructuralCorruptFace,
    /// Vertex count for profile exceeded.
    StructuralVertexCount,
    /// Edge count for profile exceeded.
    StructuralEdgeCount,
    /// Surfedge count for profile exceeded.
    StructuralSurfedgeCount,
    /// Face count for profile exceeded.
    StructuralFaceCount,
    /// Markface count for profile exceeded.
    StructuralMarkfaceCount,
    /// Node count for profile exceeded.
    StructuralNodeCount,
    /// Leaf count for profile exceeded.
    StructuralLeafCount,
    /// Clipnode count for profile exceeded.
    StructuralClipnodeCount,
    /// Model count for profile exceeded.
    StructuralModelCount,
    /// Entity string too large.
    EntityStringTooLarge,
    /// Light style slot exceeds supported count.
    UnsupportedStyleSlot,

    // ── Security ──
    /// Path escape attempt in resource reference.
    SecurityPathTraversal,
    /// Symlink escape in package root.
    SecuritySymlinkEscape,
    /// Non-regular file at expected resource path.
    SecurityDeviceFile,

    // ── Missing required ──
    /// No palette available.
    MissingRequiredPalette,
    /// Referenced WAD not found in configured roots.
    MissingRequiredWad,
    /// External model not found in release mappings.
    MissingRequiredModel,
    /// Face has no valid lightmap data.
    MissingRequiredLightmap,

    // ── Optional fallback ──
    /// Using project palette fallback.
    FallbackDefaultPalette,
    /// Using embedded miptex for texture.
    FallbackEmbeddedMiptex,
    /// Using diagnostic checkerboard for missing texture.
    FallbackDiagnosticTexture,
    /// Face missing lightmap, using fullbright fallback.
    FallbackMissingLightmap,

    // ── Unknown app entity ──
    /// Entity classname not recognized by engine.
    EntityUnknownClass,

    // ── Authoring quality ──
    /// Duplicate key in entity.
    EntityDuplicateKey,
    /// Empty entity `{}`.
    EntityEmpty,
    /// Entity token is unquoted.
    EntityTokenUnquoted,
    /// Entity is unterminated.
    EntityUnterminated,
    /// Entity has nested braces.
    EntityNestedBraces,
    /// Entity has no classname but has keys.
    EntityClasslessWithKeys,
    /// Entity key without a value.
    EntityValueMissing,
    /// Allocation budget exceeded.
    AllocationExceeded,
    /// Entity count budget exceeded.
    EntityCountExceeded,
    /// Texture count budget exceeded.
    TextureCountExceeded,
    /// WAD entry count budget exceeded.
    WadEntryCountExceeded,
    /// Surface edge sign is invalid (zero in signed surfedge where ambiguous).
    SurfedgeSignInvalid,
    /// BSPX lump overlaps a standard lump.
    BspxLumpOverlap,
    /// Duplicate BSPX extension name.
    BspxDuplicateName,
    /// Conflicting colored light sources.
    ColoredLightConflict,

    // ── Extraction diagnostics ──
    /// Miptex data corrupt: offset/size/dimensions invalid.
    MiptexCorrupt,
    /// Texture pixel allocation exceeds budget.
    TextureAllocationExceeded,
    /// Animation frame sequence has a gap.
    AnimationSequenceGap,
    /// Animation frames have inconsistent dimensions.
    AnimationDimensionMismatch,
    /// Animation frames have colliding names under case-insensitive comparison.
    AnimationCaseCollision,
    /// Atlas page overflow: luxel block cannot be placed.
    AtlasPageOverflow,
    /// Lightmap style data is truncated or missing.
    LightmapStyleTruncated,
    /// Extraction invariant violated: parallel arrays with mismatched lengths.
    ExtractionInvariantViolation,
    /// Entity model reference (*N) is out of bounds.
    EntityModelOutOfBounds,
    /// Convex reconstruction failed for a brush entity hull.
    ConvexReconstructionFailed,
    /// World collision data is missing; whole-asset rejection.
    MissingCollisionData,
    /// Identity ambiguity: multiple entities share the same fingerprint.
    IdentityAmbiguous,
}

impl DiagnosticCode {
    /// The diagnostic category for this code.
    pub fn category(self) -> DiagnosticCategory {
        use DiagnosticCategory::*;
        use DiagnosticCode::*;
        match self {
            UnsupportedDialect
            | UnsupportedExtension
            | AmbiguousExtension
            | CompanionVersion
            | CompanionContentMismatch
            | StaleCompanion => UnsupportedCompatibility,
            StructuralCorruptLump
            | StructuralCorruptIndex
            | StructuralCorruptCycle
            | StructuralCorruptOverflow
            | StructuralCorruptAlignment
            | StructuralCorruptEntity
            | StructuralCorruptFace
            | StructuralVertexCount
            | StructuralEdgeCount
            | StructuralSurfedgeCount
            | StructuralFaceCount
            | StructuralMarkfaceCount
            | StructuralNodeCount
            | StructuralLeafCount
            | StructuralClipnodeCount
            | StructuralModelCount
            | EntityStringTooLarge
            | UnsupportedStyleSlot
            | SurfedgeSignInvalid => StructuralCorruption,
            SecurityPathTraversal | SecuritySymlinkEscape | SecurityDeviceFile => Security,
            MissingRequiredPalette
            | MissingRequiredWad
            | MissingRequiredModel
            | MissingRequiredLightmap => MissingRequired,
            FallbackDefaultPalette
            | FallbackEmbeddedMiptex
            | FallbackDiagnosticTexture
            | FallbackMissingLightmap => OptionalFallback,
            EntityUnknownClass => UnknownAppEntity,
            EntityTokenUnquoted | EntityUnterminated | EntityNestedBraces | EntityValueMissing => {
                StructuralCorruption
            }
            EntityDuplicateKey | EntityEmpty | EntityClasslessWithKeys => AuthoringQuality,
            AllocationExceeded
            | EntityCountExceeded
            | TextureCountExceeded
            | WadEntryCountExceeded => StructuralCorruption,
            BspxLumpOverlap | BspxDuplicateName => StructuralCorruption,
            ColoredLightConflict => UnsupportedCompatibility,

            MiptexCorrupt
            | TextureAllocationExceeded
            | AtlasPageOverflow
            | LightmapStyleTruncated
            | ExtractionInvariantViolation
            | EntityModelOutOfBounds
            | ConvexReconstructionFailed
            | MissingCollisionData => StructuralCorruption,
            AnimationSequenceGap | AnimationDimensionMismatch | AnimationCaseCollision => {
                AuthoringQuality
            }
            IdentityAmbiguous => AuthoringQuality,
        }
    }

    /// Severity in development mode (strict = false).
    pub fn dev_severity(self) -> Severity {
        use DiagnosticCode::*;
        use Severity::*;
        match self {
            // Structural — always Error
            StructuralCorruptLump
            | StructuralCorruptIndex
            | StructuralCorruptCycle
            | StructuralCorruptOverflow
            | StructuralCorruptAlignment
            | StructuralCorruptEntity
            | StructuralCorruptFace
            | StructuralVertexCount
            | StructuralEdgeCount
            | StructuralSurfedgeCount
            | StructuralFaceCount
            | StructuralMarkfaceCount
            | StructuralNodeCount
            | StructuralLeafCount
            | StructuralClipnodeCount
            | StructuralModelCount
            | EntityStringTooLarge
            | UnsupportedStyleSlot
            | SurfedgeSignInvalid
            | BspxLumpOverlap
            | BspxDuplicateName
            | AllocationExceeded
            | EntityCountExceeded
            | TextureCountExceeded
            | WadEntryCountExceeded
            | MiptexCorrupt
            | TextureAllocationExceeded
            | AtlasPageOverflow
            | LightmapStyleTruncated
            | ExtractionInvariantViolation
            | EntityModelOutOfBounds
            | MissingCollisionData => Error,
            ConvexReconstructionFailed => Warning,

            // Security — always Error
            SecurityPathTraversal | SecuritySymlinkEscape | SecurityDeviceFile => Error,

            // Missing required — palette and WAD are always required for rendering
            MissingRequiredPalette | MissingRequiredWad => Error,
            MissingRequiredLightmap => Warning,
            MissingRequiredModel => Warning,

            // Unsupported compatibility — Warning in dev
            UnsupportedDialect => Error,
            UnsupportedExtension
            | AmbiguousExtension
            | CompanionVersion
            | CompanionContentMismatch
            | StaleCompanion
            | ColoredLightConflict => Warning,

            // Optional fallback — Warning
            FallbackDefaultPalette
            | FallbackEmbeddedMiptex
            | FallbackDiagnosticTexture
            | FallbackMissingLightmap => Warning,

            // Unknown app entity — Info
            EntityUnknownClass => Info,

            // Entity grammar structural errors — always Error
            EntityTokenUnquoted | EntityUnterminated | EntityNestedBraces | EntityValueMissing => {
                Error
            }

            // Authoring quality
            EntityClasslessWithKeys => Warning,
            EntityDuplicateKey | EntityEmpty => Info,
            AnimationSequenceGap | AnimationDimensionMismatch | AnimationCaseCollision => Warning,
            IdentityAmbiguous => Info,
        }
    }

    /// Severity in strict/release mode (strict = true).
    pub fn strict_severity(self) -> Severity {
        use DiagnosticCode::*;
        use Severity::*;
        match self {
            StructuralCorruptLump
            | StructuralCorruptIndex
            | StructuralCorruptCycle
            | StructuralCorruptOverflow
            | StructuralCorruptAlignment
            | StructuralCorruptEntity
            | StructuralCorruptFace
            | StructuralVertexCount
            | StructuralEdgeCount
            | StructuralSurfedgeCount
            | StructuralFaceCount
            | StructuralMarkfaceCount
            | StructuralNodeCount
            | StructuralLeafCount
            | StructuralClipnodeCount
            | StructuralModelCount
            | EntityStringTooLarge
            | UnsupportedStyleSlot
            | SurfedgeSignInvalid
            | BspxLumpOverlap
            | BspxDuplicateName
            | AllocationExceeded
            | EntityCountExceeded
            | TextureCountExceeded
            | WadEntryCountExceeded
            | MiptexCorrupt
            | TextureAllocationExceeded
            | AtlasPageOverflow
            | LightmapStyleTruncated
            | ExtractionInvariantViolation
            | EntityModelOutOfBounds
            | ConvexReconstructionFailed
            | MissingCollisionData => Error,

            SecurityPathTraversal | SecuritySymlinkEscape | SecurityDeviceFile => Error,

            // Missing required — all Error in strict
            MissingRequiredPalette
            | MissingRequiredWad
            | MissingRequiredModel
            | MissingRequiredLightmap => Error,

            // Unsupported compatibility — Error in strict
            UnsupportedDialect
            | UnsupportedExtension
            | AmbiguousExtension
            | CompanionVersion
            | CompanionContentMismatch
            | StaleCompanion
            | ColoredLightConflict => Error,

            // Optional fallback — Warning
            FallbackDefaultPalette
            | FallbackEmbeddedMiptex
            | FallbackDiagnosticTexture
            | FallbackMissingLightmap => Warning,

            // Unknown app entity — Info
            EntityUnknownClass => Info,

            // Entity grammar structural errors — always Error
            EntityTokenUnquoted | EntityUnterminated | EntityNestedBraces | EntityValueMissing => {
                Error
            }

            // Authoring quality — Warning in strict
            EntityDuplicateKey | EntityEmpty | EntityClasslessWithKeys => Warning,
            AnimationSequenceGap | AnimationDimensionMismatch | AnimationCaseCollision => Warning,
            IdentityAmbiguous => Warning,
        }
    }

    /// Select severity based on strictness policy.
    pub fn severity(self, strict: bool) -> Severity {
        if strict {
            self.strict_severity()
        } else {
            self.dev_severity()
        }
    }

    /// Whether this diagnostic is fatal (prevents BspWorld construction).
    pub fn is_fatal(self, strict: bool) -> bool {
        self.severity(strict) == Severity::Error
    }

    /// The machine-readable code string, e.g. `"BSP-STRUCT-CORRUPT-LUMP"`.
    pub fn code_str(self) -> &'static str {
        use DiagnosticCode::*;
        match self {
            UnsupportedDialect => "BSP-UNSUPPORTED-DIALECT",
            UnsupportedExtension => "BSP-COMPAT-UNSUPPORTED-EXT",
            AmbiguousExtension => "BSP-COMPAT-AMBIGUOUS-EXT",
            CompanionVersion => "BSP-COMPAT-COMPANION-VERSION",
            CompanionContentMismatch => "BSP-COMPAT-COMPANION-MISMATCH",
            StaleCompanion => "BSP-COMPAT-STALE-COMPANION",
            StructuralCorruptLump => "BSP-STRUCT-CORRUPT-LUMP",
            StructuralCorruptIndex => "BSP-STRUCT-CORRUPT-INDEX",
            StructuralCorruptCycle => "BSP-STRUCT-CORRUPT-CYCLE",
            StructuralCorruptOverflow => "BSP-STRUCT-CORRUPT-OVERFLOW",
            StructuralCorruptAlignment => "BSP-STRUCT-CORRUPT-ALIGNMENT",
            StructuralCorruptEntity => "BSP-STRUCT-CORRUPT-ENTITY",
            StructuralCorruptFace => "BSP-STRUCT-CORRUPT-FACE",
            StructuralVertexCount => "BSP-STRUCT-VERTEX-COUNT",
            StructuralEdgeCount => "BSP-STRUCT-EDGE-COUNT",
            StructuralSurfedgeCount => "BSP-STRUCT-SURFEDGE-COUNT",
            StructuralFaceCount => "BSP-STRUCT-FACE-COUNT",
            StructuralMarkfaceCount => "BSP-STRUCT-MARKFACE-COUNT",
            StructuralNodeCount => "BSP-STRUCT-NODE-COUNT",
            StructuralLeafCount => "BSP-STRUCT-LEAF-COUNT",
            StructuralClipnodeCount => "BSP-STRUCT-CLIPNODE-COUNT",
            StructuralModelCount => "BSP-STRUCT-MODEL-COUNT",
            EntityStringTooLarge => "BSP-STRUCT-ENTITY-TOO-LARGE",
            UnsupportedStyleSlot => "BSP-STRUCT-STYLE-SLOT",
            SurfedgeSignInvalid => "BSP-STRUCT-SURFEDGE-INVALID",
            BspxLumpOverlap => "BSP-STRUCT-BSPX-OVERLAP",
            BspxDuplicateName => "BSP-STRUCT-BSPX-DUPLICATE",
            SecurityPathTraversal => "BSP-SECURITY-PATH-TRAVERSAL",
            SecuritySymlinkEscape => "BSP-SECURITY-SYMLINK-ESCAPE",
            SecurityDeviceFile => "BSP-SECURITY-DEVICE-FILE",
            MissingRequiredPalette => "BSP-MISSING-REQUIRED-PALETTE",
            MissingRequiredWad => "BSP-MISSING-REQUIRED-WAD",
            MissingRequiredModel => "BSP-MISSING-REQUIRED-MODEL",
            MissingRequiredLightmap => "BSP-MISSING-REQUIRED-LIGHTMAP",
            FallbackDefaultPalette => "BSP-FALLBACK-DEFAULT-PALETTE",
            FallbackEmbeddedMiptex => "BSP-FALLBACK-EMBEDDED-MIPTEX",
            FallbackDiagnosticTexture => "BSP-FALLBACK-DIAGNOSTIC-TEXTURE",
            FallbackMissingLightmap => "BSP-FALLBACK-MISSING-LIGHTMAP",
            EntityUnknownClass => "BSP-ENTITY-UNKNOWN-CLASS",
            EntityDuplicateKey => "BSP-ENTITY-DUPLICATE-KEY",
            EntityEmpty => "BSP-ENTITY-EMPTY",
            EntityTokenUnquoted => "BSP-ENTITY-TOKEN-UNQUOTED",
            EntityUnterminated => "BSP-ENTITY-UNTERMINATED",
            EntityNestedBraces => "BSP-ENTITY-NESTED-BRACES",
            EntityClasslessWithKeys => "BSP-ENTITY-CLASSLESS-WITH-KEYS",
            EntityValueMissing => "BSP-ENTITY-VALUE-MISSING",
            AllocationExceeded => "BSP-STRUCT-ALLOC-EXCEEDED",
            EntityCountExceeded => "BSP-STRUCT-ENTITY-COUNT",
            TextureCountExceeded => "BSP-STRUCT-TEXTURE-COUNT",
            WadEntryCountExceeded => "BSP-STRUCT-WAD-ENTRY-COUNT",
            ColoredLightConflict => "BSP-COMPAT-COLORED-LIGHT-CONFLICT",
            MiptexCorrupt => "BSP-STRUCT-MIPTEX-CORRUPT",
            TextureAllocationExceeded => "BSP-STRUCT-TEXTURE-ALLOC",
            AnimationSequenceGap => "BSP-ANIM-SEQUENCE-GAP",
            AnimationDimensionMismatch => "BSP-ANIM-DIM-MISMATCH",
            AnimationCaseCollision => "BSP-ANIM-CASE-COLLISION",
            AtlasPageOverflow => "BSP-STRUCT-ATLAS-OVERFLOW",
            LightmapStyleTruncated => "BSP-STRUCT-STYLE-TRUNCATED",
            ExtractionInvariantViolation => "BSP-STRUCT-INVARIANT",
            EntityModelOutOfBounds => "BSP-STRUCT-ENTITY-MODEL-OOB",
            ConvexReconstructionFailed => "BSP-STRUCT-CONVEX-FAILED",
            MissingCollisionData => "BSP-STRUCT-MISSING-COLLISION",
            IdentityAmbiguous => "BSP-IDENTITY-AMBIGUOUS",
        }
    }
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code_str())
    }
}

/// Contextual span or byte range for a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSpan {
    /// Byte range in the raw BSP file.
    ByteRange { start: usize, end: usize },
    /// Lump index with optional offset within the lump.
    Lump { index: usize, offset: Option<usize> },
    /// Entity index and optional key.
    Entity {
        index: usize,
        key: Option<&'static str>,
    },
    /// BSPX extension name.
    BspxLump { name: &'static str },
    /// Companion file reference.
    Companion { kind: &'static str },
    /// No specific span.
    None,
}

impl Default for SourceSpan {
    fn default() -> Self {
        SourceSpan::None
    }
}

/// A single diagnostic report produced during parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BspReport {
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub span: SourceSpan,
    pub message: String,
}

impl BspReport {
    /// Create a new diagnostic report.
    pub fn new(code: DiagnosticCode, strict: bool, message: impl Into<String>) -> Self {
        let severity = code.severity(strict);
        BspReport {
            code,
            severity,
            span: SourceSpan::None,
            message: message.into(),
        }
    }

    /// Attach a source span.
    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = span;
        self
    }

    /// Create a fatal structural error.
    pub fn fatal(code: DiagnosticCode, message: impl Into<String>) -> Self {
        BspReport {
            code,
            severity: Severity::Error,
            span: SourceSpan::None,
            message: message.into(),
        }
    }

    /// Whether this report is an error (fatal).
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

impl fmt::Display for BspReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.code, self.message)
    }
}

impl std::error::Error for BspReport {}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_codes_have_stable_strings() {
        assert_eq!(
            DiagnosticCode::UnsupportedDialect.code_str(),
            "BSP-UNSUPPORTED-DIALECT"
        );
        assert_eq!(
            DiagnosticCode::StructuralCorruptLump.code_str(),
            "BSP-STRUCT-CORRUPT-LUMP"
        );
        assert_eq!(
            DiagnosticCode::SecurityPathTraversal.code_str(),
            "BSP-SECURITY-PATH-TRAVERSAL"
        );
        assert_eq!(
            DiagnosticCode::MissingRequiredPalette.code_str(),
            "BSP-MISSING-REQUIRED-PALETTE"
        );
        assert_eq!(
            DiagnosticCode::EntityUnknownClass.code_str(),
            "BSP-ENTITY-UNKNOWN-CLASS"
        );
        assert_eq!(
            DiagnosticCode::EntityDuplicateKey.code_str(),
            "BSP-ENTITY-DUPLICATE-KEY"
        );
    }

    #[test]
    fn severity_policy_dev_mode() {
        // Structural is always Error
        assert_eq!(
            DiagnosticCode::StructuralCorruptLump.dev_severity(),
            Severity::Error
        );
        // Missing palette is Error even in dev
        assert_eq!(
            DiagnosticCode::MissingRequiredPalette.dev_severity(),
            Severity::Error
        );
        // Unsupported extension is Warning in dev
        assert_eq!(
            DiagnosticCode::UnsupportedExtension.dev_severity(),
            Severity::Warning
        );
        // Unknown entity is Info
        assert_eq!(
            DiagnosticCode::EntityUnknownClass.dev_severity(),
            Severity::Info
        );
        // Authoring quality is Info in dev
        assert_eq!(
            DiagnosticCode::EntityDuplicateKey.dev_severity(),
            Severity::Info
        );
    }

    #[test]
    fn severity_policy_strict_mode() {
        // Unsupported extension is Error in strict
        assert_eq!(
            DiagnosticCode::UnsupportedExtension.strict_severity(),
            Severity::Error
        );
        // Authoring quality is Warning in strict
        assert_eq!(
            DiagnosticCode::EntityDuplicateKey.strict_severity(),
            Severity::Warning
        );
        // Missing model is Error in strict (was Warning in dev)
        assert_eq!(
            DiagnosticCode::MissingRequiredModel.strict_severity(),
            Severity::Error
        );
    }

    #[test]
    fn diagnostic_categories() {
        assert_eq!(
            DiagnosticCode::StructuralCorruptLump.category(),
            DiagnosticCategory::StructuralCorruption
        );
        assert_eq!(
            DiagnosticCode::UnsupportedDialect.category(),
            DiagnosticCategory::UnsupportedCompatibility
        );
        assert_eq!(
            DiagnosticCode::SecurityPathTraversal.category(),
            DiagnosticCategory::Security
        );
    }

    #[test]
    fn bsp_report_display_includes_code() {
        let r = BspReport::fatal(DiagnosticCode::UnsupportedDialect, "bad magic 42");
        let s = format!("{}", r);
        assert!(s.contains("BSP-UNSUPPORTED-DIALECT"));
        assert!(s.contains("bad magic 42"));
    }
}
