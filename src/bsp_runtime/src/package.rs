//! Package-level BSP loading through the shared `package_io` trust boundary.
//!
//! This module uses [`PackageResolver`] and [`BudgetLedger`] from `package_io`
//! to authoritatively load BSP maps, palettes, `.lit` companions, WAD archives,
//! and replacement resources from a trusted package root. All path confinement,
//! budget enforcement, and hash verification is delegated to `package_io`.

use bsp::{BspLoader, BspReport, LoadOptions};
use package_io::resolver::PackageResolver;
use package_io::{
    ConfinedResource, ContentIdentity, ResourceKind,
};
use std::collections::HashMap;

/// Error type for package-level BSP loading failures.
#[derive(Debug)]
pub enum PackageLoadError {
    /// I/O or confinement error from `package_io`.
    Io(package_io::PackageIoError),
    /// BSP parse failure.
    Parse(BspReport),
    /// Missing required resource.
    MissingResource {
        kind: ResourceKind,
        path: String,
    },
    /// Hash mismatch for a required resource.
    HashMismatch {
        kind: ResourceKind,
        path: String,
        expected: String,
        actual: String,
    },
}

impl std::fmt::Display for PackageLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackageLoadError::Io(e) => write!(f, "package I/O error: {e}"),
            PackageLoadError::Parse(e) => write!(f, "BSP parse error: {e}"),
            PackageLoadError::MissingResource { kind, path } => {
                write!(f, "missing required {} resource: '{}'", kind.tag(), path)
            }
            PackageLoadError::HashMismatch {
                kind,
                path,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "hash mismatch for {} resource '{}': expected {}, got {}",
                    kind.tag(),
                    path,
                    expected,
                    actual
                )
            }
        }
    }
}

impl std::error::Error for PackageLoadError {}

impl From<package_io::PackageIoError> for PackageLoadError {
    fn from(e: package_io::PackageIoError) -> Self {
        PackageLoadError::Io(e)
    }
}

/// Result of loading all BSP-related resources from a package.
#[derive(Debug)]
pub struct LoadedBspPackage {
    /// The parsed BSP world.
    pub world: bsp::world::BspWorld,
    /// Authorized BSP bytes.
    pub bsp_resource: ConfinedResource,
    /// Optional palette resource (if loaded separately).
    pub palette_resource: Option<ConfinedResource>,
    /// Optional .lit companion resource.
    pub lit_resource: Option<ConfinedResource>,
    /// Loaded WAD archive bytes, keyed by archive name.
    pub wad_resources: Vec<(String, ConfinedResource)>,
    /// Auto-discovered `<texture>_norm.png` / `<texture>_gloss.png` resources.
    pub pbr_texture_resources: Vec<ConfinedResource>,
}

impl LoadedBspPackage {
    /// Convert confined PBR resources into the neutral extraction inputs.
    pub fn pbr_texture_companions(&self) -> Vec<bsp::resources::TextureCompanion> {
        self.pbr_texture_resources
            .iter()
            .map(|resource| {
                bsp::resources::TextureCompanion::new(
                    resource.id.as_str(),
                    resource.bytes.as_bytes().to_vec(),
                )
            })
            .collect()
    }
}

/// Discover supported external PBR companions under package-confined roots.
///
/// Candidate root precedence is: the BSP's directory, the game-root `textures/`
/// directory, package `textures/`, then the package root. Missing files are optional;
/// every non-missing resolver error remains fail-closed.
pub fn discover_package_pbr_texture_companions(
    resolver: &mut PackageResolver,
    bsp_path: &str,
    texture_names: &[String],
) -> Result<Vec<ConfinedResource>, PackageLoadError> {
    use package_io::DiagnosticCode;
    use std::collections::HashSet;
    use std::path::Path;

    let bsp_parent = Path::new(bsp_path)
        .parent()
        .and_then(Path::to_str)
        .filter(|path| !path.is_empty())
        .map(str::to_string);
    let game_root = Path::new(bsp_path)
        .parent()
        .and_then(Path::parent)
        .and_then(Path::to_str)
        .filter(|path| !path.is_empty())
        .map(str::to_string);

    let mut roots = Vec::new();
    if let Some(parent) = bsp_parent {
        roots.push(parent);
    }
    if let Some(root) = game_root {
        roots.push(format!("{root}/textures"));
    }
    roots.push("textures".to_string());
    roots.push(String::new());
    roots.dedup();

    let mut names = texture_names.to_vec();
    names.sort();
    names.dedup();
    let mut loaded_paths = HashSet::new();
    let mut resources = Vec::new();

    for texture_name in names {
        let Some(companion_names) = bsp::resources::pbr_companion_file_names(&texture_name) else {
            continue;
        };
        for filename in [companion_names.normal, companion_names.gloss] {
            let mut filename_variants = vec![filename.clone()];
            let lowercase = filename.to_ascii_lowercase();
            if lowercase != filename {
                filename_variants.push(lowercase);
            }

            'roots: for root in &roots {
                for filename in &filename_variants {
                    let candidate = if root.is_empty() {
                        filename.clone()
                    } else {
                        format!("{root}/{filename}")
                    };
                    if loaded_paths.contains(&candidate) {
                        break 'roots;
                    }
                    match resolver.resolve(&candidate, ResourceKind::Texture) {
                        Ok(resource) => {
                            loaded_paths.insert(candidate);
                            resources.push(resource);
                            break 'roots;
                        }
                        Err(error) if error.code == DiagnosticCode::PackageIoNotFound => {}
                        Err(error) => return Err(PackageLoadError::Io(error)),
                    }
                }
            }
        }
    }

    Ok(resources)
}

/// Load a BSP and its companions from a package resolver.
///
/// This is the primary entry point for runtime BSP loading. It:
/// 1. Uses the shared [`PackageResolver`] to load the BSP bytes
/// 2. Loads the palette
/// 3. Loads optional .lit companion if bound
/// 4. Loads configured WAD archives
/// 5. Builds [`LoadOptions`] and parses through the `bsp` crate
pub fn load_bsp_package(
    resolver: &mut PackageResolver,
    bsp_path: &str,
    palette_path: &str,
    lit_path: Option<&str>,
    wad_paths: &[String],
    strict: bool,
) -> Result<LoadedBspPackage, PackageLoadError> {
    // Load BSP
    let bsp_resource = resolver.resolve(bsp_path, ResourceKind::Bsp)?;

    // Load palette
    let palette_resource = resolver.resolve(palette_path, ResourceKind::Palette)?;

    // Load optional .lit
    let lit_resource = if let Some(lit_path) = lit_path {
        Some(resolver.resolve(lit_path, ResourceKind::Lit)?)
    } else {
        None
    };

    // Load WAD archives
    let mut wad_resources = Vec::new();
    for wad_path in wad_paths {
        let wad = resolver.resolve(wad_path, ResourceKind::Wad)?;
        // Use the basename as the archive key
        let name = std::path::Path::new(wad_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(wad_path)
            .to_string();
        wad_resources.push((name, wad));
    }

    // Build LoadOptions
    let options = LoadOptions {
        strict,
        palette: Some(palette_resource.bytes.as_bytes().to_vec()),
        lit_data: lit_resource
            .as_ref()
            .map(|r| r.bytes.as_bytes().to_vec()),
        wad_archives: wad_resources
            .iter()
            .map(|(name, r)| (name.clone(), r.bytes.as_bytes().to_vec()))
            .collect(),
        texture_overrides: Vec::new(),
        source_identity: bsp_path.to_string(),
    };

    // Parse BSP
    let world = BspLoader::load(bsp_resource.bytes.as_bytes(), &options)
        .map_err(PackageLoadError::Parse)?;
    let texture_names = bsp::resources::collect_miptex_names(&world.miptex_data);
    let pbr_texture_resources =
        discover_package_pbr_texture_companions(resolver, bsp_path, &texture_names)?;

    Ok(LoadedBspPackage {
        world,
        bsp_resource,
        palette_resource: Some(palette_resource),
        lit_resource,
        wad_resources,
        pbr_texture_resources,
    })
}

/// Verify declared hashes against loaded resources.
///
/// Checks that the content identity of each loaded resource matches
/// the expected hash declared in the package manifest.
pub fn verify_resource_hashes(
    expected: &HashMap<String, ContentIdentity>,
    loaded: &HashMap<String, &ConfinedResource>,
) -> Result<(), PackageLoadError> {
    for (path, expected_hash) in expected {
        match loaded.get(path.as_str()) {
            Some(resource) => {
                if resource.identity != *expected_hash {
                    return Err(PackageLoadError::HashMismatch {
                        kind: resource.id.kind(),
                        path: path.clone(),
                        expected: expected_hash.hex(),
                        actual: resource.identity.hex(),
                    });
                }
            }
            None => {
                return Err(PackageLoadError::MissingResource {
                    kind: ResourceKind::Generic,
                    path: path.clone(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use package_io::budget::BudgetLedger;
    use package_io::PackageRoot;
    use std::fs;

    fn temp_dir() -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "bsp-pkg-test-{}-{nanos}",
            std::process::id()
        ))
    }

    /// Build a minimal valid BSP29 file for testing.
    fn make_minimal_bsp29() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&29u32.to_le_bytes());

        let mut current_offset: u32 = 124;
        let entity_bytes = b"{\"classname\" \"worldspawn\"}\0";
        let entity_offset = current_offset;
        let entity_size = entity_bytes.len() as u32;
        current_offset += entity_size;
        let plane_offset = current_offset;
        let plane_size = 20u32;
        current_offset += plane_size;

        let lumps: [(u32, u32); 15] = [
            (entity_offset, entity_size),
            (plane_offset, plane_size),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
            (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
        ];
        for (off, sz) in &lumps {
            data.extend_from_slice(&off.to_le_bytes());
            data.extend_from_slice(&sz.to_le_bytes());
        }
        data.extend_from_slice(entity_bytes);
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0i32.to_le_bytes());
        data
    }

    #[test]
    fn load_bsp_package_from_temp_dir() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();

        let maps = dir.join("maps");
        fs::create_dir_all(&maps).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();

        let palettes = dir.join("palettes");
        fs::create_dir_all(&palettes).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let result = load_bsp_package(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &[],
            false,
        );
        assert!(result.is_ok());
        let pkg = result.unwrap();
        assert_eq!(pkg.world.entities.len(), 1);
        assert!(pkg.palette_resource.is_some());
        let mut coordinator = crate::coordinator::BspCoordinator::new();
        assert!(coordinator
            .prepare_from_loaded_package(pkg, Some(0.0254))
            .is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_bsp_package_missing_bsp() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir_all(dir.join("palettes")).unwrap();
        fs::write(dir.join("palettes/pal.lmp"), &[0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let result = load_bsp_package(
            &mut resolver,
            "maps/test.bsp",
            "palettes/pal.lmp",
            None,
            &[],
            false,
        );
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pbr_companions_are_discovered_in_texture_root() {
        let dir = temp_dir();
        fs::create_dir_all(dir.join("textures")).unwrap();
        fs::write(dir.join("textures/brick1_2_norm.png"), b"normal").unwrap();
        fs::write(dir.join("textures/brick1_2_gloss.png"), b"gloss").unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);
        let resources = discover_package_pbr_texture_companions(
            &mut resolver,
            "maps/test.bsp",
            &["brick1_2".to_string()],
        )
        .unwrap();
        assert_eq!(resources.len(), 2);
        let companions = resources
            .iter()
            .map(|resource| {
                bsp::resources::TextureCompanion::new(
                    resource.id.as_str(),
                    resource.bytes.as_bytes().to_vec(),
                )
            })
            .collect::<Vec<_>>();
        let found = bsp::resources::discover_pbr_texture_companions("brick1_2", &companions);
        assert!(found.normal.is_some());
        assert!(found.gloss.is_some());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_hashes_match() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let maps = dir.join("maps");
        fs::create_dir_all(&maps).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        let palettes = dir.join("palettes");
        fs::create_dir_all(&palettes).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let bsp = resolver.resolve("maps/test.bsp", ResourceKind::Bsp).unwrap();
        let bsp_id = bsp.identity;

        let mut expected = HashMap::new();
        expected.insert("maps/test.bsp".to_string(), bsp_id);

        let mut loaded = HashMap::new();
        loaded.insert("maps/test.bsp".to_string(), &bsp);

        assert!(verify_resource_hashes(&expected, &loaded).is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_hashes_mismatch() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let maps = dir.join("maps");
        fs::create_dir_all(&maps).unwrap();
        fs::write(maps.join("test.bsp"), make_minimal_bsp29()).unwrap();
        let palettes = dir.join("palettes");
        fs::create_dir_all(&palettes).unwrap();
        fs::write(palettes.join("pal.lmp"), &[0u8; 768]).unwrap();

        let root = PackageRoot::new(&dir).unwrap();
        let ledger = BudgetLedger::default_ledger();
        let mut resolver = PackageResolver::new(root, ledger);

        let bsp = resolver.resolve("maps/test.bsp", ResourceKind::Bsp).unwrap();
        let wrong_id = ContentIdentity::from_bytes(b"wrong");

        let mut expected = HashMap::new();
        expected.insert("maps/test.bsp".to_string(), wrong_id);

        let mut loaded = HashMap::new();
        loaded.insert("maps/test.bsp".to_string(), &bsp);

        let err = verify_resource_hashes(&expected, &loaded).unwrap_err();
        assert!(matches!(err, PackageLoadError::HashMismatch { .. }));

        let _ = fs::remove_dir_all(&dir);
    }
}
