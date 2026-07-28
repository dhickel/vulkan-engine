//! Bounded entity-string tokenizer and parser.
//!
//! Parses Quake entity strings per `bsp-compatibility.md` §5 with:
//! - Preserved source entity index, raw bytes, ordered key/value pairs
//! - Duplicate keys/ordinals, unknown keys, encoding diagnostics
//! - Typed singleton access with last-value-wins
//! - Classification: worldspawn, light, point, inline-brush, trigger, spawn, unknown
//! - Unknown/custom entities are never discarded

use crate::diagnostic::{BspReport, DiagnosticCode, SourceSpan};
use crate::limits::{MAX_ENTITY_COUNT, MAX_ENTITY_STRING_LENGTH};

/// Parsed entity with raw bytes preserved.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Index in the source entity string (0-based, in source order).
    pub source_index: u32,
    /// Raw entity bytes (including braces).
    pub raw: Vec<u8>,
    /// Parsed key/value pairs in source order.
    pub key_values: Vec<KeyValue>,
    /// Classification.
    pub class: EntityClass,
}

/// A single key/value pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
    /// Ordinal when key is duplicated (0 = first occurrence).
    pub ordinal: u32,
}

/// Recognized entity classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityClass {
    /// `worldspawn` — the world entity.
    Worldspawn,
    /// Any light entity (`light`, `light_fluoro`, etc.).
    Light,
    /// Generic point entity (non-worldspawn, non-brush-model).
    PointEntity,
    /// Inline brush model: `func_door`, `func_button`, `func_plat`, `func_wall`,
    /// `func_illusionary`, etc.
    InlineBrushModel,
    /// Trigger entity: `trigger_once`, `trigger_multiple`, `trigger_push`, etc.
    Trigger,
    /// Spawn marker: `info_player_start`, `info_player_deathmatch`,
    /// `info_teleport_destination`, etc.
    SpawnMarker,
    /// Unknown classname — preserved as-is.
    Unknown,
}

impl EntityClass {
    pub fn display_name(&self) -> &'static str {
        match self {
            EntityClass::Worldspawn => "worldspawn",
            EntityClass::Light => "light",
            EntityClass::PointEntity => "point",
            EntityClass::InlineBrushModel => "inline-brush",
            EntityClass::Trigger => "trigger",
            EntityClass::SpawnMarker => "spawn",
            EntityClass::Unknown => "unknown",
        }
    }
}

/// Classify an entity by its classname.
pub fn classify_entity(classname: &str) -> EntityClass {
    let lower = classname.to_ascii_lowercase();
    match lower.as_str() {
        "worldspawn" => EntityClass::Worldspawn,

        // Lights
        "light"
        | "light_fluoro"
        | "light_flame_large_yellow"
        | "light_torch_small_walltorch"
        | "light_spot"
        | "light_globe"
        | "light_flame_small_yellow"
        | "light_flame_small_white" => EntityClass::Light,

        // Inline brush models
        "func_door" | "func_button" | "func_plat" | "func_wall" | "func_illusionary"
        | "func_door_secret" | "func_train" | "func_rotate" | "func_pendulum" => {
            EntityClass::InlineBrushModel
        }

        // Triggers
        "trigger_once"
        | "trigger_multiple"
        | "trigger_push"
        | "trigger_teleport"
        | "trigger_changelevel"
        | "trigger_hurt"
        | "trigger_counter"
        | "trigger_relay"
        | "trigger_setskill" => EntityClass::Trigger,

        // Spawn markers
        "info_player_start"
        | "info_player_deathmatch"
        | "info_player_coop"
        | "info_teleport_destination" => EntityClass::SpawnMarker,

        _ => {
            // Check prefixes
            if lower.starts_with("light_") || lower.starts_with("light") {
                EntityClass::Light
            } else if lower.starts_with("func_") {
                EntityClass::InlineBrushModel
            } else if lower.starts_with("trigger_") {
                EntityClass::Trigger
            } else if lower.starts_with("info_player_") {
                EntityClass::SpawnMarker
            } else if lower.starts_with("info_") {
                EntityClass::PointEntity
            } else {
                EntityClass::Unknown
            }
        }
    }
}

/// Parse entity string bytes into a list of `Entity`.
///
/// Returns diagnostics for any issues found. Non-fatal issues (duplicate keys,
/// unknown classnames, empty entities) are diagnosed but don't prevent parsing.
/// Only structural corruption prevents successful parsing.
pub fn parse_entities(
    raw_data: &[u8],
    strict: bool,
) -> Result<(Vec<Entity>, Vec<BspReport>), BspReport> {
    let mut reports = Vec::new();

    if raw_data.len() > MAX_ENTITY_STRING_LENGTH as usize {
        return Err(BspReport::fatal(
            DiagnosticCode::EntityStringTooLarge,
            format!(
                "entity string {} bytes exceeds limit {}",
                raw_data.len(),
                MAX_ENTITY_STRING_LENGTH
            ),
        ));
    }

    // Try to interpret as UTF-8, diagnose non-UTF-8
    let text = match std::str::from_utf8(raw_data) {
        Ok(s) => s.to_string(),
        Err(_) => {
            // Latin-1 fallback: decode each byte individually
            reports.push(BspReport::new(
                DiagnosticCode::EntityClasslessWithKeys,
                strict,
                "entity string is not valid UTF-8; using Latin-1 fallback",
            ));
            raw_data.iter().map(|&b| b as char).collect()
        }
    };

    let entities = tokenize_and_parse(&text, strict, &mut reports)?;

    // Check entity count budget
    if entities.len() > MAX_ENTITY_COUNT as usize {
        return Err(BspReport::fatal(
            DiagnosticCode::EntityCountExceeded,
            format!(
                "entity count {} exceeds limit {}",
                entities.len(),
                MAX_ENTITY_COUNT
            ),
        ));
    }

    Ok((entities, reports))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenState {
    Outside,
    InEntity,
}

/// Tokenize and parse the entity string.
fn tokenize_and_parse(
    text: &str,
    strict: bool,
    reports: &mut Vec<BspReport>,
) -> Result<Vec<Entity>, BspReport> {
    let mut entities: Vec<Entity> = Vec::new();
    let mut state = TokenState::Outside;
    let mut entity_start: usize = 0;
    let mut current_key: Option<String> = None;
    let mut current_key_values: Vec<KeyValue> = Vec::new();
    let mut chars = text.char_indices().peekable();
    let mut in_quotes = false;
    let mut escape_next = false;
    let mut current_token = String::new();

    while let Some((pos, ch)) = chars.next() {
        match state {
            TokenState::Outside => {
                if ch == '{' {
                    state = TokenState::InEntity;
                    entity_start = pos;
                    current_key = None;
                    current_key_values.clear();
                    in_quotes = false;
                    escape_next = false;
                    current_token.clear();
                } else if ch == '}' {
                    return Err(BspReport::fatal(
                        DiagnosticCode::StructuralCorruptEntity,
                        format!("unexpected '}}' outside entity at position {}", pos),
                    )
                    .with_span(SourceSpan::ByteRange {
                        start: pos,
                        end: pos + 1,
                    }));
                } else if ch == '/' && chars.peek().map_or(false, |(_, c)| *c == '/') {
                    let _ = chars.next();
                    while let Some((_, comment_ch)) = chars.next() {
                        if comment_ch == '\n' {
                            break;
                        }
                    }
                } else if !(ch.is_whitespace() || ch == '\0') {
                    return Err(BspReport::fatal(
                        DiagnosticCode::EntityTokenUnquoted,
                        format!("unquoted token outside entity at position {}", pos),
                    )
                    .with_span(SourceSpan::ByteRange {
                        start: pos,
                        end: pos + ch.len_utf8(),
                    }));
                }
            }
            TokenState::InEntity => {
                if in_quotes {
                    if escape_next {
                        match ch {
                            'n' => current_token.push('\n'),
                            '"' => current_token.push('"'),
                            '\\' => current_token.push('\\'),
                            _ => {
                                current_token.push('\\');
                                current_token.push(ch);
                            }
                        }
                        escape_next = false;
                    } else if ch == '\\' {
                        escape_next = true;
                    } else if ch == '"' {
                        in_quotes = false;
                        let token = std::mem::take(&mut current_token);
                        if let Some(key) = current_key.take() {
                            let ordinal =
                                current_key_values.iter().filter(|kv| kv.key == key).count() as u32;
                            if ordinal > 0 {
                                reports.push(
                                    BspReport::new(
                                        DiagnosticCode::EntityDuplicateKey,
                                        strict,
                                        format!(
                                            "duplicate key '{}' in entity {} (ordinal {})",
                                            key,
                                            entities.len(),
                                            ordinal
                                        ),
                                    )
                                    .with_span(
                                        SourceSpan::Entity {
                                            index: entities.len(),
                                            key: None,
                                        },
                                    ),
                                );
                            }
                            current_key_values.push(KeyValue {
                                key,
                                value: token,
                                ordinal,
                            });
                        } else {
                            current_key = Some(token);
                        }
                    } else {
                        current_token.push(ch);
                    }
                } else if ch == '"' {
                    in_quotes = true;
                    escape_next = false;
                    current_token.clear();
                } else if ch == '{' {
                    return Err(BspReport::fatal(
                        DiagnosticCode::EntityNestedBraces,
                        format!(
                            "nested '{{' inside entity {} at position {}",
                            entities.len(),
                            pos
                        ),
                    )
                    .with_span(SourceSpan::Entity {
                        index: entities.len(),
                        key: None,
                    }));
                } else if ch == '}' {
                    if current_key.is_some() {
                        return Err(BspReport::fatal(
                            DiagnosticCode::EntityValueMissing,
                            format!(
                                "entity {} has key without value before closing brace",
                                entities.len()
                            ),
                        )
                        .with_span(SourceSpan::Entity {
                            index: entities.len(),
                            key: None,
                        }));
                    }
                    if entities.len() >= MAX_ENTITY_COUNT as usize {
                        return Err(BspReport::fatal(
                            DiagnosticCode::EntityCountExceeded,
                            format!("entity count exceeds limit {}", MAX_ENTITY_COUNT),
                        ));
                    }
                    let raw = text[entity_start..=pos].as_bytes().to_vec();
                    let classname = current_key_values
                        .iter()
                        .find(|kv| kv.key == "classname")
                        .map(|kv| kv.value.as_str())
                        .unwrap_or("");

                    let class = if classname.is_empty() {
                        if !current_key_values.is_empty() {
                            reports.push(
                                BspReport::new(
                                    DiagnosticCode::EntityClasslessWithKeys,
                                    strict,
                                    format!("entity {} has keys but no classname", entities.len()),
                                )
                                .with_span(SourceSpan::Entity {
                                    index: entities.len(),
                                    key: None,
                                }),
                            );
                        } else {
                            reports.push(
                                BspReport::new(
                                    DiagnosticCode::EntityEmpty,
                                    strict,
                                    format!("entity {} is empty", entities.len()),
                                )
                                .with_span(SourceSpan::Entity {
                                    index: entities.len(),
                                    key: None,
                                }),
                            );
                        }
                        EntityClass::Unknown
                    } else {
                        classify_entity(classname)
                    };

                    if class == EntityClass::Unknown && !classname.is_empty() {
                        reports.push(
                            BspReport::new(
                                DiagnosticCode::EntityUnknownClass,
                                strict,
                                format!(
                                    "entity {} has unknown classname '{}'",
                                    entities.len(),
                                    classname
                                ),
                            )
                            .with_span(SourceSpan::Entity {
                                index: entities.len(),
                                key: None,
                            }),
                        );
                    }

                    entities.push(Entity {
                        source_index: entities.len() as u32,
                        raw,
                        key_values: std::mem::take(&mut current_key_values),
                        class,
                    });
                    state = TokenState::Outside;
                } else if ch == '/' && chars.peek().map_or(false, |(_, c)| *c == '/') {
                    let _ = chars.next();
                    while let Some((_, comment_ch)) = chars.next() {
                        if comment_ch == '\n' {
                            break;
                        }
                    }
                } else if !(ch.is_whitespace() || ch == '\0') {
                    return Err(BspReport::fatal(
                        DiagnosticCode::EntityTokenUnquoted,
                        format!(
                            "unquoted token in entity {} at position {}",
                            entities.len(),
                            pos
                        ),
                    )
                    .with_span(SourceSpan::Entity {
                        index: entities.len(),
                        key: None,
                    }));
                }
            }
        }
    }

    if state == TokenState::InEntity || in_quotes || escape_next {
        return Err(BspReport::fatal(
            DiagnosticCode::EntityUnterminated,
            format!("entity {} is unterminated", entities.len()),
        )
        .with_span(SourceSpan::Entity {
            index: entities.len(),
            key: None,
        }));
    }

    Ok(entities)
}

/// Get a singleton value for a key (last-value-wins).
pub fn get_singleton<'a>(entity: &'a Entity, key: &str) -> Option<&'a str> {
    entity
        .key_values
        .iter()
        .rev()
        .find(|kv| kv.key == key)
        .map(|kv| kv.value.as_str())
}

/// Get all values for a key in source order.
pub fn get_all_values<'a>(entity: &'a Entity, key: &str) -> Vec<&'a str> {
    entity
        .key_values
        .iter()
        .filter(|kv| kv.key == key)
        .map(|kv| kv.value.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_entity() {
        let input = br#"{"classname" "worldspawn"}"#;
        let (entities, reports) = parse_entities(input, false).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].class, EntityClass::Worldspawn);
        assert_eq!(get_singleton(&entities[0], "classname"), Some("worldspawn"));
        assert!(reports
            .iter()
            .all(|r| r.code != DiagnosticCode::EntityDuplicateKey));
    }

    #[test]
    fn parse_multiple_entities() {
        let input = br#"{
"classname" "worldspawn"
}
{
"classname" "light"
"origin" "0 0 0"
}"#;
        let (entities, _reports) = parse_entities(input, false).unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].class, EntityClass::Worldspawn);
        assert_eq!(entities[1].class, EntityClass::Light);
        assert_eq!(get_singleton(&entities[1], "origin"), Some("0 0 0"));
    }

    #[test]
    fn parse_duplicate_keys_reported() {
        let input = br#"{"classname" "light" "light" "100" "light" "200"}"#;
        let (entities, reports) = parse_entities(input, false).unwrap();
        assert_eq!(entities.len(), 1);
        // Last value wins for singleton access
        assert_eq!(get_singleton(&entities[0], "light"), Some("200"));
        // Duplicate key diagnostic
        assert!(reports
            .iter()
            .any(|r| r.code == DiagnosticCode::EntityDuplicateKey));
    }

    #[test]
    fn parse_empty_entity() {
        let input = br#"{"classname" "worldspawn"} {} {"classname" "light"}"#;
        let (entities, reports) = parse_entities(input, false).unwrap();
        assert_eq!(entities.len(), 3);
        assert!(reports
            .iter()
            .any(|r| r.code == DiagnosticCode::EntityEmpty));
    }

    #[test]
    fn parse_entity_with_escape() {
        let input = br#"{"classname" "light" "message" "hello \"world\"\nline2"}"#;
        let (entities, _reports) = parse_entities(input, false).unwrap();
        assert_eq!(entities.len(), 1);
        let msg = get_singleton(&entities[0], "message").unwrap();
        assert!(msg.contains("world"));
        assert!(msg.contains("\nline2"));
    }

    #[test]
    fn parse_unterminated_entity_error() {
        let input = br#"{"classname" "worldspawn""#;
        let r = parse_entities(input, false);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, DiagnosticCode::EntityUnterminated);
    }

    #[test]
    fn parse_unknown_classname_reported() {
        let input = br#"{"classname" "my_custom_thing"}"#;
        let (entities, reports) = parse_entities(input, false).unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].class, EntityClass::Unknown);
        assert!(reports
            .iter()
            .any(|r| r.code == DiagnosticCode::EntityUnknownClass));
    }

    #[test]
    fn classify_all_types() {
        assert_eq!(classify_entity("worldspawn"), EntityClass::Worldspawn);
        assert_eq!(classify_entity("light"), EntityClass::Light);
        assert_eq!(classify_entity("light_fluoro"), EntityClass::Light);
        assert_eq!(classify_entity("func_door"), EntityClass::InlineBrushModel);
        assert_eq!(classify_entity("trigger_once"), EntityClass::Trigger);
        assert_eq!(
            classify_entity("info_player_start"),
            EntityClass::SpawnMarker
        );
        // ambient_generic is not recognized by the parser — classified as Unknown
        assert_eq!(classify_entity("ambient_generic"), EntityClass::Unknown);
        assert_eq!(classify_entity("totally_made_up"), EntityClass::Unknown);
    }

    #[test]
    fn get_all_values_preserves_order() {
        let entity = Entity {
            source_index: 0,
            raw: Vec::new(),
            key_values: vec![
                KeyValue {
                    key: "a".into(),
                    value: "1".into(),
                    ordinal: 0,
                },
                KeyValue {
                    key: "a".into(),
                    value: "2".into(),
                    ordinal: 1,
                },
                KeyValue {
                    key: "b".into(),
                    value: "3".into(),
                    ordinal: 0,
                },
            ],
            class: EntityClass::Unknown,
        };
        let vals = get_all_values(&entity, "a");
        assert_eq!(vals, vec!["1", "2"]);
    }
}
