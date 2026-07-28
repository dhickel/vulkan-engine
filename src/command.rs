//! Undo/redo command primitives re-exported from the renderer crate.
//!
//! This module is a thin re-export layer. For custom commands or advanced
//! history manipulation, use `renderer` directly.

pub use renderer::{
    Command, CommandHistory, CommandResult, SceneNodeRemap,
    // Concrete commands
    AddNodeCommand, PlaceAssetCommand, RemoveNodeCommand, SetTransformCommand,
    AttachComponentCommand, DuplicateObjectsCommand, RemoveComponentCommand,
    RemoveObjectsCommand, ReplaceComponentStateCommand, SetComponentPropertyCommand,
    SetObjectParentCommand, SetObjectTransformCommand,
};
