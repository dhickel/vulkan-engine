use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::data::handles::TextureHandle;

use super::errors::{HookError, HookFailureEntry};

pub struct RenderHookContext<'a> {
    pub frame_index: u64,
    pub viewport_size: (u32, u32),
    /// Depth buffer from the current frame, if available.
    /// Available after PrepareTargetsPass has executed.
    pub depth_texture: Option<TextureHandle>,
    // Prevent external construction.
    _private: PhantomData<&'a mut ()>,
}

impl<'a> RenderHookContext<'a> {
    pub(crate) fn new(
        frame_index: u64,
        viewport_size: (u32, u32),
        depth_texture: Option<TextureHandle>,
    ) -> Self {
        Self {
            frame_index,
            viewport_size,
            depth_texture,
            _private: PhantomData,
        }
    }
}

/// Trait for render hook callbacks.
///
/// Implement this trait on your own types for named, testable hooks.
/// For closures, use [`boxed_render_hook`] to wrap them.
pub trait RenderHook: Send {
    fn invoke(&mut self, ctx: &mut RenderHookContext<'_>) -> Result<(), HookError>;
}

/// Stored form of a render hook.
pub type BoxedRenderHook = Box<dyn RenderHook>;

/// Create a `BoxedRenderHook` from a closure.
///
/// This is the primary way to construct render hook callbacks from closures.
/// Named structs that implement [`RenderHook`] can be boxed directly via
/// `Box::new(...)`.
pub fn boxed_render_hook<F>(f: F) -> BoxedRenderHook
where
    F: FnMut(&mut RenderHookContext<'_>) -> Result<(), HookError> + Send + 'static,
{
    Box::new(FnHook(Box::new(f)))
}

/// Hidden wrapper that adapts FnMut closures to the RenderHook trait.
/// This works around the HRTB limitation of the blanket impl approach.
struct FnHook(Box<dyn FnMut(&mut RenderHookContext<'_>) -> Result<(), HookError> + Send>);

impl RenderHook for FnHook {
    fn invoke(&mut self, ctx: &mut RenderHookContext<'_>) -> Result<(), HookError> {
        (self.0)(ctx)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum RenderHookStage {
    PreRender,
    PostRender,
}

impl RenderHookStage {
    fn label(self) -> &'static str {
        match self {
            Self::PreRender => "pre_render",
            Self::PostRender => "post_render",
        }
    }
}

pub(crate) fn invoke_render_hook(
    hook: &mut Option<BoxedRenderHook>,
    stage: RenderHookStage,
    frame_index: u64,
    viewport_size: (u32, u32),
    depth_texture: Option<TextureHandle>,
) -> (Result<(), HookError>, Option<HookFailureEntry>) {
    let Some(hook) = hook.as_mut() else {
        return (Ok(()), None);
    };

    let mut context = RenderHookContext::new(frame_index, viewport_size, depth_texture);
    let callback_result = catch_unwind(AssertUnwindSafe(|| hook.invoke(&mut context)));

    match callback_result {
        Ok(Ok(())) => (Ok(()), None),
        Ok(Err(err)) => {
            let message = format!("{} hook failed: {}", stage.label(), err);
            let entry = match stage {
                RenderHookStage::PreRender => HookFailureEntry::pre_render(frame_index, &message),
                RenderHookStage::PostRender => HookFailureEntry::post_render(frame_index, &message),
            };
            (Err(HookError::Invocation(message)), Some(entry))
        }
        Err(panic) => {
            let message = format!(
                "{} hook panicked: {}",
                stage.label(),
                super::utils::panic_payload_to_string(panic)
            );
            let entry = match stage {
                RenderHookStage::PreRender => HookFailureEntry::pre_render(frame_index, &message),
                RenderHookStage::PostRender => HookFailureEntry::post_render(frame_index, &message),
            };
            (Err(HookError::Invocation(message)), Some(entry))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{
        boxed_render_hook, invoke_render_hook, BoxedRenderHook, RenderHook, RenderHookStage,
    };
    use crate::api::errors::HookError;

    #[test]
    fn invoke_render_hook_runs_pre_then_post_in_call_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let pre_order = Arc::clone(&order);
        let post_order = Arc::clone(&order);

        let mut pre_hook: Option<BoxedRenderHook> = Some(boxed_render_hook(move |_| {
            pre_order.lock().unwrap().push("pre");
            Ok(())
        }));
        let mut post_hook: Option<BoxedRenderHook> = Some(boxed_render_hook(move |_| {
            post_order.lock().unwrap().push("post");
            Ok(())
        }));

        let (result, _) = invoke_render_hook(
            &mut pre_hook,
            RenderHookStage::PreRender,
            7,
            (1280, 720),
            None,
        );
        result.unwrap();
        let (result, _) = invoke_render_hook(
            &mut post_hook,
            RenderHookStage::PostRender,
            7,
            (1280, 720),
            None,
        );
        result.unwrap();

        let order = order.lock().unwrap();
        assert_eq!(order.as_slice(), ["pre", "post"]);
    }

    #[test]
    fn invoke_render_hook_wraps_hook_errors_as_invocation_failures() {
        let mut pre_hook: Option<BoxedRenderHook> = Some(boxed_render_hook(|_| {
            Err(HookError::Registration("bad registration".to_string()))
        }));

        let (result, entry) =
            invoke_render_hook(&mut pre_hook, RenderHookStage::PreRender, 1, (1, 1), None);
        let err = result.unwrap_err();
        match err {
            HookError::Invocation(msg) => {
                assert!(msg.contains("pre_render hook failed"));
                assert!(msg.contains("bad registration"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        // Verify structured entry is present.
        let entry = entry.expect("should have a failure entry");
        assert_eq!(entry.frame_index, 1);
        assert!(entry.message.contains("bad registration"));
    }

    #[test]
    fn invoke_render_hook_converts_panics_to_invocation_errors() {
        let mut post_hook: Option<BoxedRenderHook> = Some(boxed_render_hook(|_| {
            panic!("boom");
        }));

        let (result, entry) = invoke_render_hook(
            &mut post_hook,
            RenderHookStage::PostRender,
            9,
            (800, 600),
            None,
        );
        let err = result.unwrap_err();
        match err {
            HookError::Invocation(msg) => {
                assert!(msg.contains("post_render hook panicked"));
                assert!(msg.contains("boom"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        let entry = entry.expect("should have a failure entry for panic");
        assert_eq!(entry.frame_index, 9);
        assert!(entry.message.contains("boom"));
    }

    #[test]
    fn invoke_render_hook_is_noop_when_unset() {
        let mut hook: Option<BoxedRenderHook> = None;
        let (result, entry) =
            invoke_render_hook(&mut hook, RenderHookStage::PreRender, 0, (0, 0), None);
        result.unwrap();
        assert!(entry.is_none());
    }

    #[test]
    fn named_struct_implements_render_hook_trait() {
        struct CountingHook {
            count: u32,
        }

        impl RenderHook for CountingHook {
            fn invoke(&mut self, _ctx: &mut super::RenderHookContext<'_>) -> Result<(), HookError> {
                self.count += 1;
                Ok(())
            }
        }

        let mut hook: Option<BoxedRenderHook> = Some(Box::new(CountingHook { count: 0 }));
        let (result, _) =
            invoke_render_hook(&mut hook, RenderHookStage::PreRender, 0, (0, 0), None);
        result.unwrap();
        // The hook ran once
    }
}
