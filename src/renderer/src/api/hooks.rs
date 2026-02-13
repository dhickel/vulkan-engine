use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::errors::HookError;

pub struct RenderHookContext<'a> {
    pub frame_index: u64,
    pub viewport_size: (u32, u32),
    // Prevent external construction and preserve room for future internal context data.
    _private: PhantomData<&'a mut ()>,
}

impl<'a> RenderHookContext<'a> {
    pub(crate) fn new(frame_index: u64, viewport_size: (u32, u32)) -> Self {
        Self {
            frame_index,
            viewport_size,
            _private: PhantomData,
        }
    }
}

pub type RenderHook = Box<dyn FnMut(&mut RenderHookContext<'_>) -> Result<(), HookError> + Send>;

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
    hook: &mut Option<RenderHook>,
    stage: RenderHookStage,
    frame_index: u64,
    viewport_size: (u32, u32),
) -> Result<(), HookError> {
    let Some(hook) = hook.as_mut() else {
        return Ok(());
    };

    let mut context = RenderHookContext::new(frame_index, viewport_size);
    let callback_result =
        catch_unwind(AssertUnwindSafe(|| hook(&mut context))).map_err(|panic| {
            HookError::Invocation(format!(
                "{} hook panicked: {}",
                stage.label(),
                panic_payload_to_string(panic)
            ))
        })?;

    callback_result
        .map_err(|err| HookError::Invocation(format!("{} hook failed: {}", stage.label(), err)))
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    let payload = payload.as_ref();
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        return (*msg).to_string();
    }
    "unknown panic payload".to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{invoke_render_hook, RenderHook, RenderHookStage};
    use crate::api::errors::HookError;

    #[test]
    fn invoke_render_hook_runs_pre_then_post_in_call_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let pre_order = Arc::clone(&order);
        let post_order = Arc::clone(&order);

        let mut pre_hook: Option<RenderHook> = Some(Box::new(move |_| {
            pre_order.lock().unwrap().push("pre");
            Ok(())
        }));
        let mut post_hook: Option<RenderHook> = Some(Box::new(move |_| {
            post_order.lock().unwrap().push("post");
            Ok(())
        }));

        invoke_render_hook(&mut pre_hook, RenderHookStage::PreRender, 7, (1280, 720)).unwrap();
        invoke_render_hook(&mut post_hook, RenderHookStage::PostRender, 7, (1280, 720)).unwrap();

        let order = order.lock().unwrap();
        assert_eq!(order.as_slice(), ["pre", "post"]);
    }

    #[test]
    fn invoke_render_hook_wraps_hook_errors_as_invocation_failures() {
        let mut pre_hook: Option<RenderHook> = Some(Box::new(|_| {
            Err(HookError::Registration("bad registration".to_string()))
        }));

        let err =
            invoke_render_hook(&mut pre_hook, RenderHookStage::PreRender, 1, (1, 1)).unwrap_err();
        match err {
            HookError::Invocation(msg) => {
                assert!(msg.contains("pre_render hook failed"));
                assert!(msg.contains("bad registration"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn invoke_render_hook_converts_panics_to_invocation_errors() {
        let mut post_hook: Option<RenderHook> = Some(Box::new(|_| {
            panic!("boom");
        }));

        let err = invoke_render_hook(&mut post_hook, RenderHookStage::PostRender, 9, (800, 600))
            .unwrap_err();
        match err {
            HookError::Invocation(msg) => {
                assert!(msg.contains("post_render hook panicked"));
                assert!(msg.contains("boom"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn invoke_render_hook_is_noop_when_unset() {
        let mut hook: Option<RenderHook> = None;
        invoke_render_hook(&mut hook, RenderHookStage::PreRender, 0, (0, 0)).unwrap();
    }
}
