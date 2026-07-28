//! Model Context Protocol server for BSP beta's headless runtime.
//!
//! The transport is newline-delimited JSON-RPC 2.0 over stdin/stdout. Engine
//! diagnostics continue to use stderr through the normal logging facade.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use bsp::coords::QuakeToEngine;
use bsp::extract::{ExtractedBsp, ExtractedVisibility};
use glam::Vec3;
use renderer::api::bsp::PreparedBspMount;
use renderer::api::{CaptureTarget, FrameCaptureRequest, FrameCaptureStatus};
use renderer::{Renderer, Scene};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{render_app_frame, AppLoopState};

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const CAPTURE_WIDTH: u32 = 1920;
const CAPTURE_HEIGHT: u32 = 1080;
const CAPTURE_POLL_FRAMES: usize = 16;

/// BSP data retained by the MCP runtime after the coordinator consumes its
/// staged extraction during publication.
pub(super) struct McpMap {
    faces: usize,
    batches: usize,
    materials: usize,
    textures: usize,
    pbr_count: usize,
    bsp_size: usize,
    transform: QuakeToEngine,
    visibility: ExtractedVisibility,
}

impl McpMap {
    pub(super) fn from_mount(
        extracted: &ExtractedBsp,
        mount: &PreparedBspMount,
        bsp_size: usize,
    ) -> Self {
        let demand = mount.upload_demand;
        let mut pbr_textures = vec![false; extracted.textures.len()];
        for material in &extracted.face_materials {
            if !matches!(
                material.surface_class,
                bsp::materials::SurfaceClass::Opaque | bsp::materials::SurfaceClass::AlphaMask
            ) {
                continue;
            }
            let Ok(texture_index) = usize::try_from(material.material_index) else {
                continue;
            };
            if extracted
                .textures
                .get(texture_index)
                .is_some_and(|texture| !texture.pbr_companions.is_empty())
            {
                pbr_textures[texture_index] = true;
            }
        }

        Self {
            faces: extracted.face_geometries.len(),
            batches: demand
                .map(|demand| demand.batch_count)
                .unwrap_or(mount.render_batches.len()),
            materials: demand
                .map(|demand| demand.material_count)
                .unwrap_or(mount.batch_materials.len()),
            textures: demand
                .map(|demand| demand.texture_count)
                .unwrap_or(extracted.textures.len()),
            pbr_count: pbr_textures.into_iter().filter(|is_pbr| *is_pbr).count(),
            bsp_size,
            transform: extracted.transform,
            visibility: extracted.visibility.clone(),
        }
    }

    fn info(&self, camera_position: Vec3) -> Value {
        json!({
            "faces": self.faces,
            "batches": self.batches,
            "materials": self.materials,
            "textures": self.textures,
            "pbr_count": self.pbr_count,
            "bsp_size": self.bsp_size,
            "camera_pos": vec3_json(camera_position),
        })
    }

    fn point_contents(&self, point: Vec3) -> Value {
        let contents = bsp::point_contents_with_transform(
            point,
            &self.visibility.nodes,
            &self.visibility.leaves,
            &self.visibility.planes,
            &self.transform,
        );
        let leaf = self.leaf_index(point);
        json!({
            "solid": contents.is_solid(),
            "leaf": leaf,
        })
    }

    fn leaf_index(&self, point: Vec3) -> i64 {
        if self.visibility.nodes.is_empty()
            || self.visibility.leaves.is_empty()
            || self.visibility.planes.is_empty()
            || self.transform.scale.abs() < f32::EPSILON
        {
            return -1;
        }

        let inverse_scale = 1.0 / self.transform.scale;
        let quake_point = Vec3::new(
            point.x * inverse_scale,
            -point.z * inverse_scale,
            point.y * inverse_scale,
        );
        let result = bsp::camera_leaf_index(
            &quake_point,
            &self.visibility.nodes,
            &self.visibility.leaves,
            &self.visibility.planes,
        );
        if result.outside {
            -1
        } else {
            i64::from(result.leaf_index)
        }
    }
}

struct RuntimeTools<'a> {
    renderer: &'a mut Renderer,
    scene: &'a mut Scene,
    loop_state: &'a mut AppLoopState,
    map: McpMap,
    capture_sequence: u64,
    /// Phase 07: Active evidence request key (if pending).
    evidence_key: Option<renderer::api::bsp::BspEvidenceRequestKey>,
}

impl ToolBackend for RuntimeTools<'_> {
    fn set_camera(&mut self, params: CameraParams) -> Result<Value, String> {
        let position = Vec3::new(params.x, params.y, params.z);
        let pitch_limit = std::f32::consts::FRAC_PI_2 - 0.01;
        let pitch = params.pitch.clamp(-pitch_limit, pitch_limit);
        let mut camera = renderer::Camera::new(position);
        camera.update_rotation(params.yaw, pitch);
        self.loop_state.camera = camera;

        Ok(json!({
            "position": vec3_json(position),
            "yaw": params.yaw,
            "pitch": pitch,
        }))
    }

    fn capture(&mut self, requested_path: Option<PathBuf>) -> Result<Value, String> {
        let output_path = requested_path.unwrap_or_else(|| {
            let path = PathBuf::from(format!(
                ".internal-dev/captures/bsp-beta/mcp-{}/capture-{:04}.png",
                std::process::id(),
                self.capture_sequence,
            ));
            self.capture_sequence = self.capture_sequence.saturating_add(1);
            path
        });
        if output_path.as_os_str().is_empty() {
            return Err("capture path must not be empty".to_string());
        }
        if let Some(parent) = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!("create capture directory '{}': {error}", parent.display())
            })?;
        }

        self.renderer
            .request_frame_capture(FrameCaptureRequest {
                target: CaptureTarget::Draw,
                output_path: output_path.clone(),
                sidecar_path: None,
            })
            .map_err(|error| format!("queue capture: {error}"))?;

        for _ in 0..CAPTURE_POLL_FRAMES {
            render_app_frame(
                self.renderer,
                self.scene,
                self.loop_state,
                CAPTURE_WIDTH,
                CAPTURE_HEIGHT,
                true,
            )
            .map_err(|error| format!("render capture frame: {error}"))?;

            match self.renderer.last_frame_capture_status() {
                Some(FrameCaptureStatus::Succeeded {
                    output_path: completed_path,
                    width,
                    height,
                    ..
                }) if completed_path == &output_path => {
                    return Ok(json!({
                        "path": completed_path.to_string_lossy(),
                        "width": width,
                        "height": height,
                    }));
                }
                Some(FrameCaptureStatus::Failed {
                    output_path: failed_path,
                    message,
                    ..
                }) if failed_path == &output_path => {
                    return Err(format!("capture failed: {message}"));
                }
                Some(FrameCaptureStatus::BackendNotImplemented {
                    output_path: failed_path,
                    ..
                }) if failed_path == &output_path => {
                    return Err("capture backend not implemented".to_string());
                }
                _ => {}
            }
        }

        Err(format!(
            "capture remained pending after {CAPTURE_POLL_FRAMES} frame polls: {}",
            output_path.display()
        ))
    }

    fn get_info(&mut self) -> Result<Value, String> {
        Ok(self.map.info(self.loop_state.camera.get_position()))
    }

    fn point_contents(&mut self, point: Vec3) -> Result<Value, String> {
        Ok(self.map.point_contents(point))
    }

    /// Phase 07: Request a BSP stats evidence report.
    fn stats(&mut self, all_visible: bool) -> Result<Value, String> {
        use renderer::api::bsp::BspEvidenceVisibility;

        let visibility = if all_visible {
            BspEvidenceVisibility::AllVisible
        } else {
            BspEvidenceVisibility::NormalPvs
        };

        let corpus = self.map.info(self.loop_state.camera.get_position())
            .get("faces")
            .and_then(|v| v.as_u64())
            .map(|f| format!("mcp-bsp-{f}-faces"))
            .unwrap_or_else(|| "mcp-bsp-unknown".to_string());

        let key = self.renderer
            .request_bsp_frame_evidence(corpus, "mcp-stats".to_string(), visibility)
            .map_err(|e| format!("evidence request failed: {e}"))?;
        self.evidence_key = Some(key);

        // Render frames until evidence is ready
        for _ in 0..CAPTURE_POLL_FRAMES {
            render_app_frame(
                self.renderer,
                self.scene,
                self.loop_state,
                CAPTURE_WIDTH,
                CAPTURE_HEIGHT,
                true,
            )
            .map_err(|error| format!("render stats frame: {error}"))?;

            match self.renderer.take_bsp_frame_evidence(key) {
                renderer::api::bsp::BspEvidenceStatus::Sealed(report) => {
                    self.evidence_key = None;
                    return Ok(serde_json::to_value(&report)
                        .map_err(|e| format!("serialize report: {e}"))?);
                }
                renderer::api::bsp::BspEvidenceStatus::RejectedNoMount => {
                    self.evidence_key = None;
                    return Err("no active BSP mount".to_string());
                }
                renderer::api::bsp::BspEvidenceStatus::MissingReport => {
                    self.evidence_key = None;
                    return Err("evidence report missing".to_string());
                }
                renderer::api::bsp::BspEvidenceStatus::Pending => {
                    // Continue polling
                }
            }
        }

        self.evidence_key = None;
        Err(format!("stats report not ready after {CAPTURE_POLL_FRAMES} frames"))
    }
}

/// Serve MCP requests until stdin reaches EOF.
pub(super) fn serve(
    renderer: &mut Renderer,
    scene: &mut Scene,
    loop_state: &mut AppLoopState,
    map: McpMap,
) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut tools = RuntimeTools {
        renderer,
        scene,
        loop_state,
        map,
        capture_sequence: 0,
        evidence_key: None,
    };
    serve_io(stdin.lock(), stdout.lock(), &mut tools)
}

trait ToolBackend {
    fn set_camera(&mut self, params: CameraParams) -> Result<Value, String>;
    fn capture(&mut self, path: Option<PathBuf>) -> Result<Value, String>;
    fn get_info(&mut self) -> Result<Value, String>;
    fn point_contents(&mut self, point: Vec3) -> Result<Value, String>;
    /// Phase 07: Request a BSP draw evidence report.
    fn stats(&mut self, all_visible: bool) -> Result<Value, String>;
}

#[derive(Debug, Clone, Copy)]
struct CameraParams {
    x: f32,
    y: f32,
    z: f32,
    yaw: f32,
    pitch: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCameraParams {
    x: f64,
    y: f64,
    z: f64,
    yaw: f64,
    pitch: f64,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CaptureParams {
    path: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPointParams {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug)]
struct RpcError {
    code: i32,
    message: String,
}

impl RpcError {
    fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
        }
    }
}

fn serve_io(
    mut reader: impl BufRead,
    mut writer: impl Write,
    backend: &mut impl ToolBackend,
) -> io::Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Value>(&line) {
            Ok(message) => handle_message(message, backend),
            Err(error) => Some(error_response(
                Value::Null,
                -32700,
                "Parse error",
                Some(json!({ "detail": error.to_string() })),
            )),
        };
        if let Some(response) = response {
            serde_json::to_writer(&mut writer, &response).map_err(io::Error::other)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
    }
}

fn handle_message(message: Value, backend: &mut impl ToolBackend) -> Option<Value> {
    let Some(request) = message.as_object() else {
        return Some(error_response(Value::Null, -32600, "Invalid Request", None));
    };
    if request.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(error_response(
            request_id(request).unwrap_or(Value::Null),
            -32600,
            "Invalid Request",
            None,
        ));
    }
    let Some(method) = request.get("method").and_then(Value::as_str) else {
        return Some(error_response(
            request_id(request).unwrap_or(Value::Null),
            -32600,
            "Invalid Request",
            None,
        ));
    };
    if request.get("id").is_some() && request_id(request).is_none() {
        return Some(error_response(Value::Null, -32600, "Invalid Request", None));
    }

    let id = request_id(request);
    let result = dispatch(method, request.get("params"), backend);
    let Some(id) = id else {
        return None;
    };
    Some(match result {
        Ok(result) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
        Err(error) => error_response(id, error.code, &error.message, None),
    })
}

fn request_id(request: &Map<String, Value>) -> Option<Value> {
    match request.get("id") {
        None => None,
        Some(value @ (Value::Null | Value::Number(_) | Value::String(_))) => Some(value.clone()),
        Some(_) => None,
    }
}

fn dispatch(
    method: &str,
    params: Option<&Value>,
    backend: &mut impl ToolBackend,
) -> Result<Value, RpcError> {
    match method {
        "initialize" => initialize_result(params),
        "notifications/initialized" => Ok(Value::Null),
        "ping" => Ok(json!({})),
        "tools/list" => {
            optional_object(params)?;
            Ok(json!({ "tools": tool_descriptors() }))
        }
        "tools/call" => call_tool(params, backend),
        _ => Err(RpcError {
            code: -32601,
            message: "Method not found".to_string(),
        }),
    }
}

fn initialize_result(params: Option<&Value>) -> Result<Value, RpcError> {
    let params = optional_object(params)?;
    let protocol_version = match params.and_then(|params| params.get("protocolVersion")) {
        None | Some(Value::Null) => MCP_PROTOCOL_VERSION,
        Some(Value::String(version)) if !version.is_empty() => version,
        Some(_) => {
            return Err(RpcError::invalid_params(
                "initialize protocolVersion must be a non-empty string",
            ));
        }
    };

    Ok(json!({
        "protocolVersion": protocol_version,
        "capabilities": {
            "tools": {
                "listChanged": false,
            },
        },
        "serverInfo": {
            "name": "bsp_beta",
            "version": env!("CARGO_PKG_VERSION"),
        },
    }))
}

fn optional_object(params: Option<&Value>) -> Result<Option<&Map<String, Value>>, RpcError> {
    match params {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(object)),
        Some(_) => Err(RpcError::invalid_params("params must be an object")),
    }
}

fn call_tool(params: Option<&Value>, backend: &mut impl ToolBackend) -> Result<Value, RpcError> {
    let params = optional_object(params)?
        .ok_or_else(|| RpcError::invalid_params("tools/call requires params"))?;
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::invalid_params("tools/call requires a string name"))?;
    let arguments = match params.get("arguments") {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::Object(arguments)) => Value::Object(arguments.clone()),
        Some(_) => return Err(RpcError::invalid_params("tool arguments must be an object")),
    };

    let result = match name {
        "set_camera" => {
            let raw: RawCameraParams = decode_arguments(arguments)?;
            let params = CameraParams {
                x: finite_f32("x", raw.x)?,
                y: finite_f32("y", raw.y)?,
                z: finite_f32("z", raw.z)?,
                yaw: finite_f32("yaw", raw.yaw)?,
                pitch: finite_f32("pitch", raw.pitch)?,
            };
            backend.set_camera(params)
        }
        "capture" => {
            let params: CaptureParams = decode_arguments(arguments)?;
            backend.capture(params.path.map(PathBuf::from))
        }
        "get_info" => {
            ensure_empty_arguments(&arguments)?;
            backend.get_info()
        }
        "point_contents" => {
            let raw: RawPointParams = decode_arguments(arguments)?;
            backend.point_contents(Vec3::new(
                finite_f32("x", raw.x)?,
                finite_f32("y", raw.y)?,
                finite_f32("z", raw.z)?,
            ))
        }
        "stats" => {
            #[derive(Deserialize, Default)]
            #[serde(default)]
            struct StatsParams {
                all_visible: bool,
            }
            let params: StatsParams = decode_arguments(arguments)?;
            backend.stats(params.all_visible)
        }
        _ => Err(format!("unknown tool: {name}")),
    };

    Ok(tool_result(result))
}

fn decode_arguments<T: for<'de> Deserialize<'de>>(arguments: Value) -> Result<T, RpcError> {
    serde_json::from_value(arguments)
        .map_err(|error| RpcError::invalid_params(format!("invalid tool arguments: {error}")))
}

fn ensure_empty_arguments(arguments: &Value) -> Result<(), RpcError> {
    if arguments.as_object().is_some_and(Map::is_empty) {
        Ok(())
    } else {
        Err(RpcError::invalid_params(
            "get_info does not accept arguments",
        ))
    }
}

fn finite_f32(name: &str, value: f64) -> Result<f32, RpcError> {
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(RpcError::invalid_params(format!(
            "{name} must be a finite 32-bit number"
        )));
    }
    Ok(value as f32)
}

fn tool_result(result: Result<Value, String>) -> Value {
    match result {
        Ok(data) => {
            let text = data.to_string();
            json!({
                "content": [{ "type": "text", "text": text }],
                "structuredContent": data,
                "isError": false,
            })
        }
        Err(message) => json!({
            "content": [{ "type": "text", "text": message }],
            "isError": true,
        }),
    }
}

fn tool_descriptors() -> Vec<Value> {
    vec![
        json!({
            "name": "set_camera",
            "description": "Reposition the BSP view. Yaw and pitch are in radians.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number" },
                    "y": { "type": "number" },
                    "z": { "type": "number" },
                    "yaw": { "type": "number", "description": "Yaw in radians." },
                    "pitch": { "type": "number", "description": "Pitch in radians." },
                },
                "required": ["x", "y", "z", "yaw", "pitch"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "capture",
            "description": "Render and save one 1920x1080 headless PNG frame.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Optional PNG output path." },
                },
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "get_info",
            "description": "Return BSP render counts, source size, and camera position.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "point_contents",
            "description": "Query whether an engine-space point is solid and return its BSP leaf.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number" },
                    "y": { "type": "number" },
                    "z": { "type": "number" },
                },
                "required": ["x", "y", "z"],
                "additionalProperties": false,
            },
        }),
        json!({
            "name": "stats",
            "description": "Request a bounded post-command evidence report for BSP static-world draws (batch count, draw call count, triangle count, material count, atlas bytes, frame time).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "all_visible": { "type": "boolean", "description": "Use all-visible visibility mode (default: false, normal PVS)." },
                },
                "additionalProperties": false,
            },
        }),
    ]
}

fn error_response(id: Value, code: i32, message: &str, data: Option<Value>) -> Value {
    let mut error = json!({
        "code": code,
        "message": message,
    });
    if let Some(data) = data {
        error["data"] = data;
    }
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": error,
    })
}

fn vec3_json(value: Vec3) -> Value {
    json!({
        "x": value.x,
        "y": value.y,
        "z": value.z,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[derive(Default)]
    struct FakeTools {
        camera: Option<CameraParams>,
    }

    impl ToolBackend for FakeTools {
        fn set_camera(&mut self, params: CameraParams) -> Result<Value, String> {
            self.camera = Some(params);
            Ok(json!({ "ok": true }))
        }

        fn capture(&mut self, path: Option<PathBuf>) -> Result<Value, String> {
            Ok(json!({
                "path": path.unwrap_or_else(|| PathBuf::from("capture.png")),
                "width": CAPTURE_WIDTH,
                "height": CAPTURE_HEIGHT,
            }))
        }

        fn get_info(&mut self) -> Result<Value, String> {
            Ok(json!({ "faces": 12 }))
        }

        fn point_contents(&mut self, point: Vec3) -> Result<Value, String> {
            Ok(json!({ "solid": point.x < 0.0, "leaf": 2 }))
        }

        fn stats(&mut self, _all_visible: bool) -> Result<Value, String> {
            Ok(json!({ "batch_count": 10, "draw_call_count": 10 }))
        }
    }

    fn run(input: &str, tools: &mut FakeTools) -> Vec<Value> {
        let mut output = Vec::new();
        serve_io(Cursor::new(input.as_bytes()), &mut output, tools).unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn initialize_and_list_tools() {
        let responses = run(
            concat!(
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\"}}\n",
                "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n",
                "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n",
            ),
            &mut FakeTools::default(),
        );

        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(responses[0]["result"]["serverInfo"]["name"], "bsp_beta");
        let names = responses[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["set_camera", "capture", "get_info", "point_contents", "stats"]
        );
    }

    #[test]
    fn tool_calls_execute_and_return_structured_content() {
        let mut tools = FakeTools::default();
        let responses = run(
            concat!(
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"set_camera\",\"arguments\":{\"x\":1,\"y\":2,\"z\":3,\"yaw\":0.5,\"pitch\":-0.25}}}\n",
                "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"point_contents\",\"arguments\":{\"x\":-1,\"y\":0,\"z\":0}}}\n",
            ),
            &mut tools,
        );

        let camera = tools.camera.unwrap();
        assert_eq!((camera.x, camera.y, camera.z), (1.0, 2.0, 3.0));
        assert_eq!(responses[0]["result"]["isError"], false);
        assert_eq!(responses[1]["result"]["structuredContent"]["solid"], true);
        assert_eq!(responses[1]["result"]["structuredContent"]["leaf"], 2);
    }

    #[test]
    fn malformed_json_and_unknown_method_return_json_rpc_errors() {
        let responses = run(
            concat!(
                "not json\n",
                "{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"missing\"}\n",
            ),
            &mut FakeTools::default(),
        );

        assert_eq!(responses[0]["error"]["code"], -32700);
        assert_eq!(responses[0]["id"], Value::Null);
        assert_eq!(responses[1]["error"]["code"], -32601);
        assert_eq!(responses[1]["id"], 9);
    }
}
