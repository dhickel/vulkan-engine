use crate::vulkan::vk_types::VkGpuMeshBuffers;
use std::fmt::format;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read};
use std::marker::PhantomData;
use std::path::Path;

use ash::vk;
use glfw::Glfw;
use gltf::mesh::util::{ReadColors, ReadIndices, ReadNormals, ReadPositions, ReadTexCoords};
use gltf::texture::{MagFilter, MinFilter};
use gltf::{Gltf, Material, Semantic};
use std::rc::Rc;

use crate::data::gpu_data::{MaterialInstance, MeshNode, Vertex};
use log::log;

#[derive(Debug)]
pub struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
    pub material: Rc<GLTFMaterial>,
}



#[derive(Debug)]
pub struct GLTFMaterial {
    pub data: MaterialInstance,
}

impl GLTFMaterial {
    pub fn null() -> Self {
        Self {
            data: MaterialInstance::null(),
        }
    }

    pub fn new(data: MaterialInstance) -> Self {
        Self { data }
    }
}

impl GeoSurface {
    pub fn new(start_index: u32, count: u32, material: Rc<GLTFMaterial>) -> Self {
        Self {
            start_index,
            count,
            material,
        }
    }
}

#[derive(Debug)]
pub struct MeshAsset {
    pub name: String,
    pub surfaces: Vec<GeoSurface>,
    pub mesh_buffers: VkGpuMeshBuffers,
}

impl MeshAsset {
    // pub fn new() -> self {
    //     Self {
    //         name: "".to_string(),
    //         surfaces: vec![],
    //         mesh_buffers: VkGpuMeshBuffers {},
    //     }
    // }
}

pub fn load_meshes<F>(path: &str, mut upload_fn: F) -> Result<Vec<MeshAsset>, String>
where
    F: FnMut(&[u32], &[Vertex]) -> VkGpuMeshBuffers,
{
    log::info!("Loading Mesh: {:?}", path);

    let path = std::path::Path::new(path);
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let gltf = Gltf::from_reader(reader).unwrap();

    let buffer_data = load_buffers(&gltf, &path).unwrap();

    let mut meshes = Vec::<MeshAsset>::new();

    let mut indices = Vec::<u32>::new();
    let mut vertices = Vec::<Vertex>::new();
    let mut unnamed_idx = 0;

    for mesh in gltf.meshes() {
        let mut surfaces = Vec::<GeoSurface>::new();

        indices.clear();
        vertices.clear();

        for prim in mesh.primitives() {
            let new_surface = GeoSurface {
                start_index: indices.len() as u32,
                count: prim.indices().unwrap().count() as u32,
                material: Rc::new(GLTFMaterial::null()),
            };
            surfaces.push(new_surface);

            let init_vtx = vertices.len();
            let reader = prim.reader(|buffer| Some(buffer_data[buffer.index()].as_slice()));

            match reader.read_indices().unwrap() {
                ReadIndices::U8(val) => {
                    indices.extend(val.map(|idx| idx as u32 + init_vtx as u32));
                }
                ReadIndices::U16(val) => {
                    indices.extend(val.map(|idx| idx as u32 + init_vtx as u32));
                }
                ReadIndices::U32(val) => {
                    indices.extend(val.map(|idx| idx + init_vtx as u32));
                }
            }

            // load vertex positions
            match reader.read_positions().unwrap() {
                ReadPositions::Standard(pos_iter) => {
                    for pos in pos_iter {
                        let vert = Vertex {
                            position: glam::Vec3::from_array(pos),
                            normal: glam::vec3(1.0, 0.0, 0.0),
                            color: glam::Vec4::ONE,
                            uv_x: 0.0,
                            uv_y: 0.0,
                        };
                        vertices.push(vert);
                    }
                }
                ReadPositions::Sparse(pos) => {
                    panic!("Sparse")
                }
            };

            if let Some(norms) = reader.read_normals() {
                match norms {
                    ReadNormals::Standard(norm_iter) => {
                        for (idx, norm) in norm_iter.enumerate() {
                            vertices[init_vtx + idx].normal = glam::Vec3::from_array(norm);
                        }
                    }
                    ReadNormals::Sparse(_) => {
                        panic!("Sparse")
                    }
                }
            }

            if let Some(uvs) = reader.read_tex_coords(0) {
                match uvs {
                    ReadTexCoords::U8(_) => {
                        panic!("u8 uvs")
                    }
                    ReadTexCoords::U16(_) => {
                        panic!("u16 uvs")
                    }
                    ReadTexCoords::F32(cord_iter) => {
                        for (idx, cord) in cord_iter.enumerate() {
                            vertices[init_vtx + idx].uv_x = cord[0];
                            vertices[init_vtx + idx].uv_y = cord[1];
                        }
                    }
                }
            }

            if let Some(colors) = reader.read_colors(0) {
                match colors {
                    ReadColors::RgbF32(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            vertices[init_vtx + idx].color =
                                glam::Vec3::from_array(color).extend(1.0);
                        }
                    }
                    ReadColors::RgbaF32(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            vertices[init_vtx + idx].color = glam::Vec4::from_array(color)
                        }
                    }
                    _ => panic!("Non float colors"),
                }
            }
        }

        let override_colors = false;
        if override_colors {
            vertices
                .iter_mut()
                .for_each(|vtx| vtx.color = vtx.normal.extend(1.0));
        }

        let mesh_buffers = upload_fn(&indices, &vertices);

        let name = mesh.name().map_or_else(
            || {
                unnamed_idx += 1;
                format!("unnamed_{}", unnamed_idx - 1)
            },
            |n| n.to_string(),
        );

        let new_mesh = MeshAsset {
            name,
            surfaces,
            mesh_buffers,
        };

        meshes.push(new_mesh);
    }
    Ok(meshes)
}

fn load_buffers(gltf: &gltf::Gltf, gltf_path: &Path) -> Result<Vec<Vec<u8>>, std::io::Error> {
    gltf.buffers()
        .map(|buffer| match buffer.source() {
            gltf::buffer::Source::Uri(uri) => {
                let buffer_path = gltf_path.parent().unwrap().join(uri);
                let mut file = File::open(buffer_path)?;
                let mut buffer_data = Vec::new();
                file.read_to_end(&mut buffer_data)?;
                Ok(buffer_data)
            }
            gltf::buffer::Source::Bin => {
                gltf.blob.as_ref().map(|blob| blob.to_vec()).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, "Binary buffer not found")
                })
            }
        })
        .collect()
}

fn map_mag_filter(filter: gltf::texture::MagFilter) -> vk::Filter {
    match filter {
        MagFilter::Nearest => vk::Filter::NEAREST,
        MagFilter::Linear => vk::Filter::LINEAR,
    }
}

fn map_min_filter(filter: gltf::texture::MinFilter) -> vk::Filter {
    match filter {
        MinFilter::Nearest | MinFilter::NearestMipmapNearest | MinFilter::NearestMipmapLinear => {
            vk::Filter::NEAREST
        }
        MinFilter::Linear | MinFilter::LinearMipmapNearest | MinFilter::LinearMipmapLinear => {
            vk::Filter::LINEAR
        }
    }
}

pub struct RawSurface {
    start_index: u32,
    count: u32,
    material_index: Option<usize>,
}

pub struct RawMeshData {
    pub name: String,
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub surfaces: Vec<RawSurface>,
}

pub struct RawNodeData {
    pub name: String,
    pub mesh_index: Option<usize>,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub children: Vec<usize>,
}

pub struct RawSceneData<'a> {
    pub meshes: Vec<RawMeshData>,
    pub materials: Vec<gltf::Material<'a>>,
    pub nodes: Vec<RawNodeData>,
    pub root_indices: Vec<usize>,
}

pub fn parse_gltf_to_raw(path: &str) -> Result<String, String> {
    log::info!("Loading Mesh: {:?}", path);

    // let path = std::path::Path::new(path);
    //    let file = File::open(path).unwrap();
    //  let reader = BufReader::new(file);
    //  let gltf_data = Gltf::from_reader(reader).unwrap();

    //  let buffer_data = load_buffers(&gltf_data, &path).unwrap();

    let (gltf_data, buffer_data, images) =
        gltf::import(path).map_err(|err| format!("Error loading GLTF file: {:?}", err))?;

    
    println!("Image Count: {}", images.len());
    let mut parsed_meshes = Vec::<RawMeshData>::new();
    let materials: Vec<gltf::Material> = gltf_data.materials().collect();

    let mut unnamed_mesh_index = 0;
    let mut unnamed_mat_index = 0;

    for mesh in gltf_data.meshes() {
        let name = if let Some(name) = mesh.name() {
            name.to_string()
        } else {
            unnamed_mesh_index += 1;
            format!("unnamed_mesh_{:?}", unnamed_mesh_index - 1).to_string()
        };

        let mut tmp_indices = Vec::<u32>::new();
        let mut tmp_vertices = Vec::<Vertex>::new();
        let mut surfaces = Vec::<RawSurface>::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(buffer_data[buffer.index()].0.as_slice()));
            let start_index = tmp_vertices.len();
            let count = primitive.indices().unwrap().count();

            let material = primitive.material();

            // INDICES
            match reader
                .read_indices()
                .ok_or_else(|| "No Indices found".to_string())?
            {
                ReadIndices::U8(val) => {
                    tmp_indices.extend(val.map(|idx| idx as u32 + start_index as u32));
                }
                ReadIndices::U16(val) => {
                    tmp_indices.extend(val.map(|idx| idx as u32 + start_index as u32));
                }
                ReadIndices::U32(val) => {
                    tmp_indices.extend(val.map(|idx| idx + start_index as u32));
                }
            }

            // VERTICES
            match reader
                .read_positions()
                .ok_or_else(|| "No Vertices found".to_string())?
            {
                ReadPositions::Standard(pos_iter) => {
                    for pos in pos_iter {
                        let vert = Vertex {
                            position: glam::Vec3::from_array(pos),
                            normal: glam::vec3(1.0, 0.0, 0.0),
                            color: glam::Vec4::ONE,
                            uv_x: 0.0,
                            uv_y: 0.0,
                        };
                        tmp_vertices.push(vert);
                    }
                }
                ReadPositions::Sparse(pos_iter) => {
                    for pos in pos_iter {
                        let vert = Vertex {
                            position: glam::Vec3::from_array(pos),
                            normal: glam::vec3(1.0, 0.0, 0.0),
                            color: glam::Vec4::ONE,
                            uv_x: 0.0,
                            uv_y: 0.0,
                        };
                        tmp_vertices.push(vert);
                    }
                }
            };

            // NORMALS
            if let Some(norms) = reader.read_normals() {
                match norms {
                    ReadNormals::Standard(norm_iter) => {
                        for (idx, norm) in norm_iter.enumerate() {
                            tmp_vertices[start_index + idx].normal = glam::Vec3::from_array(norm);
                        }
                    }
                    ReadNormals::Sparse(norm_iter) => {
                        for (idx, norm) in norm_iter.enumerate() {
                            tmp_vertices[start_index + idx].normal = glam::Vec3::from_array(norm);
                        }
                    }
                }
            }

            // UV COORDS
            if let Some(uvs) = reader.read_tex_coords(0) {
                match uvs {
                    ReadTexCoords::U8(cord_iter) => {
                        for (idx, cord) in cord_iter.enumerate() {
                            tmp_vertices[start_index + idx].uv_x = cord[0] as f32;
                            tmp_vertices[start_index + idx].uv_y = cord[1] as f32;
                        }
                    }
                    ReadTexCoords::U16(cord_iter) => {
                        for (idx, cord) in cord_iter.enumerate() {
                            tmp_vertices[start_index + idx].uv_x = cord[0] as f32;
                            tmp_vertices[start_index + idx].uv_y = cord[1] as f32;
                        }
                    }
                    ReadTexCoords::F32(cord_iter) => {
                        for (idx, cord) in cord_iter.enumerate() {
                            tmp_vertices[start_index + idx].uv_x = cord[0];
                            tmp_vertices[start_index + idx].uv_y = cord[1];
                        }
                    }
                }
            }

            // COLORS
            if let Some(colors) = reader.read_colors(0) {
                match colors {
                    ReadColors::RgbF32(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            tmp_vertices[start_index + idx].color =
                                glam::Vec3::from_array(color).extend(1.0);
                        }
                    }
                    ReadColors::RgbaF32(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            tmp_vertices[start_index + idx].color = glam::Vec4::from_array(color)
                        }
                    }
                    ReadColors::RgbU8(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            let color: [u8; 3] = color.into();
                            let normalized_color = [
                                color[0] as f32 / 255.0,
                                color[1] as f32 / 255.0,
                                color[2] as f32 / 255.0,
                                1.0,
                            ];
                            tmp_vertices[start_index + idx].color =
                                glam::Vec4::from_array(normalized_color);
                        }
                    }
                    ReadColors::RgbaU8(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            let color: [u8; 4] = color.into();
                            let normalized_color = [
                                color[0] as f32 / 255.0,
                                color[1] as f32 / 255.0,
                                color[2] as f32 / 255.0,
                                color[3] as f32 / 255.0,
                            ];
                            tmp_vertices[start_index + idx].color =
                                glam::Vec4::from_array(normalized_color);
                        }
                    }
                    ReadColors::RgbU16(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            let color: [u16; 3] = color.into();
                            let normalized_color = [
                                color[0] as f32 / 65535.0,
                                color[1] as f32 / 65535.0,
                                color[2] as f32 / 65535.0,
                                1.0,
                            ];
                            tmp_vertices[start_index + idx].color =
                                glam::Vec4::from_array(normalized_color);
                        }
                    }
                    ReadColors::RgbaU16(color_iter) => {
                        for (idx, color) in color_iter.enumerate() {
                            let color: [u16; 4] = color.into();
                            let normalized_color = [
                                color[0] as f32 / 65535.0,
                                color[1] as f32 / 65535.0,
                                color[2] as f32 / 65535.0,
                                color[3] as f32 / 65535.0,
                            ];
                            tmp_vertices[start_index + idx].color =
                                glam::Vec4::from_array(normalized_color);
                        }
                    }
                }
            }

            surfaces.push(RawSurface {
                start_index: start_index as u32,
                count: count as u32,
                material_index: primitive.material().index(),
            })
        }

        parsed_meshes.push(RawMeshData {
            name,
            indices: tmp_indices,
            vertices: tmp_vertices,
            surfaces,
        })
    }

    // NODES, INDICES & TRANSFORMS
    let mut root_indices: Vec<Option<usize>> = gltf_data.nodes().map(|n| Some(n.index())).collect();
    let mut nodes = Vec::<RawNodeData>::with_capacity(gltf_data.nodes().count());

    let mut unnamed_node_idx = 0;
    for node in gltf_data.nodes() {
        let name = if let Some(name) = node.name() {
            name.to_string()
        } else {
            unnamed_node_idx += 1;
            format!("unnamed_node_{:?}", unnamed_node_idx).to_string()
        };

        let mesh_index = node.mesh().map(|m| m.index());

        let (translation, rotation, scale) = node.transform().decomposed();
        let position = glam::Vec3::from_array(translation);
        let rotation = glam::Quat::from_array(rotation);
        let scale = glam::Vec3::from_array(scale);

        let children: Vec<usize> = node
            .children()
            .map(|c| {
                let index = c.index();
                root_indices[index] = None;
                index
            })
            .collect();

        let node_data = RawNodeData {
            name,
            mesh_index,
            position,
            rotation,
            scale,
            children,
        };

        nodes.push(node_data);
    }

    let scene_data = RawSceneData {
        meshes: parsed_meshes,
        materials,
        nodes,
        root_indices: root_indices
            .into_iter()
            .flatten()
            .collect(),
    };
    Err("Fucky".to_string())

}

fn extract_scene_data(scene: gltf::Scene) {}

// fn load_buffers(
//     gltf: &gltf::Gltf,
//     path: &Path
// ) -> Result<Vec<Vec<u8>>, Gltf::GltfError> {
//     const VALID_MIME_TYPES: &[&str] = &["application/octet-stream", "application/gltf-buffer"];
//     let mut buffer_data = Vec::new();
//     for buffer in gltf.buffers() {
//         match buffer.source() {
//             gltf::buffer::Source::Uri(uri) => {
//                 let uri = percent_encoding::percent_decode_str(uri)
//                     .decode_utf8()
//                     .unwrap();
//                 let uri = uri.as_ref();
//                 let buffer_bytes = match DataUri::parse(uri) {
//                     Ok(data_uri) if VALID_MIME_TYPES.contains(&data_uri.mime_type) => {
//                         datauri.decode()?
//                     }
//                     Ok() => return Err(GltfError::BufferFormatUnsupported),
//                     Err(()) => {
//                         // TODO: Remove this and add dep
//                         let buffer_path = load_context.path().parent().unwrap().join(uri);
//                         load_context.read_asset_bytes(buffer_path).await?
//                     }
//                 };
//                 buffer_data.push(buffer_bytes);
//             }
//             gltf::buffer::Source::Bin => {
//                 if let Some(blob) = gltf.blob.as_deref() {
//                     buffer_data.push(blob.into());
//                 } else {
//                     return Err(Gltf::GltfError::MissingBlob);
//                 }
//             }
//         }
//     }
//     Ok(buffer_data)
// }
