use crate::vulkan::vk_types::VkGpuMeshBuffers;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use glfw::Glfw;
use gltf::mesh::util::{ReadColors, ReadIndices, ReadNormals, ReadPositions, ReadTexCoords};
use gltf::{Gltf, Semantic};
use std::rc::Rc;

use crate::data::gpu_data::Vertex;
use log::log;

#[derive(Debug)]
pub struct GeoSurface {
    pub start_index: u32,
    pub count: u32,
}



impl GeoSurface {
    pub fn new(start_index: u32, count: u32) -> Self {
        Self { start_index, count }
    }
}
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

pub fn load_meshes<F>(path: &str, mut upload_fn: F) -> Result<Vec<Rc<MeshAsset>>, String>
where
    F: FnMut(&[u32], &[Vertex]) -> VkGpuMeshBuffers,
{
    log::info!("Loading Mesh: {:?}", path);

    let path = std::path::Path::new(path);
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let gltf = Gltf::from_reader(reader).unwrap();

    let buffer_data = load_buffers(&gltf, &path).unwrap();

    let mut meshes = Vec::<Rc<MeshAsset>>::new();

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

        let override_colors = true;
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

        meshes.push(Rc::new(new_mesh));
    }
    Ok(meshes)
}

fn load_buffers(gltf: &gltf::Gltf, gltf_path: &Path) -> Result<Vec<Vec<u8>>, std::io::Error> {
    gltf.buffers().map(|buffer| {
        match buffer.source() {
            gltf::buffer::Source::Uri(uri) => {
                let buffer_path = gltf_path.parent().unwrap().join(uri);
                let mut file = File::open(buffer_path)?;
                let mut buffer_data = Vec::new();
                file.read_to_end(&mut buffer_data)?;
                Ok(buffer_data)
            },
            gltf::buffer::Source::Bin => {
                gltf.blob.as_ref()
                    .map(|blob| blob.to_vec())
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Binary buffer not found"))
            }
        }
    }).collect()
}

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
