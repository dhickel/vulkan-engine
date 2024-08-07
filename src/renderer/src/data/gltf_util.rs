// use std::any::Any;
// use std::cell::RefCell;
// use std::collections::HashMap;
// use std::default::Default;
// 
// use std::fmt::format;
// use std::fs;
// use std::fs::File;
// use std::io::{BufReader, Read};
// use std::marker::PhantomData;
// use std::path::Path;
// 
// use crate::data::data_cache::{MeshCache, TextureCache};
// use crate::data::gpu_data;
// use crate::data::gpu_data::{
//     EmissiveMap, MeshMeta, Node, NodeMeta, NormalMap, OcclusionMap, SurfaceMeta, TextureMeta,
//     Transform, Vertex, VkGpuMeshBuffers, VkGpuTextureBuffer,
// };
// use ash::vk;
// use glam::Vec2;
// use gltf::image::{Data, Format};
// use gltf::material::AlphaMode;
// use gltf::mesh::util::{
//     ReadColors, ReadIndices, ReadNormals, ReadPositions, ReadTangents, ReadTexCoords,
// };
// use gltf::mesh::Reader;
// use gltf::texture::{MagFilter, MinFilter};
// use gltf::{Document, Gltf, Material, Semantic};
// use imgui::sys::ImDrawFlags_None;
// use log::{info, log};
// use mikktspace::{generate_tangents, Geometry};
// use std::rc::{Rc, Weak};
// 
// // TODO: instead of using a index/vertex array for each mesh in a file, use a single pair for the
// //  whole file and index the meshes and surfaces into it.
// 
// // TODO: Handle non binary files
// 
// pub fn parse_gltf_to_raw(
//     path: &str,
//     texture_cache: &mut TextureCache,
//     mesh_cache: &mut MeshCache,
// ) -> Result<Rc<RefCell<gpu_data::Node>>, String> {
//     log::info!("\nLoading Mesh: {:?}", path);
// 
//     // let path = std::path::Path::new(path);
//     //    let file = File::open(path).unwrap();
//     //  let reader = BufReader::new(file);
//     //  let gltf_data = Gltf::from_reader(reader).unwrap();
// 
//     //  let buffer_data = load_buffers(&gltf_data, &path).unwrap();
// 
//     let (gltf_data, buffer_data, images) =
//         gltf::import(path).map_err(|err| format!("Error loading GLTF file: {:?}", err))?;
// 
//     let mut parsed_meshes = Vec::<MeshMeta>::new();
//     let parsed_materials: Vec<gltf::Material> = gltf_data.materials().collect();
// 
//     let mut unnamed_mesh_index = 0;
//     for mesh in gltf_data.meshes() {
//         let name = if let Some(name) = mesh.name() {
//             name.to_string()
//         } else {
//             unnamed_mesh_index += 1;
//             format!("unnamed_mesh_{:?}", unnamed_mesh_index - 1).to_string()
//         };
// 
//         let mut tmp_indices = Vec::<u32>::new();
//         let mut tmp_vertices = Vec::<Vertex>::new();
//         let mut surfaces = Vec::<SurfaceMeta>::with_capacity(mesh.primitives().len());
// 
//         for primitive in mesh.primitives() {
//             let reader = primitive.reader(|buffer| Some(buffer_data[buffer.index()].0.as_slice()));
// 
//             // Reserve to alloc once per surface
//             tmp_indices.reserve(reader.read_indices().iter().len());
//             tmp_vertices.reserve(reader.read_positions().iter().len());
// 
//             // All vertices and indices are stored in a flat array for the mesh as a whole,
//             // each surface has a start index, and indices count, start index is also used
//             // to index into other arrays while mapping
//             let start_index = tmp_vertices.len();
//             let count = primitive.indices().unwrap().count();
// 
//             let material = primitive.material();
// 
//             // INDICES
//             if let Some(indices_reader) = reader.read_indices() {
//                 parse_indices(indices_reader, start_index, &mut tmp_indices);
//             } else {
//                 return Err("No indices found".to_string());
//             }
// 
//             // VERTICES
//             if let Some(vertex_reader) = reader.read_positions() {
//                 parse_vertices(vertex_reader, &mut tmp_vertices);
//             } else {
//                 return Err("No vertices found".to_string());
//             }
// 
//             // NORMALS
//             if let Some(normal_reader) = reader.read_normals() {
//                 parse_normals(normal_reader, start_index, &mut tmp_vertices);
//             } else {
//                 info!("Generating smooth normals for surface in mesh: {:?}", name);
//                 generate_smooth_normals(
//                     &mut tmp_vertices, // Pass whole array as indices are array indices
//                     &tmp_indices[start_index..start_index + count],// Use only applicable surface indices
//                 );
//             }
// 
//             // UV COORDS
//             if let Some(uv_reader) = reader.read_tex_coords(0) {
//                 parse_texture_coords(uv_reader, start_index, &mut tmp_vertices);
//             } else {
//                 info!("Using default uv of (0.0, 0.0) for surface in mesh: {:?}", name);
//             }
// 
//             // TANGENTS
//             if let Some(tangent_reader) = reader.read_tangents() {
//                 parse_tangents(tangent_reader, start_index, &mut tmp_vertices);
//             } else {
//                 info!("Generating mikktspace tangents for surface in mesh: {:?}", name);
//                 let mut primitive = Primitive {
//                     vertices: &mut tmp_vertices, // Pass whole array as indices are array indices
//                     indices: &tmp_indices[start_index..start_index + count], // Use only applicable surface indices
//                 };
//                 if let false = generate_tangents(&mut primitive) {
//                     return Err("Failed to generate mikktspace tangents".to_string())
//                 }
//             }
// 
//             // COLORS
//             if let Some(color_reader) = reader.read_colors(0) {
//                 parse_colors(color_reader, start_index, &mut tmp_vertices);
//             } else {
//                 info!("Using default color of (1.0, 1.0, 1.0, 1.0) for surface in mesh: {:?}", name);
//             }
// 
//             // Push parsed info as new surface
//             surfaces.push(SurfaceMeta {
//                 start_index: start_index as u32,
//                 count: count as u32,
//                 material_index: primitive.material().index().map(|idx| idx as u32),
//             })
//         }
// 
//         //  Push parsed surfaces as new mesh
//         parsed_meshes.push(MeshMeta {
//             name,
//             indices: tmp_indices,
//             vertices: tmp_vertices,
//             surfaces,
//         })
//     }
// 
//     // Load material into texture cache, return hashmap mapping gltf index -> texture cache index for later use
//     let mapped_materials = map_materials(&parsed_materials, &images, texture_cache)?;
// 
//     // Load meshed into mesh cache, update surface material to texture cache id and
//     // return mapping of parsed_meshed -> mesh cache index for later use
//     let mapped_meshes = map_meshes(parsed_meshes, mesh_cache, &mapped_materials);
// 
//     // Parse nodes from gltf file, construct NodeMeta from needed data updated with the mesh id stored in the mesh cache
//     // Returns vec of NodeMeta of all constructed nodes, and a Vec of usize of all top most nodes for later mapping
//     let (parsed_nodes, top_node_indices) = parse_nodes(&gltf_data, &mapped_meshes);
// 
//     //Construct graph representation of the models meshes and return the root node
//     let graph_root = construct_node_graph(parsed_nodes, top_node_indices);
//     Ok(graph_root)
// }
// 
// fn parse_indices(reader: ReadIndices, start_index: usize, tmp_indices: &mut Vec<u32>) {
//     let start_index = start_index as u32;
//     match reader {
//         ReadIndices::U8(val) => {
//             tmp_indices.extend(val.map(|idx| start_index + idx as u32));
//         }
//         ReadIndices::U16(val) => {
//             tmp_indices.extend(val.map(|idx| start_index + idx as u32));
//         }
//         ReadIndices::U32(val) => {
//             tmp_indices.extend(val.map(|idx| start_index + idx));
//         }
//     }
// }
// 
// fn parse_vertices(reader: ReadPositions, tmp_vertices: &mut Vec<Vertex>) {
//     match reader {
//         ReadPositions::Standard(pos_iter) => {
//             for pos in pos_iter {
//                 let vert = Vertex {
//                     position: glam::Vec3::from_array(pos),
//                     normal: glam::vec3(1.0, 0.0, 0.0),
//                     color: glam::Vec4::ONE,
//                     uv_x: 0.0,
//                     uv_y: 0.0,
//                     tangent: Default::default(),
//                 };
//                 tmp_vertices.push(vert);
//             }
//         }
// 
//         ReadPositions::Sparse(pos_iter) => {
//             panic!("Sparse not implemented");
//             // for pos in pos_iter {
//             //     let vert = Vertex {
//             //         position: glam::Vec3::from_array(pos),
//             //         normal: glam::vec3(1.0, 0.0, 0.0),
//             //         color: glam::Vec4::ONE,
//             //         uv_x: 0.0,
//             //         uv_y: 0.0
//             //     };
//             //     tmp_vertices.push(vert);
//             // }
//         }
//     }
// }
// 
// fn parse_normals(reader: ReadNormals, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
//     match reader {
//         ReadNormals::Standard(norm_iter) => {
//             for (idx, norm) in norm_iter.enumerate() {
//                 tmp_vertices[start_index + idx].normal = glam::Vec3::from_array(norm);
//             }
//         }
//         ReadNormals::Sparse(norm_iter) => {
//             panic!("Sparse not implemented");
//             // for (idx, norm) in norm_iter.enumerate() {
//             //     tmp_vertices[start_index + idx].normal = glam::Vec3::from_array(norm);
//             // }
//         }
//     }
// }
// 
// fn generate_smooth_normals(tmp_vertices: &mut [Vertex], tmp_indices: &[u32]) {
//     // Initialize a vector to accumulate normals
//     let mut accumulated_normals = vec![glam::Vec3::ZERO; tmp_vertices.len()];
// 
//     // Calculate face normals and accumulate them for each vertex
//     for chunk in tmp_indices.chunks(3) {
//         let v0 = tmp_vertices[chunk[0] as usize].position;
//         let v1 = tmp_vertices[chunk[1] as usize].position;
//         let v2 = tmp_vertices[chunk[2] as usize].position;
// 
//         let edge1 = v1 - v0;
//         let edge2 = v2 - v0;
//         let normal = edge1.cross(edge2);
// 
//         for &index in chunk {
//             accumulated_normals[index as usize] += normal;
//         }
//     }
// 
//     // Normalize accumulated normals and update vertices
//     for (vertex, accumulated_normal) in tmp_vertices.iter_mut().zip(accumulated_normals.iter()) {
//         vertex.normal = accumulated_normal.normalize_or_zero();
//     }
// }
// 
// fn parse_texture_coords(reader: ReadTexCoords, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
//     match reader {
//         ReadTexCoords::U8(cord_iter) => {
//             for (idx, cord) in cord_iter.enumerate() {
//                 tmp_vertices[start_index + idx].uv_x = cord[0] as f32 / 255.0;
//                 tmp_vertices[start_index + idx].uv_y = cord[1] as f32 / 255.0;
//             }
//         }
//         ReadTexCoords::U16(cord_iter) => {
//             for (idx, cord) in cord_iter.enumerate() {
//                 tmp_vertices[start_index + idx].uv_x = cord[0] as f32 / 65535.0;
//                 tmp_vertices[start_index + idx].uv_y = cord[1] as f32 / 65535.0;
//             }
//         }
//         ReadTexCoords::F32(cord_iter) => {
//             for (idx, cord) in cord_iter.enumerate() {
//                 tmp_vertices[start_index + idx].uv_x = cord[0];
//                 tmp_vertices[start_index + idx].uv_y = cord[1];
//             }
//         }
//     }
// }
// 
// fn parse_tangents(reader: ReadTangents, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
//     match reader {
//         ReadTangents::Standard(tan_iter) => {
//             for (idx, tangent) in tan_iter.enumerate() {
//                 tmp_vertices[start_index + idx].tangent = glam::Vec4::from_array(tangent);
//             }
//         }
//         ReadTangents::Sparse(_) => {
//             panic!("Sparse not implemented")
//         }
//     }
// }
// 
// fn parse_colors(reader: ReadColors, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
//     match reader {
//         ReadColors::RgbF32(color_iter) => {
//             for (idx, color) in color_iter.enumerate() {
//                 tmp_vertices[start_index + idx].color = glam::Vec3::from_array(color).extend(1.0);
//             }
//         }
//         ReadColors::RgbaF32(color_iter) => {
//             for (idx, color) in color_iter.enumerate() {
//                 tmp_vertices[start_index + idx].color = glam::Vec4::from_array(color)
//             }
//         }
//         ReadColors::RgbU8(color_iter) => {
//             for (idx, color) in color_iter.enumerate() {
//                 let color: [u8; 3] = color.into();
//                 let normalized_color = [
//                     color[0] as f32 / 255.0,
//                     color[1] as f32 / 255.0,
//                     color[2] as f32 / 255.0,
//                     1.0,
//                 ];
//                 tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
//             }
//         }
//         ReadColors::RgbaU8(color_iter) => {
//             for (idx, color) in color_iter.enumerate() {
//                 let color: [u8; 4] = color.into();
//                 let normalized_color = [
//                     color[0] as f32 / 255.0,
//                     color[1] as f32 / 255.0,
//                     color[2] as f32 / 255.0,
//                     color[3] as f32 / 255.0,
//                 ];
//                 tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
//             }
//         }
//         ReadColors::RgbU16(color_iter) => {
//             for (idx, color) in color_iter.enumerate() {
//                 let color: [u16; 3] = color.into();
//                 let normalized_color = [
//                     color[0] as f32 / 65535.0,
//                     color[1] as f32 / 65535.0,
//                     color[2] as f32 / 65535.0,
//                     1.0,
//                 ];
//                 tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
//             }
//         }
//         ReadColors::RgbaU16(color_iter) => {
//             for (idx, color) in color_iter.enumerate() {
//                 let color: [u16; 4] = color.into();
//                 let normalized_color = [
//                     color[0] as f32 / 65535.0,
//                     color[1] as f32 / 65535.0,
//                     color[2] as f32 / 65535.0,
//                     color[3] as f32 / 65535.0,
//                 ];
//                 tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
//             }
//         }
//     }
// }
// 
// fn map_materials(
//     parsed_materials: &Vec<Material>,
//     images: &Vec<Data>,
//     texture_cache: &mut TextureCache,
// ) -> Result<HashMap<u32, u32>, String> {
//     let mut mapped_materials = HashMap::<u32, u32>::with_capacity(parsed_materials.len());
//     let mut unnamed_mat_index = 0;
// 
//     for material in parsed_materials {
//         let name = material
//             .name()
//             .or({
//                 unnamed_mat_index += 1;
//                 Some(format!("unnamed_material_{}", unnamed_mat_index - 1).as_str())
//             })
//             .unwrap()
//             .to_string();
// 
//         let (base_color_tex_id, base_color_factor) =
//             if let Some(tex) = material.pbr_metallic_roughness().base_color_texture() {
//                 (
//                     Some(tex.texture().source().index()),
//                     Some(glam::Vec4::from_array(
//                         material.pbr_metallic_roughness().base_color_factor(),
//                     )),
//                 )
//             } else {
//                 (None, None) // TODO add better default
//             };
// 
//         let (metallic_roughness_tex_id, metallic_factor, roughness_factor) = if let Some(tex) =
//             material
//                 .pbr_metallic_roughness()
//                 .metallic_roughness_texture()
//         {
//             (
//                 Some(tex.texture().source().index()),
//                 Some(material.pbr_metallic_roughness().metallic_factor()),
//                 Some(material.pbr_metallic_roughness().roughness_factor()),
//             )
//         } else {
//             (None, None, None)
//         };
// 
//         let (normal_id, normal_scale) = if let Some(tex) = material.normal_texture() {
//             (Some(tex.texture().source().index()), Some(tex.scale()))
//         } else {
//             (None, None)
//         };
// 
//         let (occlusion_id, occlusion_strength) = if let Some(tex) = material.occlusion_texture() {
//             (Some(tex.texture().source().index()), Some(tex.strength()))
//         } else {
//             (None, None)
//         };
// 
//         let (emissive_id, emissive_factor) = if let Some(tex) = material.emissive_texture() {
//             (
//                 Some(tex.texture().source().index()),
//                 Some(material.emissive_factor()),
//             )
//         } else {
//             (None, None)
//         };
// 
//         let alpha_mode = match material.alpha_mode() {
//             AlphaMode::Opaque => gpu_data::AlphaMode::Opaque,
//             AlphaMode::Mask => gpu_data::AlphaMode::Mask,
//             AlphaMode::Blend => gpu_data::AlphaMode::Blend,
//         };
// 
//         let alpha_cutoff = material.alpha_cutoff().unwrap_or_else(|| 1.0);
// 
//         let (base_color_tex_id, base_color_factor) = if let Some(bc_id) = base_color_tex_id {
//             let bc_factor = base_color_factor.unwrap();
//             let data = images.get(bc_id).ok_or_else(|| {
//                 format!(
//                     "Could not locate texture index {} for material: {:?}",
//                     bc_id, name
//                 )
//             })?;
// 
//             let texture_data = gpu_data::TextureMeta::from_gltf_texture(data);
//             let texture_id = texture_cache.add_texture(texture_data);
//             (texture_id, bc_factor)
//         } else {
//             log::info!("Used Default Base Color For Material: {:?}", name);
//             (
//                 TextureCache::DEFAULT_COLOR_TEX,
//                 glam::vec4(1.0, 1.0, 1.0, 1.0),
//             )
//         };
// 
//         let (metallic_roughness_tex_id, metallic_factor, roughness_factor) =
//             if let Some(mr_id) = metallic_roughness_tex_id {
//                 let met_factor = metallic_factor.unwrap();
//                 let rough_factor = roughness_factor.unwrap();
//                 let data = images.get(mr_id).ok_or_else(|| {
//                     format!(
//                         "Could not locate texture index {} for material: {:?}",
//                         mr_id, name
//                     )
//                 })?;
// 
//                 let texture_data = gpu_data::TextureMeta::from_gltf_texture(data);
//                 let texture_id = texture_cache.add_texture(texture_data);
//                 (texture_id, met_factor, rough_factor)
//             } else {
//                 log::info!("Used Default Metallic Roughness For Material: {:?}", name);
// 
//                 (TextureCache::DEFAULT_ROUGH_TEX, 0.0, 1.0)
//             };
// 
//         let normal_map = if let Some(norm_id) = normal_id {
//             let scale = normal_scale.unwrap();
//             let data = images.get(norm_id).ok_or_else(|| {
//                 format!(
//                     "Could not locate texture index {} for material: {:?}",
//                     norm_id, name
//                 )
//             })?;
// 
//             let texture_data = gpu_data::TextureMeta::from_gltf_texture(data);
//             let texture_id = texture_cache.add_texture(texture_data);
//             Some(NormalMap { texture_id, scale })
//         } else {
//             None
//         };
// 
//         let occlusion_map = if let Some(occ_id) = occlusion_id {
//             let strength = occlusion_strength.unwrap();
//             let data = images.get(occ_id).ok_or_else(|| {
//                 format!(
//                     "Could not locate texture index {} for material: {:?}",
//                     occ_id, name
//                 )
//             })?;
// 
//             let texture_data = gpu_data::TextureMeta::from_gltf_texture(data);
//             let texture_id = texture_cache.add_texture(texture_data);
//             Some(OcclusionMap {
//                 texture_id,
//                 strength,
//             })
//         } else {
//             None
//         };
// 
//         let emissive_map = if let Some(e_id) = emissive_id {
//             let factor = glam::Vec3::from_array(emissive_factor.unwrap());
//             let data = images.get(e_id).ok_or_else(|| {
//                 format!(
//                     "Could not locate texture index {} for material: {:?}",
//                     e_id, name
//                 )
//             })?;
// 
//             let texture_data = gpu_data::TextureMeta::from_gltf_texture(data);
//             let texture_id = texture_cache.add_texture(texture_data);
//             Some(EmissiveMap { texture_id, factor })
//         } else {
//             None
//         };
// 
//         let mat_data = gpu_data::MaterialMeta {
//             base_color_factor,
//             base_color_tex_id,
//             metallic_factor,
//             roughness_factor,
//             metallic_roughness_tex_id,
//             alpha_mode,
//             alpha_cutoff,
//             normal_map,
//             occlusion_map,
//             emissive_map,
//         };
// 
//         let mat_id = texture_cache.add_material(mat_data);
//         mapped_materials.insert(material.index().unwrap() as u32, mat_id);
//     }
//     Ok(mapped_materials)
// }
// 
// fn map_meshes(
//     mut parsed_meshes: Vec<MeshMeta>,
//     mesh_cache: &mut MeshCache,
//     mapped_materials: &HashMap<u32, u32>,
// ) -> HashMap<u32, u32> {
//     let mut mapped_meshes = HashMap::<u32, u32>::with_capacity(parsed_meshes.len());
//     for (idx, mut mesh) in parsed_meshes.drain(..).enumerate() {
//         for surface in &mut mesh.surfaces {
//             if let Some(mat_index) = surface.material_index {
//                 if let Some(cache_index) = mapped_materials.get(&mat_index) {
//                     surface.material_index = Some(*cache_index);
//                 }
//             }
//         }
// 
//         let mesh_index = mesh_cache.add_mesh(mesh);
//         mapped_meshes.insert(idx as u32, mesh_index);
//     }
//     mapped_meshes
// }
// 
// fn parse_nodes(
//     gltf_data: &Document,
//     mapped_meshes: &HashMap<u32, u32>,
// ) -> (Vec<NodeMeta>, Vec<usize>) {
//     let mut top_node_indices: Vec<Option<usize>> =
//         gltf_data.nodes().map(|n| Some(n.index())).collect();
//     let mut parsed_nodes = Vec::<NodeMeta>::with_capacity(gltf_data.nodes().count());
// 
//     let mut unnamed_node_idx = 0;
//     for node in gltf_data.nodes() {
//         let name = if let Some(name) = node.name() {
//             name.to_string()
//         } else {
//             unnamed_node_idx += 1;
//             format!("unnamed_node_{:?}", unnamed_node_idx).to_string()
//         };
// 
//         let mesh_index = node.mesh().map(|m| m.index());
// 
//         let (translation, rotation, scale) = node.transform().decomposed();
// 
//         let transform = Transform::new_vulkan_adjusted(translation, rotation, scale);
// 
//         let og_matrix = {
//             let tl = glam::Vec3::new(translation[0], translation[1], translation[2]);
//             let rot = glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
//             let sc = glam::Vec3::new(scale[0], scale[1], scale[2]);
// 
//             let tm = glam::Mat4::from_translation(tl);
//             let rm = glam::Mat4::from_quat(rot);
//             let sm = glam::Mat4::from_scale(sc);
//             tm * rm * sm
//         };
// 
//         let children: Vec<usize> = node
//             .children()
//             .map(|c| {
//                 let index = c.index();
//                 top_node_indices[index] = None;
//                 index
//             })
//             .collect();
// 
//         let mapped_mesh_index = mesh_index
//             .and_then(|index| mapped_meshes.get(&(index as u32)))
//             .copied();
// 
//         let node_data = NodeMeta {
//             name,
//             og_matrix,
//             mesh_index: mapped_mesh_index,
//             local_transform: transform,
//             children: children.into_iter().map(|idx| idx as u32).collect(),
//         };
// 
//         parsed_nodes.push(node_data);
//     }
//     // flatted to removed None indexes that were removed
//     (
//         parsed_nodes,
//         top_node_indices.into_iter().flatten().collect(),
//     )
// }
// 
// fn construct_node_graph(
//     parsed_nodes: Vec<NodeMeta>,
//     top_node_indices: Vec<usize>,
// ) -> Rc<RefCell<Node>> {
//     let root_node = gpu_data::Node::default();
// 
//     let root_node = Rc::new(RefCell::new(root_node));
// 
//     let root_children = top_node_indices
//         .iter()
//         .map(|&node_index| {
//             let parent_node = &parsed_nodes[node_index];
// 
//             let root_child = Rc::new(RefCell::new(gpu_data::Node {
//                 parent: Some(Rc::downgrade(&root_node)),
//                 children: vec![],
//                 meshes: parent_node.mesh_index,
//                 world_transform: parent_node.og_matrix,
//                 local_transform: parent_node.local_transform,
//                 dirty: true,
//             }));
// 
//             // Recursively set the children
//             let child_nodes = recur_children(&root_child, &parent_node.children, &parsed_nodes);
//             root_child.borrow_mut().children = child_nodes;
// 
//             root_child
//         })
//         .collect();
// 
//     root_node.borrow_mut().children = root_children;
//     root_node
// }
// 
// fn recur_children(
//     parent: &Rc<RefCell<gpu_data::Node>>,
//     children: &[u32],
//     parsed_nodes: &[NodeMeta],
// ) -> Vec<Rc<RefCell<gpu_data::Node>>> {
//     // Terminate on leaf nodes with no children
//     if children.is_empty() {
//         return vec![];
//     }
// 
//     children
//         .iter()
//         .map(|&child_index| {
//             let child_meta = &parsed_nodes[child_index as usize];
// 
//             let child_node = Rc::new(RefCell::new(gpu_data::Node {
//                 parent: Some(Rc::downgrade(parent)),
//                 children: vec![],
//                 meshes: child_meta.mesh_index,
//                 world_transform: child_meta.og_matrix,
//                 local_transform: child_meta.local_transform,
//                 dirty: true,
//             }));
// 
//             // Recursively construct child nodes
//             let child_children = recur_children(&child_node, &child_meta.children, parsed_nodes);
//             child_node.borrow_mut().children = child_children;
// 
//             child_node
//         })
//         .collect()
// }
// 
// pub fn gltf_format_to_vk_format(format: Format) -> vk::Format {
//     match format {
//         Format::R8 => vk::Format::R8_UNORM,
//         Format::R8G8 => vk::Format::R8G8_UNORM,
//         Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
//         Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
//         Format::R16 => vk::Format::R16_UNORM,
//         Format::R16G16 => vk::Format::R16G16_UNORM,
//         Format::R16G16B16 => vk::Format::R16G16B16_UNORM,
//         Format::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
//         Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
//         Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
//     }
// }
// 
// // Used as a temp structure if needing to use mikktspace for generating tangents
// pub struct Primitive<'a> {
//     pub vertices: &'a mut [Vertex],
//     pub indices: &'a [u32],
// }
// 
// impl<'a> Geometry for Primitive<'a> {
//     fn num_faces(&self) -> usize {
//         self.indices.len() / 3
//     }
// 
//     fn num_vertices_of_face(&self, _face: usize) -> usize {
//         3
//     }
// 
//     fn position(&self, face: usize, vert: usize) -> [f32; 3] {
//         let index = self.indices[face * 3 + vert] as usize;
//         self.vertices[index].position.to_array()
//     }
// 
//     fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
//         let index = self.indices[face * 3 + vert] as usize;
//         self.vertices[index].normal.to_array()
//     }
// 
//     fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
//         let index = self.indices[face * 3 + vert] as usize;
//         [self.vertices[index].uv_x, self.vertices[index].uv_y]
//     }
// 
//     fn set_tangent(
//         &mut self,
//         tangent: [f32; 3],
//         bi_tangent: [f32; 3],
//         f_mag_s: f32,
//         f_mag_t: f32,
//         bi_tangent_preserves_orientation: bool,
//         face: usize,
//         vert: usize,
//     ) {
//         let index = self.indices[face * 3 + vert] as usize;
//         let sign = if bi_tangent_preserves_orientation {
//             1.0
//         } else {
//             -1.0
//         };
//         self.vertices[index].tangent = glam::Vec4::new(tangent[0], tangent[1], tangent[2], sign);
//     }
// 
//     fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
//         let index = self.indices[face * 3 + vert] as usize;
//         self.vertices[index].tangent = glam::Vec4::from_array(tangent);
//     }
// }
