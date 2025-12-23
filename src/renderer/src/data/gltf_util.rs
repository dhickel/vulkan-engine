use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ash::vk;
use glam::Vec2;
use gltf::{Document, Material};
use gltf::image::{Data, Format};
use gltf::material::AlphaMode;
use gltf::mesh::util::{
    ReadColors, ReadIndices, ReadNormals, ReadPositions, ReadTangents, ReadTexCoords,
};
use gltf::mesh::Reader;
use log::info;
use mikktspace::{generate_tangents, Geometry};

use crate::data::data_cache::{MeshCache, TextureCache};
use crate::data::gpu_data;
use crate::data::gpu_data::{
    MaterialMeta, MeshMeta, Node, NodeMeta, TextureMeta, Transform, Vertex,
};

// TODO: instead of using a index/vertex array for each mesh in a file, use a single pair for the
//  whole file and index the meshes and surfaces into it.

pub fn parse_gltf_to_raw(
    path: &str,
    texture_cache: &mut TextureCache,
    mesh_cache: &mut MeshCache,
) -> Result<Rc<RefCell<gpu_data::Node>>, String> {
    log::info!("\nLoading Mesh: {:?}", path);

    let (gltf_data, buffer_data, images) =
        gltf::import(path).map_err(|err| format!("Error loading GLTF file: {:?}", err))?;

    // Load material into texture cache, return hashmap mapping gltf index -> texture cache index for later use
    let mapped_materials = map_materials(&gltf_data.materials().collect(), &images, texture_cache)?;

    let mut mapped_meshes = HashMap::<u32, Vec<u32>>::new();

    let mut unnamed_mesh_index = 0;
    for mesh in gltf_data.meshes() {
        let name_prefix = if let Some(name) = mesh.name() {
            name.to_string()
        } else {
            unnamed_mesh_index += 1;
            format!("unnamed_mesh_{:?}", unnamed_mesh_index - 1)
        };

        let mut mesh_indices_for_node = Vec::new();

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            let name = format!("{}_prim_{}", name_prefix, primitive_index);
            let reader = primitive.reader(|buffer| Some(buffer_data[buffer.index()].0.as_slice()));

            let mut tmp_indices = Vec::<u32>::new();
            let mut tmp_vertices = Vec::<Vertex>::new();

            // Reserve to alloc once per surface
            if let Some(iter) = reader.read_indices() {
                 tmp_indices.reserve(match iter {
                     ReadIndices::U8(i) => i.len(),
                     ReadIndices::U16(i) => i.len(),
                     ReadIndices::U32(i) => i.len(),
                 });
            }
            if let Some(iter) = reader.read_positions() {
                tmp_vertices.reserve(iter.len());
            }

            // All vertices and indices are stored in a flat array for the mesh as a whole,
            // each surface has a start index, and indices count, start index is also used
            // to index into other arrays while mapping
            let start_index = tmp_vertices.len(); // Should be 0 since we create new Vec per primitive

            // INDICES
            if let Some(indices_reader) = reader.read_indices() {
                parse_indices(indices_reader, start_index, &mut tmp_indices);
            } else {
                return Err(format!("No indices found for mesh {}", name));
            }
             let count = tmp_indices.len() as u32; // This is the index count

            // VERTICES
            if let Some(vertex_reader) = reader.read_positions() {
                parse_vertices(vertex_reader, &mut tmp_vertices);
            } else {
                return Err(format!("No vertices found for mesh {}", name));
            }

            // NORMALS
            if let Some(normal_reader) = reader.read_normals() {
                parse_normals(normal_reader, start_index, &mut tmp_vertices);
            } else {
                info!("Generating smooth normals for surface in mesh: {:?}", name);
                generate_smooth_normals(
                    &mut tmp_vertices,
                    &tmp_indices,
                );
            }

            // UV COORDS
            if let Some(uv_reader) = reader.read_tex_coords(0) {
                parse_texture_coords(uv_reader, start_index, &mut tmp_vertices);
            } else {
                info!("Using default uv of (0.0, 0.0) for surface in mesh: {:?}", name);
            }

            // TANGENTS
            if let Some(tangent_reader) = reader.read_tangents() {
                parse_tangents(tangent_reader, start_index, &mut tmp_vertices);
            } else {
                info!("Generating mikktspace tangents for surface in mesh: {:?}", name);
                let mut primitive_geom = Primitive {
                    vertices: &mut tmp_vertices,
                    indices: &tmp_indices,
                };
                if !generate_tangents(&mut primitive_geom) {
                     log::warn!("Failed to generate mikktspace tangents for {}", name);
                     // return Err("Failed to generate mikktspace tangents".to_string())
                }
            }

            // COLORS
            if let Some(color_reader) = reader.read_colors(0) {
                parse_colors(color_reader, start_index, &mut tmp_vertices);
            } else {
                info!("Using default color of (1.0, 1.0, 1.0, 1.0) for surface in mesh: {:?}", name);
            }

            let material_index = primitive.material().index().map(|idx| idx as u32);
            let mapped_material_index = material_index.and_then(|idx| mapped_materials.get(&idx)).copied();

            let mesh_meta = MeshMeta {
                name: name.clone(),
                indices: tmp_indices,
                vertices: tmp_vertices,
                material_index: mapped_material_index,
            };

            let mesh_id = mesh_cache.add(mesh_meta);
            mesh_indices_for_node.push(mesh_id);
        }

        mapped_meshes.insert(mesh.index() as u32, mesh_indices_for_node);
    }

    // Parse nodes from gltf file, construct NodeMeta from needed data updated with the mesh id stored in the mesh cache
    // Returns vec of NodeMeta of all constructed nodes, and a Vec of usize of all top most nodes for later mapping
    let (parsed_nodes, top_node_indices) = parse_nodes(&gltf_data, &mapped_meshes);

    //Construct graph representation of the models meshes and return the root node
    let graph_root = construct_node_graph(parsed_nodes, top_node_indices);
    Ok(graph_root)
}

fn from_gltf_texture(data: &Data) -> TextureMeta {
    let format = gltf_format_to_vk_format(data.format);
    TextureMeta {
        bytes: data.pixels.clone(),
        width: data.width,
        height: data.height,
        format,
        mips_levels: 1,
        uv_index: 0,
    }
}

fn parse_indices(reader: ReadIndices, start_index: usize, tmp_indices: &mut Vec<u32>) {
    let start_index = start_index as u32;
    match reader {
        ReadIndices::U8(val) => {
            tmp_indices.extend(val.map(|idx| start_index + idx as u32));
        }
        ReadIndices::U16(val) => {
            tmp_indices.extend(val.map(|idx| start_index + idx as u32));
        }
        ReadIndices::U32(val) => {
            tmp_indices.extend(val.map(|idx| start_index + idx));
        }
    }
}

fn parse_vertices(reader: ReadPositions, tmp_vertices: &mut Vec<Vertex>) {
    match reader {
        ReadPositions::Standard(pos_iter) => {
            for pos in pos_iter {
                let mut vert = Vertex::default();
                vert.position = glam::Vec3::from_array(pos);
                vert.normal = glam::vec3(1.0, 0.0, 0.0);
                vert.color = glam::Vec4::ONE;
                vert.uv0_x = 0.0;
                vert.uv0_y = 0.0;
                // tangent, joints, weights are default
                tmp_vertices.push(vert);
            }
        }

        ReadPositions::Sparse(_) => {
            panic!("Sparse not implemented");
        }
    }
}

fn parse_normals(reader: ReadNormals, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
    match reader {
        ReadNormals::Standard(norm_iter) => {
            for (idx, norm) in norm_iter.enumerate() {
                tmp_vertices[start_index + idx].normal = glam::Vec3::from_array(norm);
            }
        }
        ReadNormals::Sparse(_) => {
            panic!("Sparse not implemented");
        }
    }
}

fn generate_smooth_normals(tmp_vertices: &mut [Vertex], tmp_indices: &[u32]) {
    // Initialize a vector to accumulate normals
    let mut accumulated_normals = vec![glam::Vec3::ZERO; tmp_vertices.len()];

    // Calculate face normals and accumulate them for each vertex
    for chunk in tmp_indices.chunks(3) {
        if chunk.len() < 3 { continue; }
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;

        let v0 = tmp_vertices[i0].position;
        let v1 = tmp_vertices[i1].position;
        let v2 = tmp_vertices[i2].position;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2); // Not normalized yet to weight by area

        accumulated_normals[i0] += normal;
        accumulated_normals[i1] += normal;
        accumulated_normals[i2] += normal;
    }

    // Normalize accumulated normals and update vertices
    for (vertex, accumulated_normal) in tmp_vertices.iter_mut().zip(accumulated_normals.iter()) {
        vertex.normal = accumulated_normal.normalize_or_zero();
    }
}

fn parse_texture_coords(reader: ReadTexCoords, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
    match reader {
        ReadTexCoords::U8(cord_iter) => {
            for (idx, cord) in cord_iter.enumerate() {
                tmp_vertices[start_index + idx].uv0_x = cord[0] as f32 / 255.0;
                tmp_vertices[start_index + idx].uv0_y = cord[1] as f32 / 255.0;
            }
        }
        ReadTexCoords::U16(cord_iter) => {
            for (idx, cord) in cord_iter.enumerate() {
                tmp_vertices[start_index + idx].uv0_x = cord[0] as f32 / 65535.0;
                tmp_vertices[start_index + idx].uv0_y = cord[1] as f32 / 65535.0;
            }
        }
        ReadTexCoords::F32(cord_iter) => {
            for (idx, cord) in cord_iter.enumerate() {
                tmp_vertices[start_index + idx].uv0_x = cord[0];
                tmp_vertices[start_index + idx].uv0_y = cord[1];
            }
        }
    }
}

fn parse_tangents(reader: ReadTangents, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
    match reader {
        ReadTangents::Standard(tan_iter) => {
            for (idx, tangent) in tan_iter.enumerate() {
                tmp_vertices[start_index + idx].tangent = glam::Vec4::from_array(tangent);
            }
        }
        ReadTangents::Sparse(_) => {
            panic!("Sparse not implemented")
        }
    }
}

fn parse_colors(reader: ReadColors, start_index: usize, tmp_vertices: &mut Vec<Vertex>) {
    match reader {
        ReadColors::RgbF32(color_iter) => {
            for (idx, color) in color_iter.enumerate() {
                tmp_vertices[start_index + idx].color = glam::Vec3::from_array(color).extend(1.0);
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
                tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
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
                tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
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
                tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
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
                tmp_vertices[start_index + idx].color = glam::Vec4::from_array(normalized_color);
            }
        }
    }
}

fn map_materials(
    parsed_materials: &Vec<Material>,
    images: &Vec<Data>,
    texture_cache: &mut TextureCache,
) -> Result<HashMap<u32, u32>, String> {
    let mut mapped_materials = HashMap::<u32, u32>::with_capacity(parsed_materials.len());
    let mut unnamed_mat_index = 0;

    for material in parsed_materials {
        let name = material
            .name()
            .or({
                unnamed_mat_index += 1;
                Some(format!("unnamed_material_{}", unnamed_mat_index - 1).as_str())
            })
            .unwrap()
            .to_string();

        let mut mat_data = MaterialMeta::default();

        // Base Color
        if let Some(tex) = material.pbr_metallic_roughness().base_color_texture() {
            let tex_id = tex.texture().source().index();
            let factor = glam::Vec4::from_array(
                material.pbr_metallic_roughness().base_color_factor(),
            );

            let data = images.get(tex_id).ok_or_else(|| {
                format!(
                    "Could not locate texture index {} for material: {:?}",
                    tex_id, name
                )
            })?;

            let texture_data = from_gltf_texture(data);
            let texture_id = texture_cache.add_texture(texture_data);

            mat_data.add_base_color(texture_id, factor, tex.tex_coord());
        } else {
             mat_data.material_values.base_color_factor = glam::Vec4::from_array(
                material.pbr_metallic_roughness().base_color_factor(),
            );
        }

        // Metallic Roughness
        if let Some(tex) = material
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
        {
            let tex_id = tex.texture().source().index();
            let metallic_factor = material.pbr_metallic_roughness().metallic_factor();
            let roughness_factor = material.pbr_metallic_roughness().roughness_factor();

            let data = images.get(tex_id).ok_or_else(|| {
                format!(
                    "Could not locate texture index {} for material: {:?}",
                    tex_id, name
                )
            })?;

            let texture_data = from_gltf_texture(data);
            let texture_id = texture_cache.add_texture(texture_data);

            mat_data.add_metallic_roughness(texture_id, metallic_factor, roughness_factor, tex.tex_coord());
        } else {
            mat_data.material_values.metallic_factor = material.pbr_metallic_roughness().metallic_factor();
            mat_data.material_values.roughness_factor = material.pbr_metallic_roughness().roughness_factor();
        }

        // Normal
        if let Some(tex) = material.normal_texture() {
             let tex_id = tex.texture().source().index();
             let scale = tex.scale();

             let data = images.get(tex_id).ok_or_else(|| {
                format!(
                    "Could not locate texture index {} for material: {:?}",
                    tex_id, name
                )
            })?;

             let texture_data = from_gltf_texture(data);
             let texture_id = texture_cache.add_texture(texture_data);

             mat_data.add_normal(texture_id, scale, tex.tex_coord());
        }

        // Occlusion
        if let Some(tex) = material.occlusion_texture() {
            let tex_id = tex.texture().source().index();
            let strength = tex.strength();

            let data = images.get(tex_id).ok_or_else(|| {
                format!(
                    "Could not locate texture index {} for material: {:?}",
                    tex_id, name
                )
            })?;

             let texture_data = from_gltf_texture(data);
             let texture_id = texture_cache.add_texture(texture_data);

             mat_data.add_occlusion(texture_id, strength, tex.tex_coord());
        }

        // Emissive
        if let Some(tex) = material.emissive_texture() {
            let tex_id = tex.texture().source().index();
            let factor = glam::Vec3::from_array(material.emissive_factor());

             let data = images.get(tex_id).ok_or_else(|| {
                format!(
                    "Could not locate texture index {} for material: {:?}",
                    tex_id, name
                )
            })?;

             let texture_data = from_gltf_texture(data);
             let texture_id = texture_cache.add_texture(texture_data);

             // Gltf doesn't seem to have emissive strength in PBR material directly?
             // Extensions like KHR_materials_emissive_strength might handle it, but standard material does not.
             // We'll use 1.0 as default strength or just use factor.
             mat_data.add_emissive(texture_id, factor, 1.0, tex.tex_coord());
        } else {
             mat_data.material_values.emissive_factor = glam::Vec3::from_array(material.emissive_factor()).extend(0.0);
        }

        mat_data.material_values.alpha_mask = match material.alpha_mode() {
            AlphaMode::Opaque => gpu_data::AlphaMode::Opaque.to_float_value(),
            AlphaMode::Mask => gpu_data::AlphaMode::Mask.to_float_value(),
            AlphaMode::Blend => gpu_data::AlphaMode::Blend.to_float_value(),
        };

        mat_data.material_values.alpha_mask_cutoff = material.alpha_cutoff().unwrap_or(0.5);

        let mat_id = texture_cache.add_material(mat_data);
        mapped_materials.insert(material.index().unwrap() as u32, mat_id);
    }
    Ok(mapped_materials)
}


fn parse_nodes(
    gltf_data: &Document,
    mapped_meshes: &HashMap<u32, Vec<u32>>,
) -> (Vec<NodeMeta>, Vec<usize>) {
    let mut top_node_indices: Vec<Option<usize>> =
        gltf_data.nodes().map(|n| Some(n.index())).collect();
    let mut parsed_nodes = Vec::<NodeMeta>::with_capacity(gltf_data.nodes().count());

    let mut unnamed_node_idx = 0;
    for node in gltf_data.nodes() {
        let name = if let Some(name) = node.name() {
            name.to_string()
        } else {
            unnamed_node_idx += 1;
            format!("unnamed_node_{:?}", unnamed_node_idx)
        };

        let mesh_indices = node.mesh().map(|m| mapped_meshes.get(&(m.index() as u32))).flatten().cloned().unwrap_or_default();

        let (translation, rotation, scale) = node.transform().decomposed();

        let transform = Transform::new_vulkan_adjusted(translation, rotation, scale);

        let og_matrix = {
            let tl = glam::Vec3::new(translation[0], translation[1], translation[2]);
            let rot = glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
            let sc = glam::Vec3::new(scale[0], scale[1], scale[2]);

            let tm = glam::Mat4::from_translation(tl);
            let rm = glam::Mat4::from_quat(rot);
            let sm = glam::Mat4::from_scale(sc);
            tm * rm * sm
        };

        let children: Vec<usize> = node
            .children()
            .map(|c| {
                let index = c.index();
                top_node_indices[index] = None;
                index
            })
            .collect();

        let node_data = NodeMeta {
            name,
            og_matrix,
            mesh_indices,
            local_transform: transform,
            children: children.into_iter().map(|idx| idx as u32).collect(),
        };

        parsed_nodes.push(node_data);
    }
    // flatted to removed None indexes that were removed
    (
        parsed_nodes,
        top_node_indices.into_iter().flatten().collect(),
    )
}

fn construct_node_graph(
    mut parsed_nodes: Vec<NodeMeta>,
    top_node_indices: Vec<usize>,
) -> Rc<RefCell<Node>> {
    let root_node = gpu_data::Node::default();

    let root_node = Rc::new(RefCell::new(root_node));

    let root_children = top_node_indices
        .iter()
        .map(|&node_index| {
            let (meshes, world_transform, local_transform, children) = {
                let parent_node = &mut parsed_nodes[node_index];
                (
                    parent_node.mesh_indices.clone(),
                    parent_node.og_matrix,
                    parent_node.local_transform.compose(),
                    parent_node.children.clone(),
                )
            };

            let root_child = Rc::new(RefCell::new(gpu_data::Node {
                parent: Some(Rc::downgrade(&root_node)),
                children: vec![],
                meshes,
                world_transform,
                local_transform,
                dirty: true,
            }));

            // Recursively set the children
            let child_nodes = recur_children(&root_child, &children, &mut parsed_nodes);
            root_child.borrow_mut().children = child_nodes;

            root_child
        })
        .collect();

    root_node.borrow_mut().children = root_children;
    root_node
}

fn recur_children(
    parent: &Rc<RefCell<gpu_data::Node>>,
    children: &[u32],
    parsed_nodes: &mut [NodeMeta],
) -> Vec<Rc<RefCell<gpu_data::Node>>> {
    // Terminate on leaf nodes with no children
    if children.is_empty() {
        return vec![];
    }

    children
        .iter()
        .map(|&child_index| {
            let (meshes, world_transform, local_transform, child_children_indices) = {
                let child_meta = &mut parsed_nodes[child_index as usize];
                (
                    child_meta.mesh_indices.clone(),
                    child_meta.og_matrix,
                    child_meta.local_transform.compose(),
                    child_meta.children.clone(),
                )
            };

            let child_node = Rc::new(RefCell::new(gpu_data::Node {
                parent: Some(Rc::downgrade(parent)),
                children: vec![],
                meshes,
                world_transform,
                local_transform,
                dirty: true,
            }));

            // Recursively construct child nodes
            let child_children = recur_children(&child_node, &child_children_indices, parsed_nodes);
            child_node.borrow_mut().children = child_children;

            child_node
        })
        .collect()
}

pub fn gltf_format_to_vk_format(format: Format) -> vk::Format {
    match format {
        Format::R8 => vk::Format::R8_UNORM,
        Format::R8G8 => vk::Format::R8G8_UNORM,
        Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
        Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
        Format::R16 => vk::Format::R16_UNORM,
        Format::R16G16 => vk::Format::R16G16_UNORM,
        Format::R16G16B16 => vk::Format::R16G16B16_UNORM,
        Format::R16G16B16A16 => vk::Format::R16G16B16A16_UNORM,
        Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
        Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
    }
}

// Used as a temp structure if needing to use mikktspace for generating tangents
pub struct Primitive<'a> {
    pub vertices: &'a mut [Vertex],
    pub indices: &'a [u32],
}

impl<'a> Geometry for Primitive<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        let index = self.indices[face * 3 + vert] as usize;
        self.vertices[index].position.to_array()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let index = self.indices[face * 3 + vert] as usize;
        self.vertices[index].normal.to_array()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        let index = self.indices[face * 3 + vert] as usize;
        [self.vertices[index].uv0_x, self.vertices[index].uv0_y]
    }

    fn set_tangent(
        &mut self,
        tangent: [f32; 3],
        bi_tangent: [f32; 3],
        f_mag_s: f32,
        f_mag_t: f32,
        bi_tangent_preserves_orientation: bool,
        face: usize,
        vert: usize,
    ) {
        let index = self.indices[face * 3 + vert] as usize;
        let sign = if bi_tangent_preserves_orientation {
            1.0
        } else {
            -1.0
        };
        self.vertices[index].tangent = glam::Vec4::new(tangent[0], tangent[1], tangent[2], sign);
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let index = self.indices[face * 3 + vert] as usize;
        self.vertices[index].tangent = glam::Vec4::from_array(tangent);
    }
}
