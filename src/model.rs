use itertools::Itertools;
use three_d::{vec3, CpuMesh, Indices, Positions, Vec3};

/// A polygonal model where all faces are regular and all edges have unit length.
#[derive(Debug, Clone)]
pub struct PolyModel {
    verts: VertVec<Vec3>,
    /// Each face of the model, listing vertices in clockwise order
    faces: FaceVec<Vec<VertIdx>>,
}

impl PolyModel {
    #[allow(dead_code)]
    pub fn cube() -> Self {
        let verts = vec![
            vec3(-0.5, -0.5, -0.5f32),
            vec3(-0.5, -0.5, 0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(0.5, -0.5, -0.5f32),
            vec3(0.5, -0.5, 0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, 0.5, 0.5),
        ];
        let faces = vec![
            vec![0, 2, 6, 4], // Front
            vec![1, 5, 7, 3], // Back
            vec![0, 1, 3, 2], // Left
            vec![4, 6, 7, 5], // Right
            vec![2, 3, 7, 6], // Top
            vec![0, 4, 5, 1], // Bottom
        ];
        Self::new(verts, faces)
    }

    #[allow(dead_code)]
    pub fn octahedron() -> Self {
        let r = 2f32.sqrt();
        let verts = vec![
            Vec3::unit_x() * -r,
            Vec3::unit_x() * r,
            Vec3::unit_y() * -r,
            Vec3::unit_y() * r,
            Vec3::unit_z() * -r,
            Vec3::unit_z() * r,
        ];
        let faces = vec![
            vec![3, 0, 5],
            vec![3, 5, 1],
            vec![3, 1, 4],
            vec![3, 4, 0],
            vec![2, 5, 0],
            vec![2, 1, 5],
            vec![2, 4, 1],
            vec![2, 0, 4],
        ];
        Self::new(verts, faces)
    }

    #[allow(dead_code)]
    pub fn cuboctahedron() -> Self {
        let r = 2f32.sqrt();
        let verts = vec![
            // XY plane
            vec3(-r, -r, 0.0),
            vec3(-r, r, 0.0),
            vec3(r, -r, 0.0),
            vec3(r, r, 0.0),
            // XZ plane
            vec3(-r, 0.0, -r),
            vec3(-r, 0.0, r),
            vec3(r, 0.0, -r),
            vec3(r, 0.0, r),
            // YZ plane
            vec3(0.0, -r, -r),
            vec3(0.0, -r, r),
            vec3(0.0, r, -r),
            vec3(0.0, r, r),
        ];
        let faces = vec![
            // Squares
            vec![4, 10, 6, 8],  // Front
            vec![7, 11, 5, 9],  // Back
            vec![6, 3, 7, 2],   // Right
            vec![1, 4, 0, 5],   // Left
            vec![1, 11, 3, 10], // Top
            vec![0, 8, 2, 9],   // Bottom
            // Triangles
            vec![1, 10, 4],
            vec![10, 3, 6],
            vec![3, 11, 7],
            vec![11, 1, 5],
            vec![4, 8, 0],
            vec![8, 6, 2],
            vec![2, 7, 9],
            vec![9, 5, 0],
        ];

        Self::new(verts, faces)
    }

    fn new(verts: Vec<Vec3>, faces: Vec<Vec<usize>>) -> Self {
        Self {
            verts: index_vec::IndexVec::from_vec(verts),
            faces: faces
                .into_iter()
                .map(|verts| verts.into_iter().map(VertIdx::new).collect_vec())
                .collect(),
        }
    }

    ///////////////
    // RENDERING //
    ///////////////

    pub fn face_mesh(&self) -> CpuMesh {
        let mut verts = Vec::new();
        let mut tri_indices = Vec::new();

        for face_verts in &self.faces {
            // Add all vertices from this face.  We have to duplicate the vertices so that each
            // face gets flat shading
            let first_vert_idx = verts.len() as u32;
            verts.extend(face_verts.iter().map(|vert_idx| self.verts[*vert_idx]));
            // Add the vert indices for this face
            for i in 2..face_verts.len() as u32 {
                tri_indices.extend_from_slice(&[
                    first_vert_idx,
                    first_vert_idx + i - 1,
                    first_vert_idx + i,
                ]);
            }
        }

        let mut mesh = CpuMesh {
            positions: Positions::F32(verts),
            indices: Indices::U32(tri_indices),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh
    }
}

index_vec::define_index_type! { pub struct VertIdx = usize; }
index_vec::define_index_type! { pub struct FaceIdx = usize; }
pub type VertVec<T> = index_vec::IndexVec<VertIdx, T>;
pub type FaceVec<T> = index_vec::IndexVec<FaceIdx, T>;
