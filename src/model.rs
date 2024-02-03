use itertools::Itertools;
use three_d::{vec3, CpuMesh, Indices, InnerSpace, Instances, Mat4, Positions, Quat, Vec3};

#[derive(Debug)]
pub struct Model {
    pub name: String,
    pub poly: PolyModel,
}

/// A polygonal model where all faces are regular and all edges have unit length.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyModel {
    verts: VertVec<Vec3>,
    /// Each face of the model, listing vertices in clockwise order
    faces: FaceVec<Vec<VertIdx>>,
}

impl PolyModel {
    ////////////
    // SHAPES //
    ////////////

    pub fn tetrahedron() -> Self {
        Self::pyramid(3)
    }

    pub fn cube() -> Self {
        Self::prism(4)
    }

    pub fn octahedron() -> Self {
        let r = 2f32.sqrt() / 2.0;
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

    pub fn cuboctahedron() -> Self {
        let r = 2f32.sqrt() / 2.0;
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

    pub fn prism(n: usize) -> Self {
        assert!(n >= 3);
        let geom = PolygonGeom::new(n);
        // Vertices
        let mut verts = Vec::new();
        for i in 0..n {
            let (x, z) = geom.offset_point(i, 0.5);
            verts.push(vec3(x, -0.5, z));
            verts.push(vec3(x, 0.5, z));
        }
        // Faces
        let mut faces = vec![
            (0..n).map(|i| i * 2 + 1).collect_vec(),   // Top face
            (0..n).rev().map(|i| i * 2).collect_vec(), // Bottom face
        ];
        for i1 in 0..n {
            let i2 = (i1 + 1) % n;
            faces.push(vec![i1 * 2 + 1, i1 * 2, i2 * 2, i2 * 2 + 1]); // Side faces
        }
        // Construct
        Self::new(verts, faces)
    }

    pub fn antiprism(n: usize) -> Self {
        assert!(n >= 3);
        let geom = PolygonGeom::new(n);
        let half_height =
            f32::sqrt((f32::cos(geom.angle / 2.0) - f32::cos(geom.angle)) / 2.0) * geom.out_radius;
        // Vertices
        let mut verts = Vec::new();
        for i in 0..n {
            let (x, z) = geom.point(i);
            verts.push(vec3(x, -half_height, z));
            let (x, z) = geom.offset_point(i, 0.5);
            verts.push(vec3(x, half_height, z));
        }
        let bottom_vert = |i: usize| (i % n) * 2;
        let top_vert = |i: usize| (i % n) * 2 + 1;
        // Faces
        let mut faces = Vec::new();
        faces.push((0..n).rev().map(bottom_vert).collect_vec());
        faces.push((0..n).map(top_vert).collect_vec());
        for i in 0..n {
            faces.push(vec![top_vert(i), bottom_vert(i), bottom_vert(i + 1)]);
            faces.push(vec![top_vert(i), bottom_vert(i + 1), top_vert(i + 1)]);
        }
        // Construct
        Self::new(verts, faces)
    }

    pub fn cupola(n: usize) -> Self {
        assert!((3..=5).contains(&n));
        let top = PolygonGeom::new(n);
        let bottom = PolygonGeom::new(n * 2);
        let rad_diff = bottom.in_radius - top.in_radius;
        let height = f32::sqrt(1.0 - rad_diff * rad_diff);
        // Verts
        let mut verts = Vec::new();
        for i in 0..n {
            let (x, z) = top.offset_point(i, 0.5);
            verts.push(vec3(x, height, z));
        }
        for i in 0..n * 2 {
            let (x, z) = bottom.offset_point(i, 0.5);
            verts.push(vec3(x, 0.0, z));
        }
        let top_vert = |i: usize| i % n;
        let bottom_vert = |i: usize| n + i % (n * 2);
        // Faces
        let mut faces = Vec::new();
        faces.push((0..n).collect_vec());
        faces.push((0..n * 2).rev().map(bottom_vert).collect_vec());
        for i in 0..n {
            faces.push(vec![
                top_vert(i + 1),
                top_vert(i),
                bottom_vert(i * 2 + 1),
                bottom_vert(i * 2 + 2),
            ]);
            faces.push(vec![
                top_vert(i),
                bottom_vert(i * 2),
                bottom_vert(i * 2 + 1),
            ]);
        }
        // Construct
        Self::new(verts, faces)
    }

    pub fn pyramid(n: usize) -> Self {
        assert!((3..=5).contains(&n));
        let geom = PolygonGeom::new(n);
        let height = f32::sqrt(1.0 - geom.out_radius * geom.out_radius);
        // Verts
        let mut verts = vec![vec3(0.0, height, 0.0)];
        for i in 0..n {
            let (x, z) = geom.point(i);
            verts.push(vec3(x, 0.0, z));
        }
        // Faces
        let mut faces = Vec::new();
        faces.push((1..=n).rev().collect_vec()); // Bottom face
        for i in 0..n {
            faces.push(vec![0, i + 1, (i + 1) % n + 1]); // Triangle faces
        }
        // Construct
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

    pub fn edges(&self) -> Vec<(VertIdx, VertIdx)> {
        let mut edges = Vec::new();
        for f in &self.faces {
            for (&v1, &v2) in f.iter().circular_tuple_windows() {
                edges.push((v1.min(v2), v1.max(v2)));
            }
        }
        // Dedup and return edges
        edges.sort();
        edges.dedup();
        edges
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

    pub fn edge_instances(&self) -> Instances {
        Instances {
            transformations: self
                .edges()
                .into_iter()
                .map(|(v1, v2)| edge_transform(self.verts[v1], self.verts[v2]))
                .collect_vec(),
            ..Default::default()
        }
    }

    pub fn vertex_instances(&self) -> Instances {
        Instances {
            transformations: self
                .verts
                .iter()
                .cloned()
                .map(Mat4::from_translation)
                .collect_vec(),
            ..Default::default()
        }
    }
}

struct PolygonGeom {
    angle: f32,
    in_radius: f32,
    out_radius: f32,
}

impl PolygonGeom {
    fn new(n: usize) -> Self {
        let angle = std::f32::consts::PI * 2.0 / n as f32;
        let in_radius = 1.0 / (2.0 * f32::tan(angle / 2.0));
        let out_radius = 1.0 / (2.0 * f32::sin(angle / 2.0));
        Self {
            angle,
            in_radius,
            out_radius,
        }
    }

    fn point(&self, i: usize) -> (f32, f32) {
        self.offset_point(i, 0.0)
    }

    fn offset_point(&self, i: usize, offset: f32) -> (f32, f32) {
        let a = self.angle * (i as f32 + offset);
        let x = a.sin() * self.out_radius;
        let y = a.cos() * self.out_radius;
        (x, y)
    }
}

index_vec::define_index_type! { pub struct VertIdx = usize; }
index_vec::define_index_type! { pub struct FaceIdx = usize; }
pub type VertVec<T> = index_vec::IndexVec<VertIdx, T>;
pub type FaceVec<T> = index_vec::IndexVec<FaceIdx, T>;

fn edge_transform(p1: Vec3, p2: Vec3) -> Mat4 {
    Mat4::from_translation(p1)
        * Mat4::from(Quat::from_arc(
            vec3(1.0, 0.0, 0.0),
            (p2 - p1).normalize(),
            None,
        ))
        * Mat4::from_nonuniform_scale((p1 - p2).magnitude(), 1.0, 1.0)
}
