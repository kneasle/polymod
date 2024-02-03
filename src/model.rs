use std::collections::HashSet;

use itertools::Itertools;
use three_d::{
    vec3, CpuMesh, Indices, InnerSpace, Instances, Mat4, Positions, Quat, SquareMatrix, Vec3, Vec4,
    Zero,
};

#[derive(Debug)]
pub struct Model {
    pub name: String,
    pub poly: Polyhedron,
}

impl Model {
    pub fn new(name: &str, poly: Polyhedron) -> Self {
        Self {
            name: name.to_owned(),
            poly,
        }
    }
}

/// A polygonal model where all faces are regular and all edges have unit length.
#[derive(Debug, Clone, PartialEq)]
pub struct Polyhedron {
    verts: VertVec<Vec3>,
    /// Each face of the model, listing vertices in clockwise order
    faces: FaceVec<Vec<VertIdx>>,
}

////////////
// SHAPES //
////////////

impl Polyhedron {
    pub fn tetrahedron() -> Self {
        Self::pyramid(3)
    }

    pub fn cube() -> Self {
        Self::prism(4)
    }

    pub fn octahedron() -> Self {
        let mut base = Self::pyramid(4);
        base.extend_pyramid(base.get_ngon(4));
        base.make_centred();
        base
    }

    pub fn icosahedron() -> Self {
        let mut shape = Self::antiprism(5);
        shape.extend_pyramid(shape.get_ngon(5));
        shape.extend_pyramid(shape.get_ngon(5));
        // Shape is already centred
        shape
    }

    pub fn cuboctahedron() -> Self {
        let mut shape = Self::cupola(3);
        shape.extend_cupola(shape.get_ngon(6), false);
        shape.make_centred();
        shape
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
        let mut m = Self {
            verts: index_vec::IndexVec::from_vec(verts),
            faces: faces
                .into_iter()
                .map(|verts| verts.into_iter().map(VertIdx::new).collect_vec())
                .collect(),
        };
        m.make_centred();
        m
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

///////////////
// MODELLING //
///////////////

impl Polyhedron {
    pub fn extend_pyramid(&mut self, face: FaceIdx) {
        let n = self.faces[face].len();
        self.extend(face, &Polyhedron::pyramid(n), FaceIdx::new(0), 0);
    }

    pub fn extend_cupola(&mut self, face: FaceIdx, gyro: bool) {
        let n = self.faces[face].len();
        assert!(n % 2 == 0);
        self.extend(face, &Self::cupola(n / 2), FaceIdx::new(1), gyro as usize);
    }

    /// 'Extend' this polyhedron by adding a copy of `other` onto the given `face`.
    /// The `other` polyhedron is attached by `its_face`.
    pub fn extend(&mut self, face: FaceIdx, other: &Self, its_face: FaceIdx, rotation: usize) {
        self.merge(face, other, its_face, rotation, true);
    }

    /// 'Excavate' this polyhedron by adding a copy of `other` onto the given `face`.
    /// The `other` polyhedron is attached by `its_face`.
    pub fn excavate(&mut self, face: FaceIdx, other: &Self, its_face: FaceIdx, rotation: usize) {
        self.merge(face, other, its_face, rotation, false);
    }

    fn merge(
        &mut self,
        face: FaceIdx,
        other: &Self,
        its_face: FaceIdx,
        rotation: usize,
        is_extrude: bool,
    ) {
        assert_eq!(self.faces[face].len(), other.faces[its_face].len());
        // Find the matrix transformation required to place `other` in the right location to
        // join `its_face` to `face` at the correct `rotation`
        let self_face_transform = self.face_transform(face, 0, Side::Out);
        let other_face_transform = other.face_transform(
            its_face,
            rotation,
            if is_extrude { Side::In } else { Side::Out },
        );
        let transform = self_face_transform * other_face_transform.invert().unwrap();
        // Merge vertices into `self`
        let new_vert_indices: VertVec<VertIdx> = other
            .verts
            .iter()
            .map(|&v| self.add_vert(transform_point(v, transform)))
            .collect();
        // Add all the new faces (turning them inside out because we're excavating)
        for face_verts in &other.faces {
            let mut new_verts = face_verts
                .iter()
                .map(|v_idx| new_vert_indices[*v_idx])
                .collect_vec();
            if !is_extrude {
                new_verts.reverse();
            }
            self.faces.push(new_verts);
        }
        // Cancel any new faces (which will include cancelling the two faces used to join these
        // polyhedra)
        self.cancel_faces();
    }

    /// Remove any pairs of identical but opposite faces
    fn cancel_faces(&mut self) {
        let normalized_faces: HashSet<Vec<VertIdx>> = self
            .faces
            .iter()
            .map(|verts| normalize_face(verts))
            .collect();
        self.faces.retain(|verts| {
            let mut verts = verts.clone();
            verts.reverse();
            let norm_verts = normalize_face(&verts);
            let is_duplicate = normalized_faces.contains(&norm_verts);
            !is_duplicate // Keep faces which aren't duplicates
        });
    }

    pub fn translate(&mut self, d: Vec3) {
        self.transform(Mat4::from_translation(d));
    }

    pub fn transform(&mut self, matrix: Mat4) {
        for v in &mut self.verts {
            *v = transform_point(*v, matrix);
        }
    }

    pub fn make_centred(&mut self) {
        self.translate(-self.centroid());
    }
}

fn normalize_face(verts: &[VertIdx]) -> Vec<VertIdx> {
    let min_vert = verts.iter().position_min().unwrap();
    let mut verts = verts.to_vec();
    verts.rotate_left(min_vert);
    verts
}

///////////////
// RENDERING //
///////////////

impl Polyhedron {
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

fn edge_transform(p1: Vec3, p2: Vec3) -> Mat4 {
    Mat4::from_translation(p1)
        * Mat4::from(Quat::from_arc(
            vec3(1.0, 0.0, 0.0),
            (p2 - p1).normalize(),
            None,
        ))
        * Mat4::from_nonuniform_scale((p1 - p2).magnitude(), 1.0, 1.0)
}

///////////
// UTILS //
///////////

const VERTEX_MERGE_DIST: f32 = 0.00001;
const VERTEX_MERGE_DIST_SQUARED: f32 = VERTEX_MERGE_DIST * VERTEX_MERGE_DIST;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    In,
    Out,
}

impl Polyhedron {
    /// Create a vertex at the given coords `p`, returning its index.  If there's already a vertex
    /// at `p`, then its index is returned.
    pub fn add_vert(&mut self, p: Vec3) -> VertIdx {
        // Look for existing vertices to dedup with
        for (idx, v) in self.verts.iter_enumerated() {
            if (p - *v).magnitude2() < VERTEX_MERGE_DIST_SQUARED {
                return idx;
            }
        }
        // If vertex isn't already present, add a new one
        self.verts.push(p)
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

    /// Gets an [`Iterator`] over the [indices](FaceIdx) of every face in `self` which has `n`
    /// sides
    pub fn ngons(&self, n: usize) -> impl DoubleEndedIterator<Item = FaceIdx> + '_ {
        self.faces
            .iter()
            .positions(move |vs| vs.len() == n)
            .map(FaceIdx::new)
    }

    /// Gets the highest-indexed face in `self` which has `n` sides.
    pub fn get_ngon(&self, n: usize) -> FaceIdx {
        self.ngons(n).next_back().unwrap()
    }

    /// Returns a matrix which translates and rotates such that:
    /// - The origin is now at `self.face_vert(face, rotation)`;
    /// - The y-axis now points along the `rotation`-th edge (i.e. from `verts[rotation]` towards
    ///   `verts[rotation + 1]`);
    /// - The z-axis now points directly out of the face along its normal.
    /// - The x-axis now points towards the centre of the face, perpendicular to the y-axis.
    pub fn face_transform(&self, face: FaceIdx, rotation: usize, side: Side) -> Mat4 {
        let translation = Mat4::from_translation(self.face_vert(face, rotation));
        let rotation = self.face_rotation(face, rotation, side);
        translation * rotation
    }

    /// Returns a rotation matrix which rotates the unit (x, y, z) axis onto a face, such that:
    /// - The y-axis now points along the `rotation`-th edge (i.e. from `verts[rotation]` towards
    ///   `verts[rotation + 1]`);
    /// - The z-axis now points directly out of the face along its normal.
    /// - The x-axis now points towards the centre of the face, perpendicular to the y-axis.
    ///
    /// The `rotation` will be wrapped to fit within the vertices of the face
    pub fn face_rotation(&self, face: FaceIdx, rotation: usize, side: Side) -> Mat4 {
        let vert_offset = match side {
            Side::In => self.faces[face].len() - 1,
            Side::Out => 1,
        };
        let v0 = self.face_vert(face, rotation);
        let v1 = self.face_vert(face, rotation + vert_offset);
        let new_y = (v1 - v0).normalize();
        let mut new_z = self.face_normal(face);
        if side == Side::In {
            new_z = -new_z;
        }
        let new_x = new_z.cross(new_y);
        // Make a matrix to transform into this new coord system
        Mat4::from_cols(
            new_x.extend(0.0),
            new_y.extend(0.0),
            new_z.extend(0.0),
            Vec4::unit_w(),
        )
    }

    pub fn face_normal(&self, face: FaceIdx) -> Vec3 {
        assert!(self.faces[face].len() >= 3);
        let v0 = self.face_vert(face, 0);
        let v1 = self.face_vert(face, 1);
        let v2 = self.face_vert(face, 2);
        let d1 = v1 - v0;
        let d2 = v2 - v0;
        d1.cross(d2).normalize()
    }

    pub fn face_vert(&self, face: FaceIdx, vert: usize) -> Vec3 {
        let face = &self.faces[face];
        let vert_idx = face[vert % face.len()];
        self.verts[vert_idx]
    }

    pub fn centroid(&self) -> Vec3 {
        let mut total = Vec3::zero();
        for v in &self.verts {
            total += *v;
        }
        total / self.verts.len() as f32
    }
}

fn transform_point(v: Vec3, matrix: Mat4) -> Vec3 {
    let v4 = v.extend(1.0);
    let trans_v4 = matrix * v4;
    trans_v4.truncate()
}

index_vec::define_index_type! { pub struct VertIdx = usize; }
index_vec::define_index_type! { pub struct FaceIdx = usize; }
pub type VertVec<T> = index_vec::IndexVec<VertIdx, T>;
pub type FaceVec<T> = index_vec::IndexVec<FaceIdx, T>;
