use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use three_d::{
    vec3, CpuMesh, Indices, InnerSpace, Instances, Mat4, Positions, Quat, SquareMatrix, Vec3, Vec4,
    Zero,
};

/// A polygonal model where all faces are regular and all edges have unit length.
#[derive(Debug, Clone, PartialEq)]
pub struct Polyhedron {
    verts: VertVec<Vec3>,
    /// Each face of the model, listing vertices in clockwise order
    faces: FaceVec<Option<Face>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Face {
    verts: Vec<VertIdx>,
}

impl Face {
    fn new(verts: Vec<VertIdx>) -> Self {
        Self { verts }
    }

    pub fn verts(&self) -> &[VertIdx] {
        &self.verts
    }

    pub fn order(&self) -> usize {
        self.verts.len()
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub bottom_vert: VertIdx,
    pub top_vert: VertIdx,
    pub right_face: FaceIdx,
    pub left_face: Option<FaceIdx>,
}

impl Edge {
    fn new(bottom_vert: VertIdx, top_vert: VertIdx, right_face: FaceIdx) -> Self {
        Self {
            bottom_vert,
            top_vert,
            right_face,
            left_face: None,
        }
    }

    fn add_left_face(&mut self, v1: VertIdx, v2: VertIdx, face_idx: FaceIdx) {
        assert_eq!((v1, v2), (self.top_vert, self.bottom_vert));
        self.left_face = Some(face_idx);
    }
}

////////////
// SHAPES //
////////////

#[derive(Debug, Clone)]
pub struct Pyramid {
    pub poly: Polyhedron,
    pub base_face: FaceIdx,
}

#[derive(Debug, Clone)]
pub struct PrismLike {
    pub poly: Polyhedron,
    pub bottom_face: FaceIdx,
    pub top_face: FaceIdx,
}

impl Polyhedron {
    pub fn tetrahedron() -> Self {
        Self::pyramid(3).poly
    }

    pub fn cube() -> Self {
        Self::prism(4).poly
    }

    pub fn octahedron() -> Self {
        let Pyramid {
            mut poly,
            base_face,
        } = Self::pyramid(4);
        poly.extend_pyramid(base_face);
        poly.make_centred();
        poly
    }

    pub fn icosahedron() -> Self {
        let PrismLike {
            mut poly,
            bottom_face,
            top_face,
        } = Self::antiprism(5);
        poly.extend_pyramid(bottom_face);
        poly.extend_pyramid(top_face);
        // Shape is already centred
        poly
    }

    pub fn cuboctahedron() -> Self {
        let PrismLike {
            mut poly,
            bottom_face,
            top_face: _,
        } = Self::cupola(3);
        poly.extend_cupola(bottom_face, false);
        poly.make_centred();
        poly
    }

    pub fn rhombicuboctahedron() -> Self {
        let PrismLike {
            mut poly,
            bottom_face,
            top_face,
        } = Self::prism(8);
        poly.extend_cupola(bottom_face, true);
        poly.extend_cupola(top_face, true);
        poly
    }

    pub fn prism(n: usize) -> PrismLike {
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
        let mut faces = FaceVec::new();
        let top_face = faces.push((0..n).map(|i| i * 2 + 1).collect_vec());
        let bottom_face = faces.push((0..n).rev().map(|i| i * 2).collect_vec());
        for i1 in 0..n {
            let i2 = (i1 + 1) % n;
            faces.push(vec![i1 * 2 + 1, i1 * 2, i2 * 2, i2 * 2 + 1]); // Side faces
        }
        // Construct
        PrismLike {
            poly: Self::new(verts, faces),
            bottom_face,
            top_face,
        }
    }

    pub fn antiprism(n: usize) -> PrismLike {
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
        let mut faces = FaceVec::new();
        let bottom_face = faces.push((0..n).rev().map(bottom_vert).collect_vec());
        let top_face = faces.push((0..n).map(top_vert).collect_vec());
        for i in 0..n {
            faces.push(vec![top_vert(i), bottom_vert(i), bottom_vert(i + 1)]);
            faces.push(vec![top_vert(i), bottom_vert(i + 1), top_vert(i + 1)]);
        }
        // Construct
        PrismLike {
            poly: Self::new(verts, faces),
            bottom_face,
            top_face,
        }
    }

    pub fn cupola(n: usize) -> PrismLike {
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
        let mut faces = FaceVec::new();
        let top_face = faces.push((0..n).collect_vec());
        let bottom_face = faces.push((0..n * 2).rev().map(bottom_vert).collect_vec());
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
        PrismLike {
            poly: Self::new(verts, faces),
            bottom_face,
            top_face,
        }
    }

    pub fn pyramid(n: usize) -> Pyramid {
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
        let mut faces = FaceVec::new();
        let base_face = faces.push((1..=n).rev().collect_vec()); // Bottom face
        for i in 0..n {
            faces.push(vec![0, i + 1, (i + 1) % n + 1]); // Triangle faces
        }
        // Construct
        Pyramid {
            poly: Self::new(verts, faces),
            base_face,
        }
    }

    fn new(verts: Vec<Vec3>, faces: FaceVec<Vec<usize>>) -> Self {
        let mut m = Self {
            verts: index_vec::IndexVec::from_vec(verts),
            faces: faces
                .into_iter()
                .map(|verts| Face {
                    verts: verts.into_iter().map(VertIdx::new).collect_vec(),
                })
                .map(Some)
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

#[derive(PartialEq, Eq)]
enum MergeDir {
    Extend,
    Excavate,
}

impl Polyhedron {
    /// 'Extend' this polyhedron by adding a copy of `other` onto the given `face`.
    /// The `other` polyhedron is attached by `its_face`.
    pub fn extend(
        &mut self,
        face: FaceIdx,
        other: &Self,
        its_face: FaceIdx,
        rotation: usize,
        faces_to_track: &[FaceIdx],
    ) -> Vec<FaceIdx> {
        self.merge(
            face,
            other,
            its_face,
            rotation,
            MergeDir::Extend,
            faces_to_track,
        )
    }

    pub fn extend_pyramid(&mut self, face: FaceIdx) {
        let n = self.face_order(face);
        let pyramid = Polyhedron::pyramid(n);
        self.extend(face, &pyramid.poly, pyramid.base_face, 0, &[]);
    }

    pub fn extend_prism(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Extend, 0, Self::prism)
    }

    pub fn extend_cupola(&mut self, face: FaceIdx, gyro: bool) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Extend, gyro as usize, |n| {
            assert!(n % 2 == 0);
            Self::cupola(n / 2)
        })
    }

    /// 'Excavate' this polyhedron by adding a copy of `other` onto the given `face`.
    /// The `other` polyhedron is attached by `its_face`.
    pub fn excavate(
        &mut self,
        face: FaceIdx,
        other: &Self,
        its_face: FaceIdx,
        rotation: usize,
        faces_to_track: &[FaceIdx],
    ) -> Vec<FaceIdx> {
        self.merge(
            face,
            other,
            its_face,
            rotation,
            MergeDir::Excavate,
            faces_to_track,
        )
    }

    pub fn excavate_prism(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Excavate, 0, Self::prism)
    }

    pub fn excavate_antiprism(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Excavate, 0, Self::antiprism)
    }

    pub fn excavate_cupola(&mut self, face: FaceIdx, gyro: bool) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Excavate, gyro as usize, |n| {
            assert!(n % 2 == 0);
            Self::cupola(n / 2)
        })
    }

    fn merge_prismlike(
        &mut self,
        face: FaceIdx,
        dir: MergeDir,
        rotation: usize,
        f: impl FnOnce(usize) -> PrismLike,
    ) -> FaceIdx {
        let PrismLike {
            poly,
            bottom_face,
            top_face,
        } = f(self.face_order(face));
        let tracked_faces = self.merge(face, &poly, bottom_face, rotation, dir, &[top_face]);
        assert_eq!(tracked_faces.len(), 1);
        tracked_faces[0]
    }

    fn merge(
        &mut self,
        face: FaceIdx,
        other: &Self,
        its_face: FaceIdx,
        rotation: usize,
        dir: MergeDir,
        faces_to_track: &[FaceIdx],
    ) -> Vec<FaceIdx> {
        assert_eq!(self.face_order(face), other.face_order(its_face));
        // Find the matrix transformation required to place `other` in the right location to
        // join `its_face` to `face` at the correct `rotation`
        let side = match dir {
            MergeDir::Extend => Side::In,
            MergeDir::Excavate => Side::Out,
        };
        let self_face_transform = self.face_transform(face, 0, Side::Out);
        let other_face_transform = other.face_transform(its_face, rotation, side);
        let transform = self_face_transform * other_face_transform.invert().unwrap();
        // Merge vertices into `self`, tracking which vertices they turn into
        let new_vert_indices: VertVec<VertIdx> = other
            .verts
            .iter()
            .map(|&v| self.add_vert(transform_point(v, transform)))
            .collect();
        // Add all the new faces (turning them inside out if we're excavating)
        let mut new_face_indices = FaceVec::new();
        for face in other.faces() {
            let mut new_verts = face
                .verts
                .iter()
                .map(|v_idx| new_vert_indices[*v_idx])
                .collect_vec();
            if dir == MergeDir::Excavate {
                new_verts.reverse();
            }
            let new_idx = self.faces.push(Some(Face::new(new_verts)));
            new_face_indices.push(new_idx);
        }
        // Cancel any new faces (which will include cancelling the two faces used to join these
        // polyhedra)
        self.cancel_faces();
        // Return the new indices of the faces we were asked to track
        faces_to_track
            .iter()
            .map(|f| new_face_indices[*f])
            .collect_vec()
    }

    /// Remove any pairs of identical but opposite faces
    fn cancel_faces(&mut self) {
        let normalized_faces: HashSet<Vec<VertIdx>> = self
            .faces()
            .map(|face| normalize_face(&face.verts))
            .collect();
        for f in &mut self.faces {
            if let Some(Face { verts }) = &*f {
                let mut verts = verts.clone();
                verts.reverse();
                let norm_verts = normalize_face(&verts);
                let is_duplicate = normalized_faces.contains(&norm_verts);
                if is_duplicate {
                    *f = None; // Delete the face by replacing it with `None` (thus preserving indices)
                }
            }
        }
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

        for face in self.faces() {
            // Add all vertices from this face.  We have to duplicate the vertices so that each
            // face gets flat shading
            let first_vert_idx = verts.len() as u32;
            verts.extend(face.verts.iter().map(|vert_idx| self.verts[*vert_idx]));
            // Add the vert indices for this face
            for i in 2..face.verts.len() as u32 {
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
                .map(|edge| edge_transform(self.verts[edge.bottom_vert], self.verts[edge.top_vert]))
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

    pub fn verts(&self) -> &[Vec3] {
        self.verts.as_raw_slice()
    }

    pub fn edges(&self) -> Vec<Edge> {
        let mut edges = HashMap::<(VertIdx, VertIdx), Edge>::new();
        for (face_idx, face) in self.faces_enumerated() {
            for (&v1, &v2) in face.verts.iter().circular_tuple_windows() {
                let key = (v1.min(v2), v1.max(v2));
                if let Some(edge) = edges.get_mut(&key) {
                    edge.add_left_face(v1, v2, face_idx);
                } else {
                    edges.insert(key, Edge::new(v1, v2, face_idx));
                }
            }
        }
        // Dedup and return edges
        edges.into_values().collect_vec()
    }

    /// Gets an [`Iterator`] over the [indices](FaceIdx) of every face in `self` which has `n`
    /// sides
    pub fn ngons(&self, n: usize) -> impl DoubleEndedIterator<Item = FaceIdx> + '_ {
        self.faces
            .iter()
            .positions(move |f| f.as_ref().map(|face| face.verts.len()) == Some(n))
            .map(FaceIdx::new)
    }

    /// Gets the lowest-indexed face in `self` which has `n` sides.
    pub fn get_ngon(&self, n: usize) -> FaceIdx {
        self.ngons(n).next().unwrap()
    }

    pub fn is_face(&self, idx: FaceIdx) -> bool {
        self.faces[idx].is_some()
    }

    pub fn faces(&self) -> impl DoubleEndedIterator<Item = &Face> + '_ {
        self.faces.iter().flatten()
    }

    pub fn faces_enumerated(&self) -> impl DoubleEndedIterator<Item = (FaceIdx, &Face)> + '_ {
        self.faces
            .iter_enumerated()
            .filter_map(|(idx, maybe_face)| maybe_face.as_ref().map(|face| (idx, face)))
    }

    pub fn get_face(&self, face: FaceIdx) -> &Face {
        self.faces[face].as_ref().unwrap()
    }

    pub fn face_order(&self, face: FaceIdx) -> usize {
        self.get_face(face).order()
    }

    /// Returns a matrix which translates and rotates such that:
    /// - The origin is now at `self.face_vert(face, rotation)`;
    /// - The y-axis now points along the `rotation`-th edge (i.e. from `verts[rotation]` towards
    ///   `verts[rotation + 1]`);
    /// - The z-axis now points directly out of the face along its normal.
    /// - The x-axis now points towards the centre of the face, perpendicular to the y-axis.
    pub fn face_transform(&self, face: FaceIdx, rotation: usize, side: Side) -> Mat4 {
        let verts = &self.get_face(face).verts;
        let translation = Mat4::from_translation(self.get_vert_pos(verts, rotation));
        let rotation = self.face_rotation(verts, rotation, side);
        translation * rotation
    }

    /// Returns a rotation matrix which rotates the unit (x, y, z) axis onto a face, such that:
    /// - The y-axis now points along the `rotation`-th edge (i.e. from `verts[rotation]` towards
    ///   `verts[rotation + 1]`);
    /// - The z-axis now points directly out of the face along its normal.
    /// - The x-axis now points towards the centre of the face, perpendicular to the y-axis.
    ///
    /// The `rotation` will be wrapped to fit within the vertices of the face
    pub fn face_rotation(&self, verts: &[VertIdx], rotation: usize, side: Side) -> Mat4 {
        let vert_offset = match side {
            Side::In => verts.len() - 1,
            Side::Out => 1,
        };
        let v0 = self.get_vert_pos(verts, rotation);
        let v1 = self.get_vert_pos(verts, rotation + vert_offset);
        let new_y = (v1 - v0).normalize();
        let mut new_z = self.normal_from_verts(verts);
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
        self.normal_from_verts(&self.get_face(face).verts)
    }

    pub fn normal_from_verts(&self, verts: &[VertIdx]) -> Vec3 {
        assert!(verts.len() >= 3);
        let v0 = self.get_vert_pos(verts, 0);
        let v1 = self.get_vert_pos(verts, 1);
        let v2 = self.get_vert_pos(verts, 2);
        let d1 = v1 - v0;
        let d2 = v2 - v0;
        d1.cross(d2).normalize()
    }

    fn get_vert_pos(&self, face: &[VertIdx], vert: usize) -> Vec3 {
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
