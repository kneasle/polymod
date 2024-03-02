use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
};

use itertools::Itertools;
use three_d::{
    vec3, Angle, CpuMesh, Deg, Degrees, Indices, InnerSpace, InstancedMesh, Instances, Mat4, Mesh,
    MetricSpace, Positions, Quat, Rad, Radians, SquareMatrix, Srgba, Vec3, Vec4, Zero,
};

use crate::utils::{angle_in_spherical_triangle, lerp3, normalize_perpendicular_to, Side};

/// A polygonal model where all faces are regular and all edges have unit length.
#[derive(Debug, Clone, PartialEq)]
pub struct Polyhedron {
    verts: VertVec<Vec3>,
    /// Each face of the model, listing vertices in clockwise order
    faces: FaceVec<Option<Face>>,
    edges: HashMap<(VertIdx, VertIdx), EdgeData>,
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

    pub fn is_regular(&self, polyhedron: &Polyhedron) -> bool {
        let expected_angle = Rad::full_turn() / self.order() as f32;
        for (v1, v2, v3) in self.vert_positions(polyhedron).circular_tuple_windows() {
            let angle = (v2 - v1).angle(v3 - v2);
            if f32::abs(angle.0 - expected_angle.0) > 0.0001 {
                return false; // Incorrect angle => irregular face
            }
        }
        true
    }

    pub fn is_flat(&self, polyhedron: &Polyhedron) -> bool {
        let normal = self.normal(polyhedron);
        for (v1, v2) in self.vert_positions(polyhedron).circular_tuple_windows() {
            if (v2 - v1).dot(normal) > 0.00001 {
                return false; // Edge is not perpendicular to normal => aplanar face
            }
        }
        true
    }

    pub fn normal(&self, polyhedron: &Polyhedron) -> Vec3 {
        polyhedron.normal_from_verts(&self.verts)
    }

    pub fn vert_positions<'a>(
        &'a self,
        polyhedron: &'a Polyhedron,
    ) -> impl ExactSizeIterator<Item = Vec3> + 'a + Clone {
        self.verts.iter().map(|idx| polyhedron.verts[*idx])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeData {
    color: Option<Srgba>,
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

/// Platonic solids
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

    pub fn dodecahedron() -> Self {
        let mut poly = Self::icosahedron().dual();
        poly.normalize_edge_length();
        poly
    }
}

/// Archimedean
impl Polyhedron {
    /* Tetrahedral */

    pub fn truncated_tetrahedron() -> Self {
        Self::tetrahedron().truncate_platonic(TruncationType::Standard)
    }

    /* Cubic/octahedral */

    pub fn truncated_cube() -> Self {
        Self::cube().truncate_platonic(TruncationType::Standard)
    }

    pub fn truncated_octahedron() -> Self {
        Self::octahedron().truncate_platonic(TruncationType::Standard)
    }

    pub fn cuboctahedron() -> Self {
        Self::cube().truncate_platonic(TruncationType::Alternation)
    }

    pub fn snub_cube() -> Self {
        let t = 1.8392868; // tribonacci constant
        let alpha = f32::sqrt(2.0 + 4.0 * t - 2.0 * t * t);

        let face_rotation_radians = f32::atan(t);
        let rotation_as_edges = face_rotation_radians / (2.0 * PI) * 4.0;
        let new_radius = t / alpha;
        Self::cube().snub_platonic(new_radius, rotation_as_edges)
    }

    pub fn rhombicuboctahedron() -> Self {
        Self::cube().rhombicosi_platonic(Greatness::Lesser)
    }

    pub fn great_rhombicuboctahedron() -> Self {
        Self::cube().rhombicosi_platonic(Greatness::Great)
    }

    /* Dodecaheral/icosahedral */

    pub fn truncated_dodecahedron() -> Self {
        Self::dodecahedron().truncate_platonic(TruncationType::Standard)
    }

    pub fn truncated_icosahedron() -> Self {
        Self::icosahedron().truncate_platonic(TruncationType::Standard)
    }

    pub fn icosidodecahedron() -> Self {
        Self::dodecahedron().truncate_platonic(TruncationType::Alternation)
    }

    pub fn snub_dodecahedron() -> Self {
        let new_radius = 1.9809159;
        let rotation_degrees = 13.106403;
        let rotation_sides = rotation_degrees / 360.0 * 5.0;
        Self::dodecahedron().snub_platonic(new_radius, 0.5 + rotation_sides)
    }

    pub fn rhombicosidodecahedron() -> Self {
        Self::dodecahedron().rhombicosi_platonic(Greatness::Lesser)
    }

    pub fn great_rhombicosidodecahedron() -> Self {
        Self::dodecahedron().rhombicosi_platonic(Greatness::Great)
    }

    /* Modelling operations */

    /// If `self` is a Platonic solid, return the truncated or alternated version of `self`
    fn truncate_platonic(&self, trunc_type: TruncationType) -> Self {
        // Calculate how far down each edge the new vertices need to be created
        let lerp_factor = match trunc_type {
            TruncationType::Standard => {
                let face_order = self.faces().next().unwrap().order();
                let base_geom = PolygonGeom::new(face_order);
                let scaled_geom = PolygonGeom::new(face_order * 2);
                let scale_factor = scaled_geom.in_radius / base_geom.in_radius;
                (1.0 - 1.0 / scale_factor) / 2.0
            }
            TruncationType::Alternation => 0.5,
        };
        // `new_vert_on_edge[(a, b)]` is the first vertex on the edge going from `a` to `b`
        // (therefore, `new_vert_on_edge[(a, b)]` won't be the same as `new_vert_on_edge[(b, a)]`)
        let mut new_verts = VertVec::<Vec3>::new();
        let mut new_vert_on_edge = HashMap::<(VertIdx, VertIdx), VertIdx>::new();
        let mut add_vert = |v1: VertIdx, v2: VertIdx| {
            let pos = lerp3(self.verts[v1], self.verts[v2], lerp_factor);
            let new_idx = new_verts.push(pos);
            new_vert_on_edge.insert((v1, v2), new_idx);
        };
        for e in self.edges() {
            let (v1, v2) = (e.top_vert, e.bottom_vert);
            add_vert(v1, v2);
            if trunc_type == TruncationType::Standard {
                add_vert(v2, v1);
            }
        }
        if trunc_type == TruncationType::Alternation {
            // For alternations, the same vertex is accessible from each end of the face
            let existing_edges = new_vert_on_edge.iter().map(|(k, v)| (*k, *v)).collect_vec();
            for ((v1, v2), new_vert) in existing_edges {
                new_vert_on_edge.insert((v2, v1), new_vert);
            }
        }

        // Add new faces for the main faces
        let mut new_faces = FaceVec::<Option<Face>>::new();
        for f in self.faces() {
            let mut verts = Vec::new();
            for (v1, v2) in f.verts.iter().copied().circular_tuple_windows() {
                verts.push(new_vert_on_edge[&(v1, v2)]);
                if trunc_type == TruncationType::Standard {
                    verts.push(new_vert_on_edge[&(v2, v1)]);
                }
            }
            new_faces.push(Some(Face { verts }));
        }
        // Add faces for the truncated vertices
        for vert_data in self.vert_datas() {
            let new_face_verts = vert_data
                .clockwise_loop
                .into_iter()
                .map(|(opposite_vert, _face)| new_vert_on_edge[&(vert_data.idx, opposite_vert)])
                .collect_vec();
            new_faces.push(Some(Face {
                verts: new_face_verts,
            }));
        }

        let mut model = Self {
            verts: new_verts,
            faces: new_faces,
            edges: HashMap::new(),
        };
        model.normalize_edge_length();
        model
    }

    /// If `self` is a platonic solid, return the rhombicosi- or great-rhombicosi- version of `self`
    fn rhombicosi_platonic(&self, greatness: Greatness) -> Self {
        let (first_face_idx, first_face) = self.faces_enumerated().next().unwrap();
        let face_order = first_face.order();
        let new_face_order = match greatness {
            Greatness::Lesser => face_order,
            Greatness::Great => face_order * 2,
        };
        let dihedral_angle = self.edges()[0].dihedral_angle().unwrap();
        let angle_between_adjacent_face_normals = Radians::from(Deg(180.0)) - dihedral_angle;
        // Determine model's scaling factor
        let new_face_geometry = PolygonGeom::new(new_face_order);
        let alpha = angle_between_adjacent_face_normals / 2.0;
        let new_face_radius = 0.5 / alpha.sin() + new_face_geometry.in_radius / alpha.tan();
        let existing_face_radius = self.face_centroid(first_face_idx).magnitude();
        let scaling_factor = new_face_radius / existing_face_radius;

        // Create expanded faces of `self`, recording which vertices fall onto which edges
        let mut new_verts = VertVec::new();
        let mut new_faces = FaceVec::new();
        let mut left_vert_down_edge = HashMap::<(VertIdx, VertIdx), VertIdx>::new();
        let mut right_vert_down_edge = HashMap::<(VertIdx, VertIdx), VertIdx>::new();
        for (face_idx, face) in self.faces_enumerated() {
            // Determine where to place the new face
            let centroid = self.face_centroid(face_idx);
            let edge_midpoint = lerp3(self.verts[face.verts[0]], self.verts[face.verts[1]], 0.5);
            let new_face_centroid = centroid * scaling_factor;
            let new_y_axis = (edge_midpoint - centroid).normalize();
            let new_x_axis = centroid.normalize().cross(new_y_axis);
            // Create vertices for the new face
            let mut face_verts = Vec::new();
            for i in 0..new_face_order {
                let (x, y) = new_face_geometry.offset_point(i, 0.5);
                let pos = new_face_centroid + x * new_x_axis + y * new_y_axis;
                let new_vert_idx = new_verts.push(pos);
                face_verts.push(new_vert_idx);
                // Record this vertex's presence on the new edges
                let old_vert_idx = match greatness {
                    Greatness::Lesser => i,
                    Greatness::Great => i / 2,
                };
                #[allow(clippy::identity_op)]
                let old_v0 = face.verts[(old_vert_idx + 0) % face_order];
                let old_v1 = face.verts[(old_vert_idx + 1) % face_order];
                let old_v2 = face.verts[(old_vert_idx + 2) % face_order];
                if greatness == Greatness::Lesser || i % 2 == 0 {
                    left_vert_down_edge.insert((old_v1, old_v0), new_vert_idx);
                }
                if greatness == Greatness::Lesser || i % 2 == 1 {
                    right_vert_down_edge.insert((old_v1, old_v2), new_vert_idx);
                }
            }
            // Create new face
            new_faces.push(Some(Face { verts: face_verts }));
        }

        // Add square faces for each current edge
        for edge in self.edges() {
            //         + (edge.top_vert)
            //         |
            // tl +----|----+ tr
            //    |    |    |
            //    |    |    |
            //    |    |    |
            // bl +----|----+ br
            //         |
            //         + (edge.bottom_vert)
            let tl = right_vert_down_edge[&(edge.top_vert, edge.bottom_vert)];
            let tr = left_vert_down_edge[&(edge.top_vert, edge.bottom_vert)];
            let bl = left_vert_down_edge[&(edge.bottom_vert, edge.top_vert)];
            let br = right_vert_down_edge[&(edge.bottom_vert, edge.top_vert)];
            new_faces.push(Some(Face {
                verts: vec![tl, tr, br, bl],
            }));
        }
        // Add faces for each current vertex
        for vert_data in self.vert_datas() {
            let mut new_verts = Vec::new();
            for (other_vert, _face) in vert_data.clockwise_loop {
                let edge = &(vert_data.idx, other_vert);
                new_verts.push(left_vert_down_edge[edge]);
                if greatness == Greatness::Great {
                    new_verts.push(right_vert_down_edge[edge]);
                }
            }
            new_faces.push(Some(Face { verts: new_verts }));
        }

        Self {
            verts: new_verts,
            faces: new_faces,
            edges: HashMap::new(),
        }
    }

    /// If `self` is a platonic solid, return the rhombicosi- or great-rhombicosi- version of `self`
    fn snub_platonic(&self, new_inradius: f32, rotation: f32) -> Self {
        let first_face = self.faces().next().unwrap();
        let face_order = first_face.order();
        // Determine model's scaling factor
        let face_geometry = PolygonGeom::new(face_order);

        // Create expanded faces of `self`, recording which vertices fall onto which edges
        let mut new_verts = VertVec::new();
        let mut new_faces = FaceVec::new();
        let mut left_vert_down_edge = HashMap::<(VertIdx, VertIdx), VertIdx>::new();
        let mut right_vert_down_edge = HashMap::<(VertIdx, VertIdx), VertIdx>::new();
        for (face_idx, face) in self.faces_enumerated() {
            // Determine where to place the new face
            let centroid = self.face_centroid(face_idx);
            let edge_midpoint = lerp3(self.verts[face.verts[0]], self.verts[face.verts[1]], 0.5);
            let new_face_centroid = centroid.normalize() * new_inradius;
            let new_y_axis = (edge_midpoint - centroid).normalize();
            let new_x_axis = centroid.normalize().cross(new_y_axis);
            // Create vertices for the new face
            let mut face_verts = Vec::new();
            for i in 0..face_order {
                let (x, y) = face_geometry.offset_point(i, rotation);
                let pos = new_face_centroid + x * new_x_axis + y * new_y_axis;
                let new_vert_idx = new_verts.push(pos);
                face_verts.push(new_vert_idx);
                // Record this vertex's presence on the new edges
                #[allow(clippy::identity_op)]
                let old_v0 = face.verts[(i + 0) % face_order];
                let old_v1 = face.verts[(i + 1) % face_order];
                let old_v2 = face.verts[(i + 2) % face_order];
                left_vert_down_edge.insert((old_v1, old_v0), new_vert_idx);
                right_vert_down_edge.insert((old_v1, old_v2), new_vert_idx);
            }
            // Create new face
            new_faces.push(Some(Face { verts: face_verts }));
        }

        // Add square faces for each current edge
        for edge in self.edges() {
            //         + (edge.top_vert)
            //         |
            // tl +----|----+ tr
            //    | \_ |    |
            //    |   -|-_  |
            //    |    |  \ |
            // bl +----|----+ br
            //         |
            //         + (edge.bottom_vert)
            let tl = right_vert_down_edge[&(edge.top_vert, edge.bottom_vert)];
            let tr = left_vert_down_edge[&(edge.top_vert, edge.bottom_vert)];
            let bl = left_vert_down_edge[&(edge.bottom_vert, edge.top_vert)];
            let br = right_vert_down_edge[&(edge.bottom_vert, edge.top_vert)];
            new_faces.push(Some(Face {
                verts: vec![tl, br, bl],
            }));
            new_faces.push(Some(Face {
                verts: vec![tl, tr, br],
            }));
        }
        // Add faces for each current vertex
        for vert_data in self.vert_datas() {
            let mut new_verts = Vec::new();
            for (other_vert, _face) in vert_data.clockwise_loop {
                let edge = &(vert_data.idx, other_vert);
                new_verts.push(left_vert_down_edge[edge]);
            }
            new_faces.push(Some(Face { verts: new_verts }));
        }

        Self {
            verts: new_verts,
            faces: new_faces,
            edges: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TruncationType {
    Standard,
    Alternation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Greatness {
    Lesser,
    Great,
}

/// Basic
impl Polyhedron {
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
        match n {
            3..=5 => Self::cupola_with_top(n),
            6 | 8 | 10 => Self::cupola_with_top(n / 2),
            _ => panic!("No cupola has a base or top with {n} sides"),
        }
    }

    pub fn cupola_with_top(n: usize) -> PrismLike {
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

    pub fn dual(&self) -> Self {
        // Create vertices at the centroids of each of the faces
        let mut new_verts = VertVec::new();
        let mut vert_idx_for_current_face = HashMap::<FaceIdx, VertIdx>::new();
        for (face_idx, _face) in self.faces_enumerated() {
            let new_vert_idx = new_verts.push(self.face_centroid(face_idx));
            vert_idx_for_current_face.insert(face_idx, new_vert_idx);
        }
        // For each current vert, create a new face who's vertices correspond to current model's faces
        let mut new_faces = FaceVec::new();
        for vert_data in self.vert_datas() {
            let verts = vert_data
                .clockwise_loop
                .into_iter()
                .map(|(_v, face_idx)| vert_idx_for_current_face[&face_idx])
                .collect_vec();
            new_faces.push(Some(Face { verts }));
        }
        // Construct new polygon
        Self {
            verts: new_verts,
            faces: new_faces,
            edges: HashMap::new(),
        }
    }

    /// Perform a given `operation`, and set the colours of any new edges to the given `colour`
    pub fn color_edges_added_by<T>(
        &mut self,
        color: Srgba,
        operation: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let first_new_vert_idx = self.verts.len_idx();
        let result = operation(self);
        // Colour any edges which contain one or more new vertex
        for e in self.edges() {
            if e.top_vert >= first_new_vert_idx || e.bottom_vert >= first_new_vert_idx {
                self.set_edge_color(e.bottom_vert, e.top_vert, color);
            }
        }

        result
    }

    /// Scale `self` so that the mean edge length is `1.0`.  If all edges have the same length,
    /// then this scales `self` so that _all_ edges have length `1.0`.
    pub fn normalize_edge_length(&mut self) {
        // Get average edge length
        let mut total_length = 0.0;
        let edges = self.edges();
        for e in &edges {
            total_length += e.length(self);
        }
        let average_edge_length = total_length / edges.len() as f32;
        // Scale model accordingly
        self.scale(1.0 / average_edge_length)
    }

    pub fn scale(&mut self, factor: f32) {
        self.transform(Mat4::from_scale(factor))
    }

    pub fn translate(&mut self, d: Vec3) {
        self.transform(Mat4::from_translation(d));
    }

    pub fn transform(&mut self, matrix: Mat4) {
        self.transform_verts(|v| transform_point(v, matrix))
    }

    pub fn transform_verts(&mut self, mut f: impl FnMut(Vec3) -> Vec3) {
        for v in &mut self.verts {
            *v = f(*v);
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

pub const DEFAULT_EDGE_COLOR: Srgba = Srgba::new_opaque(100, 100, 100);
const DEFAULT_WIREFRAME_COLOR: Srgba = Srgba::new_opaque(50, 50, 50);
const INSIDE_TINT: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderStyle {
    pub face: Option<FaceRenderStyle>,
    pub wireframe_edges: bool,
    pub wireframe_verts: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaceRenderStyle {
    /// Render model with solid faces and a wireframe
    Solid,
    /// Render model as ow-like origami, where each edge appears to be made from a paper module.
    /// The modules look like:
    /// ```
    ///    +--------------------------------------------+       ---+
    ///   /                                              \         |
    ///  /                                                \        | `side_ratio`
    /// +--------------------------------------------------+    ---+
    ///  \                                                / <-- internal angle = `fixed_angle`
    ///   \                                              /
    ///    +--------------------------------------------+
    /// |                                                  |
    /// |                                                  |
    /// +--------------------------------------------------+
    ///         edge length (should always be `1.0`)
    /// ```
    OwLike {
        /// The ratio of sides of the module.  If the edge length of the model is 1, then each
        /// side of the module is `side_ratio`
        side_ratio: f32,
        fixed_angle: Option<FixedAngle>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedAngle {
    pub unit_angle: Degrees,
    pub push_direction: Side,
    pub add_crinkle: bool,
}

pub struct Meshes {
    pub face_mesh: Option<Mesh>,
    pub edge_mesh: InstancedMesh,
    pub vertex_mesh: InstancedMesh,
}

impl Polyhedron {
    pub fn meshes(&self, style: RenderStyle, context: &three_d::Context) -> Meshes {
        const EDGE_RADIUS: f32 = 0.03;
        const VERTEX_RADIUS: f32 = 0.05;

        // Generate CPU-side mesh data
        let face_mesh = style.face.map(|f| self.face_mesh(f));
        let edges = match style.wireframe_edges {
            true => self.edge_instances(),
            false => Instances::default(),
        };
        let verts = match style.wireframe_verts || style.wireframe_edges {
            true => self.vertex_instances(),
            false => Instances::default(),
        };
        let vert_radius = match style.wireframe_verts {
            true => VERTEX_RADIUS,
            false => EDGE_RADIUS,
        };

        // Send mesh data to the GPU and set up instancing
        let mut sphere = CpuMesh::sphere(8);
        sphere.transform(&Mat4::from_scale(vert_radius)).unwrap();
        let mut cylinder = CpuMesh::cylinder(10);
        cylinder
            .transform(&Mat4::from_nonuniform_scale(1.0, EDGE_RADIUS, EDGE_RADIUS))
            .unwrap();
        Meshes {
            face_mesh: face_mesh.map(|m| Mesh::new(context, &m)),
            edge_mesh: InstancedMesh::new(context, &edges, &cylinder),
            vertex_mesh: InstancedMesh::new(context, &verts, &sphere),
        }
    }

    fn face_mesh(&self, style: FaceRenderStyle) -> CpuMesh {
        // Create outward-facing faces
        let faces = self.faces_to_render(style);
        let (verts, colors, tri_indices) = Self::triangulate_mesh(faces);

        // Add verts colors for inside-facing verts
        let mut all_verts = verts.clone();
        all_verts.extend_from_within(..);
        let mut all_colors = colors.clone();
        for c in colors {
            let darken = |c: u8| -> u8 { (c as f32 * INSIDE_TINT) as u8 };
            all_colors.push(Srgba {
                r: darken(c.r),
                g: darken(c.g),
                b: darken(c.b),
                a: c.a,
            });
        }
        // Add inside-facing faces
        let vert_offset = verts.len() as u32;
        let mut all_tri_indices = tri_indices.clone();
        for vs in tri_indices.chunks_exact(3) {
            all_tri_indices.extend_from_slice(&[
                vert_offset + vs[0],
                vert_offset + vs[2],
                vert_offset + vs[1],
            ]);
        }

        let mut mesh = CpuMesh {
            positions: Positions::F32(all_verts),
            colors: Some(all_colors),
            indices: Indices::U32(all_tri_indices),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh
    }

    fn faces_to_render(&self, style: FaceRenderStyle) -> Vec<(Srgba, Vec<Vec3>)> {
        let mut faces_to_render = Vec::new();
        for face in self.faces() {
            // Decide how to render them, according to the style
            match style {
                // For solid faces, just render the face as-is
                FaceRenderStyle::Solid => {
                    let verts = face
                        .verts
                        .iter()
                        .map(|vert_idx| self.verts[*vert_idx])
                        .collect_vec();
                    faces_to_render.push((DEFAULT_EDGE_COLOR, verts));
                }
                // For ow-like faces, render up to two faces per edge
                FaceRenderStyle::OwLike {
                    side_ratio,
                    fixed_angle,
                } => {
                    self.owlike_faces(&face.verts, side_ratio, fixed_angle, &mut faces_to_render);
                }
            }
        }
        faces_to_render
    }

    fn owlike_faces(
        &self,
        verts: &[VertIdx],
        side_ratio: f32,
        fixed_angle: Option<FixedAngle>,
        faces_to_render: &mut Vec<(Srgba, Vec<Vec3>)>,
    ) {
        // Geometry calculations
        let normal = self.normal_from_verts(verts);
        let in_directions = self.vertex_in_directions(verts);

        let verts_and_ins = verts.iter().zip_eq(&in_directions);
        for ((i0, in0), (i1, in1)) in verts_and_ins.circular_tuple_windows() {
            // Extract useful data
            let (v0, v1) = (self.verts[*i0], self.verts[*i1]);
            let edge_color = self.get_edge_color(*i0, *i1).unwrap_or(DEFAULT_EDGE_COLOR);
            let mut add_face = |verts: Vec<Vec3>| faces_to_render.push((edge_color, verts));

            match fixed_angle {
                // If the unit has no fixed angle, then always make the units parallel
                // to the faces
                None => add_face(vec![v0, v1, v1 + in1 * side_ratio, v0 + in0 * side_ratio]),
                Some(angle) => {
                    let FixedAngle {
                        unit_angle,
                        push_direction,
                        add_crinkle,
                    } = angle;
                    // Calculate the unit's inset angle, and therefore break the unit length down
                    // into x/y components
                    let inset_angle = unit_inset_angle(verts.len(), unit_angle.into());
                    let l_in = side_ratio * inset_angle.cos();
                    let l_up = side_ratio * inset_angle.sin();
                    // Pick a normal to use based on the push direction
                    let unit_up = match push_direction {
                        Side::Out => normal,
                        Side::In => -normal,
                    };
                    let up = unit_up * l_up;
                    // Add faces
                    if add_crinkle {
                        let v0_peak = v0 + l_in * in0 * 0.5 + up * 0.5;
                        let v1_peak = v1 + l_in * in1 * 0.5 + up * 0.5;
                        add_face(vec![v0, v1, v1_peak, v0_peak]);
                        add_face(vec![v0_peak, v1_peak, v1 + l_in * in1, v0 + l_in * in0]);
                    } else {
                        add_face(vec![v0, v1, v1 + l_in * in1 + up, v0 + l_in * in0 + up]);
                    }
                }
            }
        }
    }

    fn vertex_in_directions(&self, verts: &[VertIdx]) -> Vec<Vec3> {
        let mut in_directions = Vec::new();
        for (i0, i1, i2) in verts.iter().circular_tuple_windows() {
            let v0 = self.verts[*i0];
            let v1 = self.verts[*i1];
            let v2 = self.verts[*i2];
            let in_vec = ((v0 - v1) + (v2 - v1)).normalize();
            // Normalize so that it has a projected length of 1 perpendicular to the edge
            let normalized = normalize_perpendicular_to(in_vec, v2 - v1);
            in_directions.push(normalized);
        }
        in_directions.rotate_right(1); // The 0th in-direction is actually from the 1st vertex
        in_directions
    }

    fn triangulate_mesh(faces: Vec<(Srgba, Vec<Vec3>)>) -> (Vec<Vec3>, Vec<Srgba>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut colors = Vec::new();
        let mut tri_indices = Vec::new();

        for (color, face_verts) in faces {
            // Add all vertices from this face.  We have to duplicate the vertices so that each
            // face gets flat shading
            let first_vert_idx = verts.len() as u32;
            verts.extend_from_slice(&face_verts);
            // Add colours for this face's vertices
            colors.extend(std::iter::repeat(color).take(face_verts.len()));
            // Add the vert indices for this face
            for i in 2..face_verts.len() as u32 {
                tri_indices.extend_from_slice(&[
                    first_vert_idx,
                    first_vert_idx + i - 1,
                    first_vert_idx + i,
                ]);
            }
        }
        (verts, colors, tri_indices)
    }

    fn edge_instances(&self) -> Instances {
        let mut colors = Vec::new();
        let mut transformations = Vec::new();
        for edge in self.edges() {
            colors.push(
                self.get_edge_color(edge.bottom_vert, edge.top_vert)
                    .unwrap_or(DEFAULT_WIREFRAME_COLOR),
            );
            transformations.push(edge_transform(
                self.verts[edge.bottom_vert],
                self.verts[edge.top_vert],
            ));
        }

        Instances {
            transformations,
            colors: Some(colors),
            ..Default::default()
        }
    }

    fn vertex_instances(&self) -> Instances {
        Instances {
            transformations: self
                .verts
                .iter()
                .cloned()
                .map(Mat4::from_translation)
                .collect_vec(),
            colors: Some(
                std::iter::repeat(DEFAULT_WIREFRAME_COLOR)
                    .take(self.verts.len())
                    .collect_vec(),
            ),
            ..Default::default()
        }
    }
}

fn unit_inset_angle(n: usize, unit_angle: Radians) -> Radians {
    let interior_ngon_angle = Rad(PI - (PI * 2.0 / n as f32));
    let mut angle = angle_in_spherical_triangle(unit_angle, interior_ngon_angle, unit_angle);
    // Correct NaN angles (if the corner is too wide or the unit is level with the face and
    // rounding errors cause us to get `NaN`)
    if angle.0.is_nan() {
        angle.0 = 0.0;
    }
    angle
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

impl Polyhedron {
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
            edges: HashMap::new(),
        };
        m.make_centred();
        m
    }

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

    pub fn vert_positions(&self) -> &[Vec3] {
        self.verts.as_raw_slice()
    }

    // TODO: Handle open edges correctly
    pub fn vert_datas(&self) -> VertVec<VertData> {
        // If we are at vertex `a` looking down an edge at vertex `b`, then the next edge
        // *clockwise* around `a` will lead to `c` having gone over face `f` where:
        // `(c, f) = next_vert[&(a, b)]`.
        let mut next_vert = HashMap::<(VertIdx, VertIdx), (VertIdx, FaceIdx)>::new();
        for (face_idx, face) in self.faces_enumerated() {
            for (c, a, b) in face.verts.iter().copied().circular_tuple_windows() {
                next_vert.insert((a, b), (c, face_idx));
            }
        }

        // Loop round each vertex in turn by following links in `next_vert`
        let mut datas = VertVec::new();
        for vert_idx in self.verts.indices() {
            // Find an arbitrary edge adjacent to this vert
            let &(a, first_other_vert) = next_vert
                .keys()
                .find(|(i, _)| *i == vert_idx)
                .expect("Every vertex should have an adjacent face");
            assert_eq!(a, vert_idx);
            // Loop round this vertex, tracking which edges/faces we go over
            let mut other_vert = first_other_vert;
            let mut clockwise_loop = Vec::new();
            loop {
                let (next_other_vert, face_idx) = next_vert[&(vert_idx, other_vert)];
                clockwise_loop.push((other_vert, face_idx));
                // Move to the other side of the face, and end if we've fully looped
                other_vert = next_other_vert;
                if other_vert == first_other_vert {
                    break;
                }
            }
            // Add this vertex's loop
            datas.push(VertData {
                idx: vert_idx,
                clockwise_loop,
            });
        }
        datas
    }

    pub fn edges(&self) -> Vec<Edge> {
        let mut edges = HashMap::<(VertIdx, VertIdx), Edge>::new();
        for (face_idx, face) in self.faces_enumerated() {
            for (&v1, &v2) in face.verts.iter().circular_tuple_windows() {
                let key = (v1.min(v2), v1.max(v2));
                if let Some(edge) = edges.get_mut(&key) {
                    edge.add_left_face(v1, v2, face_idx, self);
                } else {
                    let direction = self.verts[v2] - self.verts[v1];
                    let edge = Edge {
                        bottom_vert: v1,
                        top_vert: v2,
                        length: direction.magnitude(),
                        color: self.get_edge_color(v1, v2),
                        right_face: face_idx,
                        closed: None,
                    };
                    edges.insert(key, edge);
                }
            }
        }
        // Dedup and return edges
        edges.into_values().collect_vec()
    }

    pub fn get_edge_color(&self, i0: VertIdx, i1: VertIdx) -> Option<Srgba> {
        let key = (i0.min(i1), i0.max(i1));
        self.edges.get(&key).and_then(|e| e.color)
    }

    pub fn set_edge_color(&mut self, i0: VertIdx, i1: VertIdx, color: Srgba) {
        let key = (i0.min(i1), i0.max(i1));
        self.edges.insert(key, EdgeData { color: Some(color) });
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

    pub fn face_centroid(&self, face: FaceIdx) -> Vec3 {
        let verts = &self.get_face(face).verts;
        let mut total = Vec3::zero();
        for v in verts {
            total += self.verts[*v];
        }
        total / verts.len() as f32
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
        self.faces[face].as_ref().unwrap().normal(self)
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

#[derive(Debug, Clone)]
pub struct VertData {
    pub idx: VertIdx,
    pub clockwise_loop: Vec<(VertIdx, FaceIdx)>,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub bottom_vert: VertIdx,
    pub top_vert: VertIdx,
    pub length: f32,
    pub color: Option<Srgba>,
    pub right_face: FaceIdx,
    pub closed: Option<ClosedEdgeData>,
}

#[derive(Debug, Clone)]
pub struct ClosedEdgeData {
    pub left_face: FaceIdx,
    pub dihedral_angle: Radians,
}

impl Edge {
    pub fn length(&self, polyhedron: &Polyhedron) -> f32 {
        let v1 = polyhedron.verts[self.bottom_vert];
        let v2 = polyhedron.verts[self.top_vert];
        v1.distance(v2)
    }

    pub fn dihedral_angle(&self) -> Option<Radians> {
        self.closed.as_ref().map(|c| c.dihedral_angle)
    }

    fn add_left_face(
        &mut self,
        v1: VertIdx,
        v2: VertIdx,
        left_face: FaceIdx,
        polyhedron: &Polyhedron,
    ) {
        assert_eq!((v1, v2), (self.top_vert, self.bottom_vert));
        self.closed = Some(ClosedEdgeData {
            left_face,
            dihedral_angle: self.get_dihedral_angle(left_face, polyhedron),
        });
    }

    fn get_dihedral_angle(&self, left_face: FaceIdx, polyhedron: &Polyhedron) -> Radians {
        let direction =
            (polyhedron.verts[self.top_vert] - polyhedron.verts[self.bottom_vert]).normalize();
        // Get face normals
        let left_normal = polyhedron.face_normal(left_face);
        let right_normal = polyhedron.face_normal(self.right_face);
        // Get vectors pointing along the faces, perpendicular to this edge
        let left_tangent = left_normal.cross(direction);
        let right_tangent = right_normal.cross(-direction);
        // Use these four vectors to compute the dihedral angle
        let mut dihedral = left_tangent.angle(right_tangent);
        let is_concave = left_tangent.dot(right_normal) < 0.0;
        if is_concave {
            dihedral = Radians::full_turn() - dihedral;
        }
        dihedral
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
