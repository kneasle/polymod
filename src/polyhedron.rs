use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    f32::consts::PI,
};

use itertools::Itertools;
use ordered_float::OrderedFloat;
use three_d::{
    vec3, Angle, Deg, InnerSpace, Mat4, MetricSpace, Rad, Radians, SquareMatrix, Vec3, Vec4, Zero,
};

use crate::utils::{lerp3, Side};

/// A polygonal model where all faces are regular and all edges have unit length.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polyhedron {
    verts: VertVec<Vertex>,
    /// Each face of the model, listing vertices in clockwise order
    faces: FaceVec<Option<Face>>,

    /// Indexed colors for each side of each edge.  If `edge_colors[(a, b)] = idx` then the side
    /// of the edge who's top-right vertex is `a` and who's bottom-right vertex is `b` will be
    /// given the color at `color_index[idx]`.
    ///
    /// Note that the exact colors are stored outside of the `Polyhedron` but are referred to here
    /// by [`String`] identifiers.  This allows colors to be merged correctly while merging
    /// polyhedra, but also allows the colors themselves to be modified without mutating the
    /// underlying `Polyhedron` (which shouldn't care how it will be rendered).
    half_edge_colors: HashMap<(VertIdx, VertIdx), String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    pos: Vec3,
}

impl Vertex {
    pub fn pos(&self) -> Vec3 {
        self.pos
    }
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

    pub fn centroid(&self, polyhedron: &Polyhedron) -> Vec3 {
        let mut total = Vec3::zero();
        for v in &self.verts {
            total += polyhedron.vert_pos(*v);
        }
        total / self.verts.len() as f32
    }

    pub fn is_regular(&self, polyhedron: &Polyhedron) -> bool {
        let expected_angle = Rad::full_turn() / self.order() as f32;
        for (v1, v2, v3) in self
            .vert_positions(polyhedron)
            .iter()
            .circular_tuple_windows()
        {
            let angle = (v2 - v1).angle(v3 - v2);
            if f32::abs(angle.0 - expected_angle.0) > 0.0001 {
                return false; // Incorrect angle => irregular face
            }
        }
        true
    }

    pub fn is_flat(&self, polyhedron: &Polyhedron) -> bool {
        let normal = self.normal(polyhedron);
        for (v1, v2) in self
            .vert_positions(polyhedron)
            .iter()
            .circular_tuple_windows()
        {
            if (v2 - v1).dot(normal) > 0.00001 {
                return false; // Edge is not perpendicular to normal => aplanar face
            }
        }
        true
    }

    pub fn normal(&self, polyhedron: &Polyhedron) -> Vec3 {
        polyhedron.normal_from_verts(&self.verts)
    }

    pub fn vert_positions(&self, polyhedron: &Polyhedron) -> Vec<Vec3> {
        self.verts
            .iter()
            .map(|idx| polyhedron.vert_pos(*idx))
            .collect_vec()
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
        let mut new_verts = VertVec::<Vertex>::new();
        let mut new_vert_on_edge = HashMap::<(VertIdx, VertIdx), VertIdx>::new();
        let mut add_vert = |v1: VertIdx, v2: VertIdx| {
            let pos = lerp3(self.vert_pos(v1), self.vert_pos(v2), lerp_factor);
            let new_idx = new_verts.push(Vertex { pos });
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
            ..Default::default()
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
            let edge_midpoint = lerp3(
                self.vert_pos(face.verts[0]),
                self.vert_pos(face.verts[1]),
                0.5,
            );
            let new_face_centroid = centroid * scaling_factor;
            let new_y_axis = (edge_midpoint - centroid).normalize();
            let new_x_axis = centroid.normalize().cross(new_y_axis);
            // Create vertices for the new face
            let mut face_verts = Vec::new();
            for i in 0..new_face_order {
                let (x, y) = new_face_geometry.offset_point(i, 0.5);
                let pos = new_face_centroid + x * new_x_axis + y * new_y_axis;
                let new_vert_idx = new_verts.push(Vertex { pos });
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
            ..Default::default()
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
            let edge_midpoint = lerp3(
                self.vert_pos(face.verts[0]),
                self.vert_pos(face.verts[1]),
                0.5,
            );
            let new_face_centroid = centroid.normalize() * new_inradius;
            let new_y_axis = (edge_midpoint - centroid).normalize();
            let new_x_axis = centroid.normalize().cross(new_y_axis);
            // Create vertices for the new face
            let mut face_verts = Vec::new();
            for i in 0..face_order {
                let (x, y) = face_geometry.offset_point(i, rotation);
                let pos = new_face_centroid + x * new_x_axis + y * new_y_axis;
                let new_vert_idx = new_verts.push(Vertex { pos });
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
            ..Default::default()
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

    /// Creates a cupola with `{bottom,top}_face` set such that the face with `n` sides is always
    /// the 'bottom' face
    pub fn oriented_cupola(n: usize) -> PrismLike {
        let mut cupola = Self::cupola(n);
        if n <= 5 {
            // Swap top and bottom faces, since we are creating the cupola 'upside-down'
            std::mem::swap(&mut cupola.bottom_face, &mut cupola.top_face);
        }
        cupola
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

    pub fn cuboctahedron_prismlike() -> PrismLike {
        let poly = Self::cuboctahedron();
        PrismLike {
            bottom_face: poly.get_face_with_normal(-Vec3::unit_y()),
            top_face: poly.get_face_with_normal(Vec3::unit_y()),
            poly,
        }
    }

    pub fn rotunda() -> PrismLike {
        let poly = Self::icosidodecahedron();
        // Strip any vertices which are below the XZ plane (i.e. those with negative y-coordinates)
        let mut vert_map = VertVec::new(); // Maps old vert indices to new vert indices
        let mut new_verts = VertVec::new();
        for v in &poly.verts {
            vert_map.push(if v.pos.y > -0.00001 {
                Some(new_verts.push(v.clone()))
            } else {
                None
            });
        }
        // Recreate all the faces where all their vertices are preserved
        let mut new_faces = FaceVec::new();
        'face_loop: for face in poly.faces() {
            let mut new_verts = Vec::new();
            for &vert_idx in &face.verts {
                let Some(new_vert_idx) = vert_map[vert_idx] else {
                    continue 'face_loop; // If this face contains a vertex below XZ plane, skip entire face
                };
                new_verts.push(new_vert_idx);
            }
            new_faces.push(Some(Face { verts: new_verts }));
        }
        // Create a new decagon for the base
        let mut verts_on_xz_plane = new_verts
            .iter()
            .positions(|v| v.pos.y.abs() < 0.000001)
            .map(VertIdx::new)
            .collect_vec();
        verts_on_xz_plane.sort_by_key(|v_idx| {
            let pos = new_verts[*v_idx].pos;
            OrderedFloat(-f32::atan2(pos.x, pos.z))
        });
        let bottom_face = new_faces.push(Some(Face {
            verts: verts_on_xz_plane,
        }));

        // Create a new poly for the rotunda
        let poly = Polyhedron {
            verts: new_verts,
            faces: new_faces,
            ..Default::default()
        };
        PrismLike {
            top_face: poly.get_face_with_normal(Vec3::unit_y()),
            bottom_face,
            poly,
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

#[allow(dead_code)]
impl Polyhedron {
    /// 'Extend' this polyhedron by adding a copy of `other` onto the given `face`.
    /// The `other` polyhedron is attached by `its_face`.
    pub fn extend(
        &mut self,
        face: FaceIdx,
        other: &Self,
        its_face: FaceIdx,
        rotation: usize,
    ) -> FaceVec<FaceIdx> {
        self.merge(face, other, its_face, rotation, MergeDir::Extend)
    }

    pub fn extend_pyramid(&mut self, face: FaceIdx) {
        let n = self.face_order(face);
        let pyramid = Polyhedron::pyramid(n);
        self.extend(face, &pyramid.poly, pyramid.base_face, 0);
    }

    pub fn extend_prism(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Extend, 0, Self::prism)
    }

    pub fn extend_antiprism(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Extend, 0, Self::antiprism)
    }

    pub fn extend_cupola(&mut self, face: FaceIdx, gyro: bool) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Extend, gyro as usize, Self::oriented_cupola)
    }

    pub fn extend_rotunda(&mut self, face: FaceIdx, gyro: bool) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Extend, gyro as usize, |_n| Self::rotunda())
    }

    pub fn extend_cuboctahedron(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Extend, 0, |_n| {
            Self::cuboctahedron_prismlike()
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
    ) -> FaceVec<FaceIdx> {
        self.merge(face, other, its_face, rotation, MergeDir::Excavate)
    }

    pub fn excavate_prism(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Excavate, 0, Self::prism)
    }

    pub fn excavate_antiprism(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Excavate, 0, Self::antiprism)
    }

    pub fn excavate_cupola(&mut self, face: FaceIdx, gyro: bool) -> FaceIdx {
        self.merge_prismlike(
            face,
            MergeDir::Excavate,
            gyro as usize,
            Self::oriented_cupola,
        )
    }

    pub fn excavate_rotunda(&mut self, face: FaceIdx, gyro: bool) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Excavate, gyro as usize, |_n| {
            Self::rotunda()
        })
    }

    pub fn excavate_cuboctahedron(&mut self, face: FaceIdx) -> FaceIdx {
        self.merge_prismlike(face, MergeDir::Excavate, 0, |_n| {
            Self::cuboctahedron_prismlike()
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
        let tracked_faces = self.merge(face, &poly, bottom_face, rotation, dir);
        tracked_faces[top_face]
    }

    fn merge(
        &mut self,
        face: FaceIdx,
        other: &Self,
        its_face: FaceIdx,
        rotation: usize,
        dir: MergeDir,
    ) -> FaceVec<FaceIdx> {
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
            .map(|v| self.add_vert(transform_point(v.pos, transform)))
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

        // Merge color assignments
        for ((v1, v2), col_name) in &other.half_edge_colors {
            let new_v1 = new_vert_indices[*v1];
            let new_v2 = new_vert_indices[*v2];
            self.half_edge_colors
                .insert((new_v1, new_v2), col_name.to_owned());
        }

        // Return the new indices of the faces in `other`
        new_face_indices
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
        let mut new_verts = VertVec::<Vertex>::new();
        let mut vert_idx_for_current_face = HashMap::<FaceIdx, VertIdx>::new();
        for (face_idx, _face) in self.faces_enumerated() {
            let pos = self.face_centroid(face_idx);
            let new_vert_idx = new_verts.push(Vertex { pos });
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
            ..Default::default()
        }
    }

    pub fn color_edges_added_by<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> T,
        color: &str,
    ) -> T {
        let (edges_added, value) = self.get_edges_added_by(operation);
        for e in edges_added {
            self.set_full_edge_color(e, color);
        }
        value
    }

    /// Perform a given `operation`, and set the colors of any new edges to the given `color`
    pub fn get_edges_added_by<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> T,
    ) -> (Vec<EdgeId>, T) {
        let first_new_vert_idx = self.verts.len_idx();
        let result = operation(self);
        // Color any edges which contain one or more new vertex
        let mut edges_added = Vec::new();
        for e in self.edges() {
            if e.top_vert >= first_new_vert_idx || e.bottom_vert >= first_new_vert_idx {
                edges_added.push(EdgeId::new(e.bottom_vert, e.top_vert));
            }
        }
        (edges_added, result)
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
            v.pos = f(v.pos);
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

///////////
// UTILS //
///////////

const VERTEX_MERGE_DIST: f32 = 0.00001;
const VERTEX_MERGE_DIST_SQUARED: f32 = VERTEX_MERGE_DIST * VERTEX_MERGE_DIST;

impl Polyhedron {
    fn new(verts: Vec<Vec3>, faces: FaceVec<Vec<usize>>) -> Self {
        let mut m = Self {
            verts: verts.into_iter().map(|pos| Vertex { pos }).collect(),
            faces: faces
                .into_iter()
                .map(|verts| Face {
                    verts: verts.into_iter().map(VertIdx::new).collect_vec(),
                })
                .map(Some)
                .collect(),

            half_edge_colors: HashMap::default(),
        };
        m.make_centred();
        m
    }

    /* COLORS */

    pub fn color_face(&mut self, face: FaceIdx, color_name: &str) {
        let verts = &self.get_face(face).verts.clone();
        for (v1, v2) in verts.iter().circular_tuple_windows() {
            self.set_half_edge_color(*v2, *v1, color_name);
        }
    }

    /// Give both sides of every edge of this model the given `color`
    pub fn color_all_edges(&mut self, color_name: &str) {
        for edge in self.edges() {
            self.set_full_edge_color(edge.id(), color_name);
        }
    }

    pub fn set_full_edge_color(&mut self, edge: EdgeId, color_name: &str) {
        self.set_half_edge_color(edge.v1(), edge.v2(), color_name);
        self.set_half_edge_color(edge.v2(), edge.v1(), color_name);
    }

    pub fn set_half_edge_color(&mut self, v1: VertIdx, v2: VertIdx, color_name: &str) {
        self.half_edge_colors
            .insert((v1, v2), color_name.to_owned());
    }

    pub fn get_edge_side_color(&self, a: VertIdx, b: VertIdx) -> Option<&str> {
        self.half_edge_colors.get(&(a, b)).map(String::as_str)
    }

    /// Create a vertex at the given coords `p`, returning its index.  If there's already a vertex
    /// at `p`, then its index is returned.
    pub fn add_vert(&mut self, pos: Vec3) -> VertIdx {
        // Look for existing vertices to dedup with
        for (idx, v) in self.verts.iter_enumerated() {
            if (pos - v.pos).magnitude2() < VERTEX_MERGE_DIST_SQUARED {
                return idx;
            }
        }
        // If vertex isn't already present, add a new one
        self.verts.push(Vertex { pos })
    }

    pub fn vert_pos(&self, idx: VertIdx) -> Vec3 {
        self.verts[idx].pos
    }

    pub fn verts(&self) -> &[Vertex] {
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
        let mut edges = HashMap::<EdgeId, Edge>::new();
        for (face_idx, face) in self.faces_enumerated() {
            for (&v1, &v2) in face.verts.iter().circular_tuple_windows() {
                let key = EdgeId::new(v1, v2);
                if let Some(edge) = edges.get_mut(&key) {
                    edge.add_left_face(v1, v2, face_idx, self);
                } else {
                    let direction = self.vert_pos(v2) - self.vert_pos(v1);
                    let edge = Edge {
                        bottom_vert: v1,
                        top_vert: v2,
                        length: direction.magnitude(),
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

    /// Gets an [`Iterator`] over the [indices](FaceIdx) of every face in `self` which has `n`
    /// sides
    pub fn ngons(&self, n: usize) -> Vec<FaceIdx> {
        self.faces
            .iter()
            .positions(move |f| f.as_ref().map(|face| face.verts.len()) == Some(n))
            .map(FaceIdx::new)
            .collect_vec()
    }

    /// Gets the lowest-indexed face in `self` which has `n` sides.
    pub fn get_ngon(&self, n: usize) -> FaceIdx {
        self.ngons(n)[0]
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

    pub fn get_face_with_normal(&self, normal: Vec3) -> FaceIdx {
        let (idx, _face) = self
            .faces_enumerated()
            .find(|(_idx, f)| f.normal(self).angle(normal) < Rad(0.01))
            .unwrap();
        idx
    }

    pub fn face_centroid(&self, face: FaceIdx) -> Vec3 {
        self.get_face(face).centroid(self)
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
        let translation = Mat4::from_translation(self.vert_pos_on_face(verts, rotation));
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
        let v0 = self.vert_pos_on_face(verts, rotation);
        let v1 = self.vert_pos_on_face(verts, rotation + vert_offset);
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

    fn face_normal(&self, face: FaceIdx) -> Vec3 {
        self.faces[face].as_ref().unwrap().normal(self)
    }

    fn normal_from_verts(&self, verts: &[VertIdx]) -> Vec3 {
        assert!(verts.len() >= 3);
        let v0 = self.vert_pos_on_face(verts, 0);
        let v1 = self.vert_pos_on_face(verts, 1);
        let v2 = self.vert_pos_on_face(verts, 2);
        let d1 = v1 - v0;
        let d2 = v2 - v0;
        d1.cross(d2).normalize()
    }

    fn vert_pos_on_face(&self, face: &[VertIdx], vert: usize) -> Vec3 {
        let vert_idx = face[vert % face.len()];
        self.vert_pos(vert_idx)
    }

    pub fn centroid(&self) -> Vec3 {
        let mut total = Vec3::zero();
        for v in &self.verts {
            total += v.pos;
        }
        total / self.verts.len() as f32
    }

    /// Returns the radius of the smallest sphere which fits entirely around this `Polyhedron`.
    ///
    /// Note: currently this returns the smallest sphere centred around the centroid, which is
    /// usually the same for high-symmetry models but may not always be the smallest sphere.
    pub fn outsphere_radius(&self) -> f32 {
        let centroid = self.centroid();
        self.verts
            .iter()
            .map(|v| v.pos.distance(centroid))
            .reduce(f32::max)
            .unwrap_or(0.0)
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
        let v1 = polyhedron.vert_pos(self.bottom_vert);
        let v2 = polyhedron.vert_pos(self.top_vert);
        v1.distance(v2)
    }

    pub fn dihedral_angle(&self) -> Option<Radians> {
        self.closed.as_ref().map(|c| c.dihedral_angle)
    }

    pub fn angle_type(&self) -> EdgeAngleType {
        let half_turn = Radians::full_turn() / 2.0;
        match self.dihedral_angle() {
            Some(angle) => match angle.partial_cmp(&half_turn).unwrap() {
                Ordering::Less => EdgeAngleType::Convex,
                Ordering::Equal => EdgeAngleType::Planar,
                Ordering::Greater => EdgeAngleType::Concave,
            },
            None => EdgeAngleType::Open,
        }
    }

    pub fn has_verts(&self, v1: VertIdx, v2: VertIdx) -> bool {
        (self.top_vert == v1 && self.bottom_vert == v2)
            || (self.top_vert == v2 && self.bottom_vert == v1)
    }

    pub fn id(&self) -> EdgeId {
        EdgeId::new(self.bottom_vert, self.top_vert)
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
        let direction = (polyhedron.vert_pos(self.top_vert)
            - polyhedron.vert_pos(self.bottom_vert))
        .normalize();
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeAngleType {
    Convex,
    Planar,
    Concave,
    Open,
}

impl EdgeAngleType {
    pub fn as_char(self) -> char {
        match self {
            EdgeAngleType::Convex => 'v',
            EdgeAngleType::Planar => 'p',
            EdgeAngleType::Concave => 'c',
            EdgeAngleType::Open => '-',
        }
    }
}

fn transform_point(v: Vec3, matrix: Mat4) -> Vec3 {
    let v4 = v.extend(1.0);
    let trans_v4 = matrix * v4;
    trans_v4.truncate()
}

/// A pair of vertices which make up an edge, guaranteeing that `v1 < v2`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeId {
    v1: VertIdx,
    v2: VertIdx,
}

impl EdgeId {
    pub fn new(v1: VertIdx, v2: VertIdx) -> Self {
        Self {
            v1: VertIdx::min(v1, v2),
            v2: VertIdx::max(v1, v2),
        }
    }

    pub fn v1(self) -> VertIdx {
        self.v1
    }

    pub fn v2(self) -> VertIdx {
        self.v2
    }

    #[allow(dead_code)]
    pub fn verts(self) -> (VertIdx, VertIdx) {
        (self.v1, self.v2)
    }
}

index_vec::define_index_type! { pub struct VertIdx = usize; }
index_vec::define_index_type! { pub struct FaceIdx = usize; }
pub type VertVec<T> = index_vec::IndexVec<VertIdx, T>;
pub type FaceVec<T> = index_vec::IndexVec<FaceIdx, T>;
