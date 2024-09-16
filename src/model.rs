use std::{collections::HashSet, f32::consts::PI};

use indexmap::IndexMap;
use itertools::Itertools;
use three_d::{
    egui::{self, Color32},
    vec3, Angle, CpuMesh, Deg, Degrees, Indices, InnerSpace, InstancedMesh, Instances, Mat4,
    Positions, Quat, Rad, Radians, Vec3,
};

use crate::{
    polyhedron::{Cube, EdgeId, FaceIdx, PrismLike, Pyramid, VertIdx},
    utils::{angle_in_spherical_triangle, darken_color, egui_color_to_srgba, lerp_color},
};
use crate::{
    polyhedron::{Face, Polyhedron},
    utils::{normalize_perpendicular_to, Side},
    SMALL_SPACE,
};

#[derive(Debug, Clone)]
pub struct Model {
    full_name: String,

    polyhedron: Polyhedron,

    // Display settings & Coloring
    view_geometry_settings: ViewGeomSettings,
    default_color: Color32,
    colors: ColorMap,
}

pub type ColorMap = IndexMap<String, Color32>;

impl Model {
    pub fn new(full_name: String, poly: Polyhedron) -> Self {
        Self::with_colors(full_name, poly, ColorMap::new())
    }

    pub fn with_colors(full_name: String, poly: Polyhedron, colors: ColorMap) -> Self {
        let mut model = Self {
            full_name,
            polyhedron: poly,

            view_geometry_settings: ViewGeomSettings::default(),
            default_color: Color32::GRAY,
            colors,
        };
        model.fill_color_map();
        model
    }

    pub fn full_name(&self) -> &str {
        &self.full_name
    }

    pub fn full_name_mut(&mut self) -> &mut String {
        &mut self.full_name
    }

    pub const PATH_DELIMITER: char = '\\';

    /// Returns the path of this `Model` by interpreting its `full_name` as a path delimited by
    /// back-slashes.  Note that this does not include the final segment of `full_name`, as that
    /// is considered the model's `name`.
    ///
    /// I.e. a model with name `r"Built-In\Platonic\Cube"` would return a path of
    /// `["Built-In", "Platonic"]`.  The model's `name` is `"Cube"`
    pub fn path(&self) -> Vec<&str> {
        let mut path = self.full_name.split(Self::PATH_DELIMITER).collect_vec();
        path.pop(); // Remove name from the end of path
        path
    }

    /// Just the `name` of this model.  I.e. this is `full_name` but with the initial `path`
    /// segments removed.
    pub fn name(&self) -> &str {
        self.full_name()
            .split(Self::PATH_DELIMITER)
            .next_back()
            .unwrap()
    }

    pub fn polyhedron(&self) -> &Polyhedron {
        &self.polyhedron
    }

    pub fn view_geometry_settings(&self) -> &ViewGeomSettings {
        &self.view_geometry_settings
    }

    pub fn draw_view_geom_gui(&mut self, ui: &mut egui::Ui) {
        self.view_geometry_settings.gui(ui);
        ui.add_space(SMALL_SPACE);
        ui.strong("Colors");
        self.draw_colors_gui(ui);
    }

    fn draw_colors_gui(&mut self, ui: &mut egui::Ui) {
        let mut draw_color = |col: &mut Color32, label: &str| {
            ui.horizontal(|ui| {
                ui.color_edit_button_srgba(col);
                ui.label(label);
            })
        };

        // Draw all colors, including the default one
        draw_color(&mut self.default_color, "(Default)");
        for (name, color) in &mut self.colors {
            draw_color(color, name);
        }
    }

    /* HELPERS */

    /// Make sure that the map of colours has an entry for every color referenced in the
    /// [`Polyhedron`].
    pub fn fill_color_map(&mut self) {
        const BLUE: Color32 = Color32::from_rgb(51, 90, 255);
        const RED: Color32 = Color32::from_rgb(255, 51, 90);
        const GREEN: Color32 = Color32::from_rgb(90, 255, 51);

        let mut color_iter = [BLUE, RED, GREEN].into_iter().cycle();
        for (_edge, color) in self.polyhedron.half_edge_colors() {
            if !self.colors.contains_key(color) {
                self.colors
                    .insert(color.to_owned(), color_iter.next().unwrap());
            }
        }
    }

    pub fn get_edge_side_color(&self, v1: VertIdx, v2: VertIdx) -> Color32 {
        let color_name = self.polyhedron.get_edge_side_color(v1, v2);
        *color_name
            .and_then(|name| self.colors.get(name))
            .unwrap_or(&self.default_color)
    }

    pub fn get_mixed_edge_color(&self, v1: VertIdx, v2: VertIdx) -> Color32 {
        let left_color = self.get_edge_side_color(v1, v2);
        let right_color = self.get_edge_side_color(v2, v1);
        lerp_color(left_color, right_color, 0.5)
    }

    pub fn get_color(&self, color: Option<&str>) -> Color32 {
        *color
            .and_then(|name| self.colors.get(name))
            .unwrap_or(&self.default_color)
    }
}

/////////////
// STYLING //
/////////////

#[derive(Debug, Clone)]
pub struct ViewGeomSettings {
    pub style: FaceGeomStyle,
    // Flat ow-like unit
    pub side_ratio: f32,
    // Ow-like unit
    pub paper_ratio_w: usize,
    pub paper_ratio_h: usize,
    pub unit: OwUnit,
    pub direction: Side,
    pub add_crinkle: bool,
    // Wireframe
    pub wireframe_edges: bool,
    pub wireframe_verts: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceGeomStyle {
    None,
    Solid,
    OwLikeFlat,
    OwLikeAngled,
}

impl Default for ViewGeomSettings {
    fn default() -> Self {
        ViewGeomSettings {
            style: FaceGeomStyle::OwLikeAngled,
            side_ratio: 0.25,

            paper_ratio_w: 3,
            paper_ratio_h: 2,
            unit: OwUnit::Custom3468,
            direction: Side::In,
            add_crinkle: false,

            wireframe_edges: false,
            wireframe_verts: false,
        }
    }
}

impl ViewGeomSettings {
    pub fn face_render_style(&self) -> Option<FaceRenderStyle> {
        Some(match self.style {
            FaceGeomStyle::None => return None,
            FaceGeomStyle::Solid => FaceRenderStyle::Solid,
            FaceGeomStyle::OwLikeFlat => FaceRenderStyle::OwLike {
                side_ratio: self.side_ratio,
                non_flat: None,
            },
            FaceGeomStyle::OwLikeAngled => {
                // Model the geometry of the unit
                let OwUnitGeometry {
                    paper_aspect,
                    spine_length_factor: spine_length,
                    supported_angles,
                } = self.ow_unit_geometry().unwrap();
                let unit_width = paper_aspect / 4.0;
                // Construct unit info
                FaceRenderStyle::OwLike {
                    side_ratio: unit_width / spine_length,
                    non_flat: Some(NonFlat {
                        supported_angles,
                        push_direction: self.direction,
                        add_crinkle: self.add_crinkle,
                    }),
                }
            }
        })
    }

    pub fn ow_unit_geometry(&self) -> Option<OwUnitGeometry> {
        (self.style == FaceGeomStyle::OwLikeAngled)
            .then(|| self.unit.geometry(self.paper_aspect_ratio()))
    }

    fn paper_aspect_ratio(&self) -> f32 {
        self.paper_ratio_h as f32 / self.paper_ratio_w as f32
    }

    pub fn gui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Faces");
        ui.radio_value(&mut self.style, FaceGeomStyle::None, "None");
        ui.radio_value(&mut self.style, FaceGeomStyle::Solid, "Solid");
        ui.radio_value(
            &mut self.style,
            FaceGeomStyle::OwLikeFlat,
            "Ow-like edge unit (flat)",
        );
        ui.indent("ow-like-flat", |ui| {
            ui.horizontal(|ui| {
                ui.label("Side ratio: ");
                ui.add(egui::Slider::new(&mut self.side_ratio, 0.0..=1.0).step_by(0.05));
            });
        });
        ui.radio_value(
            &mut self.style,
            FaceGeomStyle::OwLikeAngled,
            "Ow-like edge unit",
        );
        ui.indent("ow-like-angled", |ui| {
            egui::ComboBox::new("ow-unit", "")
                .selected_text(self.unit.name())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for unit in OwUnit::ALL {
                        ui.selectable_value(&mut self.unit, unit, unit.name());
                    }
                });
            ui.horizontal(|ui| {
                ui.label("Folded from");
                ui.add(egui::DragValue::new(&mut self.paper_ratio_w));
                ui.label(":");
                ui.add(egui::DragValue::new(&mut self.paper_ratio_h));
                ui.label("paper.");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.direction, Side::In, "Push in");
                ui.selectable_value(&mut self.direction, Side::Out, "Push out");
            });
            ui.checkbox(&mut self.add_crinkle, "Crinkle");
        });

        ui.add_space(SMALL_SPACE);
        ui.strong("Wireframe");
        ui.checkbox(&mut self.wireframe_edges, "Edges");
        ui.checkbox(&mut self.wireframe_verts, "Vertices");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwUnit {
    Deg60,
    SturdyEdgeModule90,
    CustomDeg90,
    Custom3468,
    CustomDeg108,
    Deg120,
    Deg135,
    Ow68,
}

#[derive(Debug, Clone)]
pub struct OwUnitGeometry {
    pub paper_aspect: f32,
    pub spine_length_factor: f32, // As a factor of paper width
    pub supported_angles: Vec<Degrees>,
}

impl OwUnit {
    const ALL: [Self; 8] = [
        OwUnit::Deg60,
        OwUnit::SturdyEdgeModule90,
        OwUnit::CustomDeg90,
        OwUnit::Custom3468,
        OwUnit::CustomDeg108,
        OwUnit::Deg120,
        OwUnit::Deg135,
        OwUnit::Ow68,
    ];

    /// If this `unit` is folded from paper with an aspect ratio of `paper_aspect`, what is the
    /// `(spine length, unit angle)` of this unit
    pub fn geometry(self, paper_aspect: f32) -> OwUnitGeometry {
        // Derived from folding patterns:
        // - https://owrigami.com/show_diagram.php?diagram=120
        // - https://owrigami.com/show_diagram.php?diagram=135
        const DEG_108_REDUCTION: f32 = 0.72654253; // 1.0 * tan(36 deg)
        const DEG_120_REDUCTION: f32 = 0.8660254; // 1.5 * tan(30 deg)
        const DEG_135_REDUCTION: f32 = 0.82842714; // 2.0 * tan(22.5 deg)

        // Reduction factor is a multiple of 1/4 of the paper's height
        let (reduction_factor, supported_angles) = match self {
            OwUnit::Deg60 => (0.0, vec![Deg(60.0)]),
            OwUnit::SturdyEdgeModule90 => (0.5, vec![Deg(90.0)]),
            OwUnit::CustomDeg90 => (1.0, vec![Deg(90.0)]),
            OwUnit::Custom3468 => (1.0, vec![Deg(60.0), Deg(90.0), Deg(120.0), Deg(135.0)]),
            OwUnit::CustomDeg108 => (DEG_108_REDUCTION, vec![Deg(108.0)]),
            OwUnit::Deg120 => (DEG_120_REDUCTION, vec![Deg(120.0)]),
            OwUnit::Deg135 => (DEG_135_REDUCTION, vec![Deg(135.0)]),
            OwUnit::Ow68 => (DEG_135_REDUCTION, vec![Deg(120.0), Deg(135.0)]),
        };
        let length_reduction = paper_aspect * 0.25 * reduction_factor;
        let spine_length = 1.0 - length_reduction * 2.0;
        OwUnitGeometry {
            paper_aspect,
            spine_length_factor: spine_length,
            supported_angles,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            OwUnit::Deg60 => "Ow's 60° unit",
            OwUnit::SturdyEdgeModule90 => "StEM (90°)",
            OwUnit::CustomDeg90 => "Custom 90° unit",
            OwUnit::Custom3468 => "60°, 90°, 120°, 135° unit",
            OwUnit::CustomDeg108 => "Custom 108° unit",
            OwUnit::Deg120 => "Ow's 120° unit",
            OwUnit::Deg135 => "Ow's 135° unit",
            OwUnit::Ow68 => "Ow's 120°/135° unit",
        }
    }
}

///////////////
// RENDERING //
///////////////

const WIREFRAME_TINT: f32 = 0.3;
const HIGHLIGHT_BRIGHTNESS: f32 = 0.8;
const INSIDE_TINT: f32 = 0.5;

const EDGE_RADIUS: f32 = 0.03;
const VERTEX_RADIUS: f32 = 0.05;

#[derive(Debug, Clone, PartialEq)]
pub enum FaceRenderStyle {
    /// Render model with solid faces
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
        non_flat: Option<NonFlat>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonFlat {
    pub supported_angles: Vec<Degrees>,
    pub push_direction: Side,
    pub add_crinkle: bool,
}

impl Model {
    pub fn face_mesh(&self) -> Option<CpuMesh> {
        let style = self.view_geometry_settings.face_render_style()?;
        Some(self.face_mesh_with_style(&style))
    }

    fn face_mesh_with_style(&self, style: &FaceRenderStyle) -> CpuMesh {
        // Create outward-facing faces
        let faces = self.faces_to_render(style);
        let (verts, colors, tri_indices) = Self::triangulate_mesh(faces);

        // Add verts colors for inside-facing verts
        let mut all_verts = verts.clone();
        all_verts.extend_from_within(..);
        let mut all_colors = colors.clone();
        for c in colors {
            all_colors.push(darken_color(c, INSIDE_TINT));
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

        let converted_colors = all_colors
            .into_iter()
            .map(egui_color_to_srgba)
            .collect_vec();
        let mut mesh = CpuMesh {
            positions: Positions::F32(all_verts),
            colors: Some(converted_colors),
            indices: Indices::U32(all_tri_indices),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh
    }

    fn faces_to_render(&self, style: &FaceRenderStyle) -> Vec<(Color32, Vec<Vec3>)> {
        let mut faces_to_render = Vec::new();
        for face in self.polyhedron.faces() {
            // Decide how to render them, according to the style
            match style {
                // For solid faces, render the face as a set of triangles
                FaceRenderStyle::Solid => {
                    let centroid = face.centroid(&self.polyhedron);
                    for (v1, v2) in face.verts().iter().circular_tuple_windows() {
                        let verts = vec![
                            centroid,
                            self.polyhedron.vert_pos(*v1),
                            self.polyhedron.vert_pos(*v2),
                        ];
                        let color = self.get_edge_side_color(*v2, *v1);
                        faces_to_render.push((color, verts));
                    }
                }
                // For ow-like faces, render up to two faces per edge
                FaceRenderStyle::OwLike {
                    side_ratio,
                    non_flat,
                } => {
                    self.owlike_faces(face, *side_ratio, non_flat.as_ref(), &mut faces_to_render);
                }
            }
        }
        faces_to_render
    }

    fn owlike_faces(
        &self,
        face: &Face,
        side_ratio: f32,
        fixed_angle: Option<&NonFlat>,
        faces_to_render: &mut Vec<(Color32, Vec<Vec3>)>,
    ) {
        // Geometry calculations
        let normal = face.normal(&self.polyhedron);
        let vert_positions = face.vert_positions(&self.polyhedron);
        let in_directions = self.vertex_in_directions(&vert_positions);

        let verts_and_ins = face
            .verts()
            .iter()
            .zip_eq(vert_positions)
            .zip_eq(&in_directions);
        for (((i0, v0), in0), ((i1, v1), in1)) in verts_and_ins.circular_tuple_windows() {
            // Extract useful data
            let edge_color = self.get_edge_side_color(*i1, *i0);
            let mut add_face = |verts: Vec<Vec3>| faces_to_render.push((edge_color, verts));

            match &fixed_angle {
                // If the unit has no fixed angle, then always make the units parallel
                // to the faces
                None => add_face(vec![v0, v1, v1 + in1 * side_ratio, v0 + in0 * side_ratio]),
                Some(angle) => {
                    let NonFlat {
                        supported_angles,
                        push_direction,
                        add_crinkle,
                    } = angle;
                    // Calculate the unit's inset angle, and therefore break the unit length down
                    // into x/y components
                    let inset_angle = unit_inset_angle(face.order(), &supported_angles);
                    let l_in = side_ratio * inset_angle.cos();
                    let l_up = side_ratio * inset_angle.sin();
                    // Pick a normal to use based on the push direction
                    let unit_up = match push_direction {
                        Side::Out => normal,
                        Side::In => -normal,
                    };
                    let up = unit_up * l_up;
                    // Add faces
                    if *add_crinkle {
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

    fn vertex_in_directions(&self, verts: &[Vec3]) -> Vec<Vec3> {
        let mut in_directions = Vec::new();
        for (v0, v1, v2) in verts.iter().circular_tuple_windows() {
            let in_vec = ((v0 - v1) + (v2 - v1)).normalize();
            // Normalize so that it has a projected length of 1 perpendicular to the edge
            let normalized = normalize_perpendicular_to(in_vec, v2 - v1);
            in_directions.push(normalized);
        }
        in_directions.rotate_right(1); // The 0th in-direction is actually from the 1st vertex
        in_directions
    }

    fn triangulate_mesh(faces: Vec<(Color32, Vec<Vec3>)>) -> (Vec<Vec3>, Vec<Color32>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut colors = Vec::new();
        let mut tri_indices = Vec::new();

        for (color, face_verts) in faces {
            // Add all vertices from this face.  We have to duplicate the vertices so that each
            // face gets flat shading
            let first_vert_idx = verts.len() as u32;
            verts.extend_from_slice(&face_verts);
            // Add colors for this face's vertices
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

    /* EDGES */

    pub fn edge_mesh(
        &self,
        context: &three_d::Context,
        edges_to_highlight: &HashSet<EdgeId>,
    ) -> InstancedMesh {
        let mut cylinder = CpuMesh::cylinder(10);
        cylinder
            .transform(&Mat4::from_nonuniform_scale(1.0, EDGE_RADIUS, EDGE_RADIUS))
            .unwrap();

        let edges = self.edge_instances(
            self.view_geometry_settings.wireframe_edges,
            edges_to_highlight,
        );
        InstancedMesh::new(context, &edges, &cylinder)
    }

    fn edge_instances(
        &self,
        show_wireframe: bool,
        edges_to_highlight: &HashSet<EdgeId>,
    ) -> Instances {
        let mut colors = Vec::new();
        let mut transformations = Vec::new();
        for edge in self.polyhedron.edges() {
            let is_highlighted = edges_to_highlight.contains(&edge.id());
            if !(show_wireframe || is_highlighted) {
                continue;
            }

            let edge_color = self.get_mixed_edge_color(edge.bottom_vert, edge.top_vert);
            let color = match is_highlighted {
                true => lerp_color(edge_color, Color32::WHITE, HIGHLIGHT_BRIGHTNESS),
                false => darken_color(edge_color, WIREFRAME_TINT),
            };
            colors.push(egui_color_to_srgba(color));
            transformations.push(edge_transform(
                self.polyhedron.vert_pos(edge.bottom_vert),
                self.polyhedron.vert_pos(edge.top_vert),
            ));
        }

        Instances {
            transformations,
            colors: Some(colors),
            ..Default::default()
        }
    }

    /* VERTICES */

    pub fn vertex_mesh(&self, context: &three_d::Context) -> InstancedMesh {
        let radius = match self.view_geometry_settings.wireframe_verts {
            true => VERTEX_RADIUS,
            false => EDGE_RADIUS,
        };
        let mut sphere = CpuMesh::sphere(8);
        sphere.transform(&Mat4::from_scale(radius)).unwrap();

        let verts = if self.view_geometry_settings.wireframe_edges
            || self.view_geometry_settings.wireframe_verts
        {
            self.vertex_instances()
        } else {
            Instances::default()
        };
        InstancedMesh::new(context, &verts, &sphere)
    }

    fn vertex_instances(&self) -> Instances {
        let verts = &self.polyhedron.verts();
        let color = egui_color_to_srgba(darken_color(self.default_color, WIREFRAME_TINT));
        Instances {
            transformations: verts
                .iter()
                .map(|v| Mat4::from_translation(v.pos()))
                .collect_vec(),
            colors: Some(vec![color; verts.len()]),
            ..Default::default()
        }
    }
}

fn unit_inset_angle(n: usize, supported_angles: &[Degrees]) -> Radians {
    // Determine which of the supported angles to take
    let interior_ngon_angle = Rad(PI - (PI * 2.0 / n as f32));
    let chosen_angle = *supported_angles
        .iter()
        .find_or_last(|&&a| Radians::from(a) >= interior_ngon_angle - Rad(0.001))
        .unwrap();
    let unit_angle = Radians::from(chosen_angle) / 2.0; // The whole interior angle is made up by two units

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

/////////////////////
// BUILT-IN MODELS //
/////////////////////

pub fn builtin_models() -> Vec<Model> {
    let platonic = [
        ("Tetrahedron", Polyhedron::tetrahedron()),
        ("Cube", Polyhedron::cube_poly()),
        ("Octahedron", Polyhedron::octahedron()),
        ("Dodecahedron", Polyhedron::dodecahedron()),
        ("Icosahedron", Polyhedron::icosahedron()),
    ];
    let archimedean_tetra = [("Truncated Tetrahedron", Polyhedron::truncated_tetrahedron())];
    let archimedean_octa = [
        ("Truncated Cube", Polyhedron::truncated_cube()),
        ("Truncated Octahedron", Polyhedron::truncated_octahedron()),
        ("Cuboctahedron", Polyhedron::cuboctahedron()),
        ("Snub Cube", Polyhedron::snub_cube()),
        ("Rhombicuboctahedron", Polyhedron::rhombicuboctahedron()),
        (
            "Great Rhombicuboctahedron",
            Polyhedron::great_rhombicuboctahedron(),
        ),
    ];
    let archimedean_dodeca = [
        (
            "Truncated Dodecahedron",
            Polyhedron::truncated_dodecahedron(),
        ),
        ("Truncated Icosahedron", Polyhedron::truncated_icosahedron()),
        ("Icosidodecahedron", Polyhedron::icosidodecahedron()),
        ("Snub Dodecahedron", Polyhedron::snub_dodecahedron()),
        (
            "Rhombicosidodecahedron",
            Polyhedron::rhombicosidodecahedron(),
        ),
        (
            "Great Rhombicosidodecahedron",
            Polyhedron::great_rhombicosidodecahedron(),
        ),
    ];

    let pyramidlikes = [
        ("3-Pyramid = Tetrahedron", Polyhedron::pyramid(3).poly),
        ("4-Pyramid", Polyhedron::pyramid(4).poly),
        ("5-Pyramid", Polyhedron::pyramid(5).poly),
        ("3-Cupola", Polyhedron::cupola(3).poly),
        ("4-Cupola", Polyhedron::cupola(4).poly),
        ("5-Cupola", Polyhedron::cupola(5).poly),
        ("Rotunda", Polyhedron::rotunda().poly),
    ];
    let prismlikes = [
        ("3-Prism", Polyhedron::prism(3).poly),
        ("4-Prism = Cube", Polyhedron::prism(4).poly),
        ("5-Prism", Polyhedron::prism(5).poly),
        ("6-Prism", Polyhedron::prism(6).poly),
        ("7-Prism", Polyhedron::prism(7).poly),
        ("8-Prism", Polyhedron::prism(8).poly),
        ("3-Antiprism = Octahedron", Polyhedron::antiprism(3).poly),
        ("4-Antiprism", Polyhedron::antiprism(4).poly),
        ("5-Antiprism", Polyhedron::antiprism(5).poly),
        ("6-Antiprism", Polyhedron::antiprism(6).poly),
        ("7-Antiprism", Polyhedron::antiprism(7).poly),
        ("8-Antiprism", Polyhedron::antiprism(8).poly),
    ];

    let groups = [
        (r"Platonic", platonic.to_vec()),
        (r"Archimedean", archimedean_tetra.to_vec()),
        (r"Archimedean\Octahedral", archimedean_octa.to_vec()),
        (r"Archimedean\Dodecahedral", archimedean_dodeca.to_vec()),
        (r"Pyramids and Cupolae", pyramidlikes.to_vec()),
        (r"Prisms and Antiprisms", prismlikes.to_vec()),
    ];

    let mut all_models = Vec::new();
    for (group_name, models) in groups {
        for (model_name, poly) in models {
            all_models.push(Model::new(full_builtin_name(group_name, model_name), poly));
        }
    }
    for mut model in toroids() {
        model.full_name = full_builtin_name("Toroids", &model.full_name);
        all_models.push(model);
    }
    for mut model in misc_models() {
        model.full_name = full_builtin_name("Misc", &model.full_name);
        all_models.push(model);
    }
    all_models
}

fn toroids() -> Vec<Model> {
    let qpq_slash_p = |gyro: bool| -> Polyhedron {
        let PrismLike {
            mut poly,
            bottom_face,
            top_face,
        } = Polyhedron::prism(6);
        let face_to_excavate = poly.get_ngon(4);
        poly.extend_cupola(bottom_face, false);
        poly.extend_cupola(top_face, gyro);
        let tunnel = Polyhedron::prism(6).poly;
        poly.excavate(face_to_excavate, &tunnel, tunnel.get_ngon(4), 1);
        poly
    };

    let toroids = [
        Model::new("Flying saucer".to_owned(), {
            let PrismLike {
                mut poly,
                bottom_face,
                top_face,
            } = Polyhedron::cupola(5);
            poly.extend_cupola(bottom_face, true);
            poly.transform_verts(|mut v| {
                v.y = f32::round(v.y * 2.0) / 2.0;
                v
            });
            poly.excavate_prism(top_face);
            poly
        }),
        Model::new("Cake pan".to_owned(), {
            let PrismLike {
                mut poly,
                bottom_face,
                top_face: _,
            } = Polyhedron::cupola(3);
            let next = poly.extend_prism(bottom_face);
            let next = poly.excavate_cupola(next, true);
            poly.excavate_prism(next);
            poly
        }),
        Model::new("Cakier pan".to_owned(), {
            let PrismLike {
                mut poly,
                bottom_face,
                top_face: _,
            } = Polyhedron::cupola(4);
            let next = poly.extend_prism(bottom_face);
            let next = poly.excavate_cupola(next, true);
            poly.excavate_prism(next);
            poly
        }),
        Model::new("Cakiest pan".to_owned(), {
            let PrismLike {
                mut poly,
                bottom_face,
                top_face: _,
            } = Polyhedron::cupola(5);
            let next = poly.extend_prism(bottom_face);
            let next = poly.excavate_cupola(next, true);
            poly.excavate_prism(next);
            poly
        }),
        Model::new("Torturous Tunnel".to_owned(), {
            let PrismLike {
                mut poly,
                bottom_face,
                top_face,
            } = Polyhedron::cupola(3);
            let bottom_face = poly.extend_cupola(bottom_face, true);
            poly.excavate_antiprism(bottom_face);
            poly.excavate_antiprism(top_face);
            poly
        }),
        Model::new("Oriental Hat".to_owned(), {
            let PrismLike {
                mut poly,
                bottom_face,
                top_face: _,
            } = Polyhedron::rotunda();
            let bottom_face = poly.excavate_cupola(bottom_face, false);
            poly.excavate_antiprism(bottom_face);
            poly
        }),
        {
            let mut poly = Polyhedron::truncated_cube();
            let face = poly.get_face_with_normal(Vec3::unit_x());
            poly.color_faces_added_by("Tunnel", |poly| {
                let next = poly.excavate_cupola(face, true);
                let next = poly.excavate_prism(next);
                poly.excavate_cupola(next, false);
            });
            Model::new("Bob".to_owned(), poly)
        },
        {
            let mut poly = Polyhedron::truncated_cube();
            let face = poly.get_face_with_normal(Vec3::unit_x());
            poly.color_faces_added_by("Tunnel", |poly| {
                let next = poly.excavate_cupola(face, false);
                let next = poly.excavate_prism(next);
                poly.excavate_cupola(next, false);
            });

            Model::new("Gyrated Bob".to_owned(), poly)
        },
        {
            let mut poly = Polyhedron::truncated_cube();
            // Extend +x and -x faces with cupolae
            let face = poly.get_face_with_normal(Vec3::unit_x());
            let back_face = poly.get_face_with_normal(-Vec3::unit_x());
            poly.extend_cupola(back_face, true);
            let next = poly.extend_cupola(face, true);
            // Tunnel with B_4 (P_4) B_4
            poly.color_faces_added_by("Tunnel", |poly| {
                let next = poly.excavate_cuboctahedron(next);
                let next = poly.excavate_prism(next);
                poly.excavate_cuboctahedron(next);
            });

            Model::new("Dumbell".to_owned(), poly)
        },
        Model::new("Q_3 P_6 Q_3 / P_6".to_owned(), qpq_slash_p(false)),
        Model::new("Q_3 P_6 gQ_3 / P_6".to_owned(), qpq_slash_p(true)),
        Model::new("Q_4^2 / B_4".to_owned(), {
            let PrismLike {
                mut poly,
                bottom_face,
                top_face,
            } = Polyhedron::cupola(4);
            poly.extend_cupola(bottom_face, true);
            poly.excavate_cuboctahedron(top_face);
            poly
        }),
        {
            let mut poly = Polyhedron::truncated_octahedron();
            // Excavate cupolas (TODO: Do this by symmetry)
            let mut inner_face = FaceIdx::new(0);
            poly.color_faces_added_by("Tunnels", |poly| {
                for face_idx in [0, 2, 4, 6] {
                    inner_face = poly.excavate_cupola(FaceIdx::new(face_idx), true);
                }
            });
            // Excavate central octahedron
            poly.color_faces_added_by("Centre", |poly| {
                poly.excavate_antiprism(inner_face);
            });
            Model::new("K_3 / 3Q_3 (S_3)".to_owned(), poly)
        },
        Model::new("K_4 (tunnel octagons)".to_owned(), {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut inner_face = FaceIdx::new(0);
            for octagon in poly.ngons(8) {
                inner_face = poly.excavate_cupola(octagon, false);
            }
            let inner = Polyhedron::rhombicuboctahedron();
            poly.color_faces_added_by("Inner", |poly| {
                poly.excavate(inner_face, &inner, inner.get_ngon(4), 0);
            });
            poly
        }),
        Model::new("K_4 (tunnel hexagons)".to_owned(), {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut inner_face = FaceIdx::new(0);
            for hexagon in poly.ngons(6) {
                inner_face = poly.excavate_cupola(hexagon, true);
            }
            let inner = Polyhedron::rhombicuboctahedron();
            poly.color_faces_added_by("Inner", |poly| {
                poly.excavate(inner_face, &inner, inner.get_ngon(3), 0);
            });
            poly
        }),
        Model::new("K_4 (tunnel cubes)".to_owned(), {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut inner_face = FaceIdx::new(0);
            for square in poly.ngons(4) {
                inner_face = poly.excavate_prism(square);
            }
            let inner = Polyhedron::rhombicuboctahedron();
            let face = *inner.ngons(4).last().unwrap();
            poly.color_faces_added_by("Inner", |poly| {
                poly.excavate(inner_face, &inner, face, 0);
            });
            poly
        }),
        {
            let mut poly = Polyhedron::great_rhombicosidodecahedron();
            for face_idx in poly.face_indices() {
                if poly.face_order(face_idx) != 10 {
                    poly.color_face(face_idx, "Outer");
                }
            }
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, true);
                inner_face = poly.excavate_antiprism(next);
            }
            let mut inner = Polyhedron::rhombicosidodecahedron();
            for face_idx in inner.face_indices() {
                if inner.face_order(face_idx) != 5 {
                    inner.color_face(face_idx, "Inner");
                }
            }
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
            Model::new("K_5 (cupola/antiprism)".to_owned(), poly)
        },
        Model::new("K_5 (rotunda)".to_owned(), {
            let mut poly = Polyhedron::great_rhombicosidodecahedron();
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                inner_face = poly.excavate_rotunda(decagon, true);
            }
            let inner = Polyhedron::rhombicosidodecahedron();
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
            poly
        }),
        Model::new("Stephanie".to_owned(), {
            // Start with a colored truncated dodecahedron
            let mut poly = Polyhedron::truncated_dodecahedron();
            // Excavate using cupolae and antiprisms to form the tunnels
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, false);
                inner_face = poly.excavate_antiprism(next);
            }
            // Excavate the central cavity, and color these edges
            let inner = Polyhedron::dodecahedron();
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
            poly
        }),
        {
            // Start with a colored truncated dodecahedron
            let mut poly = Polyhedron::truncated_dodecahedron();
            poly.color_all_edges("Outer");
            // Excavate using cupolae and antiprisms to form the tunnels
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, false);
                inner_face = poly.excavate_antiprism(next);
            }
            // Excavate the central cavity, and color these edges
            let mut inner = Polyhedron::dodecahedron();
            inner.color_all_edges("Inner");
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);

            Model::new("Stephanie (Coloring A)".to_owned(), poly)
        },
        {
            // Start with a colored truncated dodecahedron
            let mut poly = Polyhedron::truncated_dodecahedron();
            for tri in poly.ngons(3) {
                poly.color_face(tri, "Outer");
            }
            // Excavate using cupolae and antiprisms to form the tunnels
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, false);
                inner_face = poly.excavate_antiprism(next);
            }
            // Excavate the central cavity, and color these edges
            let inner = Polyhedron::dodecahedron();
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);

            Model::new("Stephanie (Coloring B)".to_owned(), poly)
        },
        {
            let mut poly = Polyhedron::truncated_icosahedron();
            for face in poly.ngons(5) {
                poly.color_face(face, "Pentagons");
            }

            Model::new("Football".to_owned(), poly)
        },
        {
            // Create a bicupola
            let PrismLike {
                mut poly,
                bottom_face,
                top_face,
            } = Polyhedron::cupola(3);
            let bottom_face = poly.extend_cupola(bottom_face, true);
            let faces_to_add_pyramids = poly
                .faces_enumerated()
                .map(|(idx, _face)| idx)
                .collect_vec();
            // Dig tunnel, and color it blue
            poly.color_edges_added_by("Tunnel", |poly| {
                poly.excavate_antiprism(bottom_face);
                poly.excavate_antiprism(top_face);
            });
            // Add pyramids to all faces in the bicupola which still exist
            for face in faces_to_add_pyramids {
                if poly.is_face(face) {
                    poly.extend_pyramid(face);
                }
            }

            Model::new("Apanar Deltahedron".to_owned(), poly)
        },
        Model::new(
            "Christopher".to_owned(),
            prism_extended_cuboctahedron("Triangles", "Squares"),
        ),
        Model::new("Cube Box (Color A)".to_owned(), cube_box_col_a(false)),
        Model::new("Cube Box (Color B)".to_owned(), cube_box_col_a(true)),
    ];

    toroids.to_vec()
}

fn misc_models() -> Vec<Model> {
    let models = [
        {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut color_face = |normal: Vec3, color: &str| {
                let face_idx = poly.get_face_with_normal(normal);
                for (v1, v2) in poly
                    .get_face(face_idx)
                    .verts()
                    .to_vec()
                    .into_iter()
                    .circular_tuple_windows()
                {
                    poly.set_full_edge_color(EdgeId::new(v1, v2), color);
                }
            };
            color_face(Vec3::unit_x(), "X");
            color_face(-Vec3::unit_x(), "X");
            color_face(Vec3::unit_y(), "Y");
            color_face(-Vec3::unit_y(), "Y");
            color_face(Vec3::unit_z(), "Z");
            color_face(-Vec3::unit_z(), "Z");
            let mut model = Model::new("XYZ Great Rhobicuboctahedron (A)".to_owned(), poly);
            model.view_geometry_settings.paper_ratio_w = 1;
            model.view_geometry_settings.paper_ratio_h = 1;
            model.view_geometry_settings.unit = OwUnit::Ow68;
            model.default_color = Color32::from_rgb(54, 54, 54);
            model
        },
        {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut color_face = |normal: Vec3, color: &str| {
                let face_idx = poly.get_face_with_normal(normal);
                poly.color_face(face_idx, color);
            };
            color_face(Vec3::unit_x(), "X");
            color_face(-Vec3::unit_x(), "X");
            color_face(Vec3::unit_y(), "Y");
            color_face(-Vec3::unit_y(), "Y");
            color_face(Vec3::unit_z(), "Z");
            color_face(-Vec3::unit_z(), "Z");
            let mut model = Model::new("XYZ Great Rhobicuboctahedron (B)".to_owned(), poly);
            model.view_geometry_settings.paper_ratio_w = 1;
            model.view_geometry_settings.paper_ratio_h = 1;
            model.view_geometry_settings.unit = OwUnit::Ow68;
            model.default_color = Color32::from_rgb(54, 54, 54);
            model
        },
    ];
    models.to_vec()
}

fn prism_extended_cuboctahedron(tri_color: &str, square_color: &str) -> Polyhedron {
    // Make the pyramids out of which the cuboctahedron's faces will be constructed
    let make_pyramid = |n: usize, color_name: &str| -> (Polyhedron, Vec<FaceIdx>) {
        let Pyramid {
            mut poly,
            base_face,
        } = Polyhedron::pyramid(n);
        // Color the base face
        let verts = &poly.get_face(base_face).verts().to_vec();
        for (v1, v2) in verts.iter().copied().circular_tuple_windows() {
            poly.set_half_edge_color(v2, v1, color_name);
        }
        // Get the side faces
        let side_faces = poly
            .faces_enumerated()
            .map(|(id, _)| id)
            .filter(|id| *id != base_face)
            .collect_vec();
        (poly, side_faces)
    };
    let (tri_pyramid, tri_pyramid_side_faces) = make_pyramid(3, tri_color);
    let (quad_pyramid, quad_pyramid_side_faces) = make_pyramid(4, square_color);

    // Recursively build the polyhedron
    let mut poly = quad_pyramid.clone();
    let mut faces_to_expand = quad_pyramid_side_faces
        .iter()
        .map(|idx| (*idx, 3))
        .collect_vec();
    while let Some((face_to_extend, order_of_new_pyramid)) = faces_to_expand.pop() {
        // Get which pyramid we're going to add
        let (pyramid, side_faces, next_pyramid_order) = match order_of_new_pyramid {
            3 => (&tri_pyramid, &tri_pyramid_side_faces, 4),
            4 => (&quad_pyramid, &quad_pyramid_side_faces, 3),
            _ => unreachable!(),
        };

        // Add prism and pyramid
        if !poly.is_face(face_to_extend) {
            continue; // Face has already been connected to something
        }
        let opposite_face = poly.extend_prism(face_to_extend);
        if !poly.is_face(opposite_face) {
            continue; // Connecting to something which already exists
        }
        let face_mapping = poly.extend(opposite_face, pyramid, side_faces[0], 0);

        // Add new faces to extend
        for &source_side_face in side_faces {
            faces_to_expand.push((face_mapping[source_side_face], next_pyramid_order));
        }
    }

    // Centre the polyhedron and return it
    poly.make_centred();
    poly
}

fn cube_box_col_a(use_concave_color: bool) -> Polyhedron {
    // Start with a central cube, which we'll use for the bottom-back-left corner
    let Cube {
        poly: cube,
        left,
        right,
        top,
        bottom,
        front,
        back,
    } = Polyhedron::cube();

    // Original cube becomes bottom-back-left central
    let mut poly = cube.clone();
    macro_rules! extend_colored {
        ($face: expr) => {{
            extend_prism_with_axis_color(&mut poly, $face)
        }};
    }
    extend_colored!(left);
    extend_colored!(back);
    extend_colored!(bottom);
    // Bottom-back-right
    let new_right = extend_colored!(right);
    let bbr_face_map = poly.extend(new_right, &cube, left, 0);
    extend_colored!(bbr_face_map[back]);
    extend_colored!(bbr_face_map[right]);
    extend_colored!(bbr_face_map[bottom]);
    // Top-back-left
    let new_top = extend_colored!(top);
    let tbl_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tbl_face_map[back]);
    extend_colored!(tbl_face_map[left]);
    extend_colored!(tbl_face_map[top]);
    // Bottom-front-left
    let new_front = extend_colored!(front);
    let bfl_face_map = poly.extend(new_front, &cube, back, 0);
    extend_colored!(bfl_face_map[front]);
    extend_colored!(bfl_face_map[left]);
    extend_colored!(bfl_face_map[bottom]);
    // Top-back-right
    let new_top = extend_colored!(bbr_face_map[top]);
    let tbr_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tbr_face_map[left]); // Links to top-back-left
    extend_colored!(tbr_face_map[back]);
    extend_colored!(tbr_face_map[right]);
    extend_colored!(tbr_face_map[top]);
    // Top-front-left
    let new_top = extend_colored!(bfl_face_map[top]);
    let tfl_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tfl_face_map[back]); // Links to top-back-left
    extend_colored!(tfl_face_map[front]);
    extend_colored!(tfl_face_map[left]);
    extend_colored!(tfl_face_map[top]);
    // Bottom-front-right
    let new_right = extend_colored!(bfl_face_map[right]);
    let bfr_face_map = poly.extend(new_right, &cube, left, 0);
    extend_colored!(bfr_face_map[back]); // Links to bottom-back-right
    extend_colored!(bfr_face_map[front]);
    extend_colored!(bfr_face_map[right]);
    extend_colored!(bfr_face_map[bottom]);
    // Top-front-right
    let new_top = extend_colored!(bfr_face_map[top]);
    let tfr_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tfr_face_map[back]); // Links to top-back-right
    extend_colored!(tfr_face_map[left]); // Links to top-front-left
    extend_colored!(tfr_face_map[front]);
    extend_colored!(tfr_face_map[right]);
    extend_colored!(tfr_face_map[top]);

    // Color concave edges black
    if use_concave_color {
        for e in poly.edges() {
            if e.is_concave() {
                poly.reset_full_edge_color(e.top_vert, e.bottom_vert);
            }
        }
    }

    poly.make_centred();
    poly
}

fn extend_prism_with_axis_color(poly: &mut Polyhedron, face: FaceIdx) -> FaceIdx {
    // Determine which axis the face is in
    let normal = poly.get_face(face).normal(poly);
    let color = if normal.dot(Vec3::unit_x()).abs() > 0.99999 {
        "X"
    } else if normal.dot(Vec3::unit_y()).abs() > 0.99999 {
        "Y"
    } else {
        assert!(normal.dot(Vec3::unit_z()).abs() > 0.99999);
        // Implicitly must be z
        "Z"
    };

    poly.color_faces_added_by(color, |poly| poly.extend_prism(face))
}

fn full_builtin_name(group_name: &str, model_name: &str) -> String {
    format!("Built-in\\{}\\{}", group_name, model_name)
}
