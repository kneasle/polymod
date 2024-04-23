use std::{
    collections::HashSet,
    f32::consts::PI,
    sync::atomic::{AtomicU32, Ordering},
};

use indexmap::IndexMap;
use itertools::Itertools;
use three_d::{
    egui::{self, Color32},
    vec3, Angle, CpuMesh, Deg, Degrees, Indices, InnerSpace, InstancedMesh, Instances, Mat4,
    Positions, Quat, Rad, Radians, Vec3,
};

use crate::{
    polyhedron::{EdgeId, VertIdx},
    utils::{angle_in_spherical_triangle, darken_color, egui_color_to_srgba, lerp_color},
};
use crate::{
    polyhedron::{Face, Polyhedron},
    utils::{normalize_perpendicular_to, Side},
    SMALL_SPACE,
};

#[derive(Debug, Clone)]
pub struct Model {
    id: ModelId,
    name: String,

    polyhedron: Polyhedron,

    // Display settings & Coloring
    view_geometry_settings: ViewGeomSettings,
    default_color: Color32,
    colors: ColorMap,
}

pub type ColorMap = IndexMap<String, Color32>;

impl Model {
    pub fn new(name: &str, poly: Polyhedron) -> Self {
        Self::with_colors(name, poly, ColorMap::new())
    }

    pub fn with_colors(name: &str, poly: Polyhedron, colors: ColorMap) -> Self {
        Self {
            id: ModelId::next_unique(),

            name: name.to_owned(),
            polyhedron: poly,

            view_geometry_settings: ViewGeomSettings::default(),
            default_color: Color32::GRAY,
            colors,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }

    pub fn polyhedron(&self) -> &Polyhedron {
        &self.polyhedron
    }

    pub fn id(&self) -> ModelId {
        self.id
    }

    pub fn set_id(&mut self, id: ModelId) {
        self.id = id;
    }

    pub fn view_geometry_settings(&self) -> &ViewGeomSettings {
        &self.view_geometry_settings
    }

    pub fn draw_view_geom_gui(&mut self, ui: &mut egui::Ui) {
        self.view_geometry_settings.gui(ui);
    }

    pub fn draw_colors_gui(&mut self, ui: &mut egui::Ui) {
        let mut draw_color = |col: &mut Color32, label: &str| {
            ui.horizontal(|ui| {
                ui.color_edit_button_srgba(col);
                ui.label(label);
            })
        };

        // Draw all colours, including the default one
        draw_color(&mut self.default_color, "(Default)");
        for (name, color) in &mut self.colors {
            draw_color(color, name);
        }
    }

    /* HELPERS */

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwUnit {
    Deg60,
    SturdyEdgeModule90,
    CustomDeg90,
    CustomDeg108,
    Deg120,
    Deg135,
}

impl Default for ViewGeomSettings {
    fn default() -> Self {
        ViewGeomSettings {
            style: FaceGeomStyle::Solid,
            side_ratio: 0.25,

            paper_ratio_w: 3,
            paper_ratio_h: 2,
            unit: OwUnit::CustomDeg90,
            direction: Side::In,
            add_crinkle: false,

            wireframe_edges: true,
            wireframe_verts: true,
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
                fixed_angle: None,
            },
            FaceGeomStyle::OwLikeAngled => {
                // Model the geometry of the unit
                let OwUnitGeometry {
                    paper_aspect,
                    spine_length_factor: spine_length,
                    angle,
                } = self.ow_unit_geometry().unwrap();
                let unit_width = paper_aspect / 4.0;
                // Construct unit info
                FaceRenderStyle::OwLike {
                    side_ratio: unit_width / spine_length,
                    fixed_angle: Some(FixedAngle {
                        angle,
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

#[derive(Debug, Clone, Copy)]
pub struct OwUnitGeometry {
    pub paper_aspect: f32,
    pub spine_length_factor: f32, // As a factor of paper width
    pub angle: Degrees,
}

impl OwUnit {
    const ALL: [Self; 6] = [
        OwUnit::Deg60,
        OwUnit::SturdyEdgeModule90,
        OwUnit::CustomDeg90,
        OwUnit::CustomDeg108,
        OwUnit::Deg120,
        OwUnit::Deg135,
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
        let (reduction_factor, full_angle) = match self {
            OwUnit::Deg60 => (0.0, Deg(60.0)),
            OwUnit::SturdyEdgeModule90 => (0.5, Deg(90.0)),
            OwUnit::CustomDeg90 => (1.0, Deg(90.0)),
            OwUnit::CustomDeg108 => (DEG_108_REDUCTION, Deg(108.0)),
            OwUnit::Deg120 => (DEG_120_REDUCTION, Deg(120.0)),
            OwUnit::Deg135 => (DEG_135_REDUCTION, Deg(135.0)),
        };
        let length_reduction = paper_aspect * 0.25 * reduction_factor;
        let spine_length = 1.0 - length_reduction * 2.0;
        OwUnitGeometry {
            paper_aspect,
            spine_length_factor: spine_length,
            angle: full_angle / 2.0,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            OwUnit::Deg60 => "Ow's 60° unit",
            OwUnit::SturdyEdgeModule90 => "StEM (90°)",
            OwUnit::CustomDeg90 => "Custom 90° unit",
            OwUnit::CustomDeg108 => "Custom 108° unit",
            OwUnit::Deg120 => "Ow's 120° unit",
            OwUnit::Deg135 => "Ow's 135° unit",
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

#[derive(Debug, Clone, Copy, PartialEq)]
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
        fixed_angle: Option<FixedAngle>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedAngle {
    pub angle: Degrees,
    pub push_direction: Side,
    pub add_crinkle: bool,
}

impl Model {
    pub fn face_mesh(&self) -> Option<CpuMesh> {
        let style = self.view_geometry_settings.face_render_style()?;
        Some(self.face_mesh_with_style(style))
    }

    fn face_mesh_with_style(&self, style: FaceRenderStyle) -> CpuMesh {
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

    fn faces_to_render(&self, style: FaceRenderStyle) -> Vec<(Color32, Vec<Vec3>)> {
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
                    fixed_angle,
                } => {
                    self.owlike_faces(face, side_ratio, fixed_angle, &mut faces_to_render);
                }
            }
        }
        faces_to_render
    }

    fn owlike_faces(
        &self,
        face: &Face,
        side_ratio: f32,
        fixed_angle: Option<FixedAngle>,
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

            match fixed_angle {
                // If the unit has no fixed angle, then always make the units parallel
                // to the faces
                None => add_face(vec![v0, v1, v1 + in1 * side_ratio, v0 + in0 * side_ratio]),
                Some(angle) => {
                    let FixedAngle {
                        angle: unit_angle,
                        push_direction,
                        add_crinkle,
                    } = angle;
                    // Calculate the unit's inset angle, and therefore break the unit length down
                    // into x/y components
                    let inset_angle = unit_inset_angle(face.order(), unit_angle.into());
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
        let verts = &self.polyhedron.vert_positions();
        let color = egui_color_to_srgba(darken_color(self.default_color, WIREFRAME_TINT));
        Instances {
            transformations: verts
                .iter()
                .cloned()
                .map(Mat4::from_translation)
                .collect_vec(),
            colors: Some(vec![color; verts.len()]),
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

/// An identifier for a specific model
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelId(pub u32);

static NEXT_ID: AtomicU32 = AtomicU32::new(0);

impl ModelId {
    pub fn next_unique() -> Self {
        let next_int = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Self(next_int)
    }
}
