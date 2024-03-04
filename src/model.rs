use std::{collections::HashMap, f32::consts::PI};

use itertools::Itertools;
use three_d::{
    egui, vec3, Angle, CpuMesh, Deg, Degrees, Indices, InnerSpace, InstancedMesh, Instances, Mat4,
    Positions, Quat, Rad, Radians, Srgba, Vec3,
};

use crate::{
    polyhedron::{Face, Polyhedron, VertIdx},
    utils::{
        angle_in_spherical_triangle, darken_color, lerp_color, normalize_perpendicular_to, Side,
    },
    SMALL_SPACE,
};

#[derive(Debug)]
pub struct Model {
    pub name: String,
    pub poly: Polyhedron,

    // Indexed colors for each side of each edge.  If `edge_colors[(a, b)] = idx` then the side
    // of the edge who's top-right vertex is `a` and who's bottom-right vertex is `b` will be
    // given the color at `color_index[idx]`.
    edge_colors: HashMap<(VertIdx, VertIdx), ColIdx>,
    color_index: ColVec<Srgba>,

    // Display settings
    view: ModelViewSettings,
}

impl Model {
    pub fn new(name: &str, poly: Polyhedron) -> Self {
        Self {
            name: name.to_owned(),
            poly,

            edge_colors: HashMap::default(),
            color_index: ColVec::default(),
            view: ModelViewSettings::default(),
        }
    }

    pub fn edge_side_color(&self, a: VertIdx, b: VertIdx) -> Srgba {
        match self.edge_colors.get(&(a, b)) {
            Some(idx) => self.color_index[*idx],
            None => DEFAULT_COLOR,
        }
    }

    pub fn mixed_edge_color(&self, a: VertIdx, b: VertIdx) -> Srgba {
        let left_color = self.edge_side_color(a, b);
        let right_color = self.edge_side_color(b, a);
        lerp_color(left_color, right_color, 0.5)
    }

    pub fn draw_view_gui(&mut self, ui: &mut egui::Ui) {
        self.view.gui(ui);
    }
}

/////////////
// STYLING //
/////////////

#[derive(Debug, Clone)]
pub struct ModelViewSettings {
    pub style: ModelViewStyle,
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
pub enum ModelViewStyle {
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
    Deg120,
    Deg135,
}

impl Default for ModelViewSettings {
    fn default() -> Self {
        ModelViewSettings {
            style: ModelViewStyle::Solid,
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

impl ModelViewSettings {
    pub fn face_render_style(&self) -> Option<FaceRenderStyle> {
        Some(match self.style {
            ModelViewStyle::None => return None,
            ModelViewStyle::Solid => FaceRenderStyle::Solid,
            ModelViewStyle::OwLikeFlat => FaceRenderStyle::OwLike {
                side_ratio: self.side_ratio,
                fixed_angle: None,
            },
            ModelViewStyle::OwLikeAngled => {
                // Model the geometry of the unit
                let paper_aspect = self.paper_ratio_h as f32 / self.paper_ratio_w as f32;
                let (length_reduction, unit_angle) = self.unit.geometry(paper_aspect);
                let unit_spine_length = 1.0 - length_reduction * 2.0;
                let unit_width = paper_aspect / 4.0;
                // Construct unit info
                FaceRenderStyle::OwLike {
                    side_ratio: unit_width / unit_spine_length,
                    fixed_angle: Some(FixedAngle {
                        unit_angle,
                        push_direction: self.direction,
                        add_crinkle: self.add_crinkle,
                    }),
                }
            }
        })
    }

    pub fn gui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Faces");
        ui.radio_value(&mut self.style, ModelViewStyle::None, "None");
        ui.radio_value(&mut self.style, ModelViewStyle::Solid, "Solid");
        ui.radio_value(
            &mut self.style,
            ModelViewStyle::OwLikeFlat,
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
            ModelViewStyle::OwLikeAngled,
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

impl OwUnit {
    const ALL: [Self; 5] = [
        OwUnit::Deg60,
        OwUnit::SturdyEdgeModule90,
        OwUnit::CustomDeg90,
        OwUnit::Deg120,
        OwUnit::Deg135,
    ];

    /// If this `unit` is folded from paper with an aspect ratio of `paper_aspect`, what is the
    /// `(width reduction, unit angle)` of this unit
    ///
    /// Note: the length perpendicular to the model's edge is `paper_aspect` times that of the length
    /// parallel to the model's edge.
    pub fn geometry(self, paper_aspect: f32) -> (f32, Degrees) {
        // Derived from folding patterns:
        // - https://owrigami.com/show_diagram.php?diagram=120
        // - https://owrigami.com/show_diagram.php?diagram=135
        const DEG_120_REDUCTION: f32 = 0.8660254; // 1.5 * tan(30 deg)
        const DEG_135_REDUCTION: f32 = 0.82842714; // 2 * tan(22.5 deg)

        // Reduction factor is a multiple of 1/4 of the paper's height
        let (reduction_factor, angle) = match self {
            OwUnit::Deg60 => (0.0, Deg(60.0)),
            OwUnit::SturdyEdgeModule90 => (0.5, Deg(90.0)),
            OwUnit::CustomDeg90 => (1.0, Deg(90.0)),
            OwUnit::Deg120 => (DEG_120_REDUCTION, Deg(120.0)),
            OwUnit::Deg135 => (DEG_135_REDUCTION, Deg(135.0)),
        };
        (paper_aspect * 0.25 * reduction_factor, angle / 2.0)
    }

    fn name(&self) -> &'static str {
        match self {
            OwUnit::Deg60 => "Ow's 60° unit",
            OwUnit::SturdyEdgeModule90 => "StEM (90°)",
            OwUnit::CustomDeg90 => "Custom 90° unit",
            OwUnit::Deg120 => "Ow's 120° unit",
            OwUnit::Deg135 => "Ow's 135° unit",
        }
    }
}

///////////////
// RENDERING //
///////////////

pub const DEFAULT_COLOR: Srgba = Srgba::new_opaque(100, 100, 100);
const WIREFRAME_TINT: f32 = 0.3;
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
    pub unit_angle: Degrees,
    pub push_direction: Side,
    pub add_crinkle: bool,
}

impl Model {
    pub fn face_mesh(&self) -> Option<CpuMesh> {
        let style = self.view.face_render_style()?;
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
        for face in self.poly.faces() {
            // Decide how to render them, according to the style
            match style {
                // For solid faces, just render the face as-is
                FaceRenderStyle::Solid => {
                    let verts = face.vert_positions(&self.poly);
                    faces_to_render.push((DEFAULT_COLOR, verts));
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
        faces_to_render: &mut Vec<(Srgba, Vec<Vec3>)>,
    ) {
        // Geometry calculations
        let normal = face.normal(&self.poly);
        let vert_positions = face.vert_positions(&self.poly);
        let in_directions = self.vertex_in_directions(&vert_positions);

        let verts_and_ins = face
            .verts()
            .iter()
            .zip_eq(vert_positions)
            .zip_eq(&in_directions);
        for (((i0, v0), in0), ((i1, v1), in1)) in verts_and_ins.circular_tuple_windows() {
            // Extract useful data
            let edge_color = self.edge_side_color(*i0, *i1);
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

    /* EDGES */

    pub fn edge_mesh(&self, context: &three_d::Context) -> InstancedMesh {
        let mut cylinder = CpuMesh::cylinder(10);
        cylinder
            .transform(&Mat4::from_nonuniform_scale(1.0, EDGE_RADIUS, EDGE_RADIUS))
            .unwrap();

        let edges = if self.view.wireframe_edges {
            self.edge_instances()
        } else {
            Instances::default()
        };
        InstancedMesh::new(context, &edges, &cylinder)
    }

    fn edge_instances(&self) -> Instances {
        let mut colors = Vec::new();
        let mut transformations = Vec::new();
        for edge in self.poly.edges() {
            colors.push(darken_color(
                self.mixed_edge_color(edge.bottom_vert, edge.top_vert),
                WIREFRAME_TINT,
            ));
            transformations.push(edge_transform(
                self.poly.vert_pos(edge.bottom_vert),
                self.poly.vert_pos(edge.top_vert),
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
        let radius = match self.view.wireframe_verts {
            true => VERTEX_RADIUS,
            false => EDGE_RADIUS,
        };
        let mut sphere = CpuMesh::sphere(8);
        sphere.transform(&Mat4::from_scale(radius)).unwrap();

        let verts = if self.view.wireframe_edges || self.view.wireframe_verts {
            self.vertex_instances()
        } else {
            Instances::default()
        };
        InstancedMesh::new(context, &verts, &sphere)
    }

    fn vertex_instances(&self) -> Instances {
        let verts = &self.poly.vert_positions();
        Instances {
            transformations: verts
                .iter()
                .cloned()
                .map(Mat4::from_translation)
                .collect_vec(),
            colors: Some(vec![
                darken_color(DEFAULT_COLOR, WIREFRAME_TINT);
                verts.len()
            ]),
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

index_vec::define_index_type! { pub struct ColIdx = usize; }
pub type ColVec<T> = index_vec::IndexVec<ColIdx, T>;
