use std::collections::BTreeMap;

use itertools::Itertools;
use ordered_float::OrderedFloat;
use polyhedron::{FaceRenderStyle, Polyhedron, PrismLike, RenderStyle};
use three_d::*;
use utils::Side;

use crate::{
    polyhedron::{ClosedEdgeData, Edge},
    utils::ngon_name,
};

mod model_view;
mod polyhedron;
mod utils;

const WIDE_SPACE: f32 = 20.0;
const SMALL_SPACE: f32 = 10.0;

#[derive(Debug)]
struct Model {
    name: String,
    poly: Polyhedron,
}

impl Model {
    pub fn new(name: &str, poly: Polyhedron) -> Self {
        Self {
            name: name.to_owned(),
            poly,
        }
    }
}

fn main() {
    // Create window
    let window = Window::new(WindowSettings {
        title: "Polygon Modeller".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

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
        poly.excavate(face_to_excavate, &tunnel, tunnel.get_ngon(4), 1, &[]);
        poly
    };

    // Base models
    let model_groups = [
        (
            "Platonic",
            vec![
                Model::new("Tetrahedron", Polyhedron::tetrahedron()),
                Model::new("Cube", Polyhedron::cube()),
                Model::new("Octahedron", Polyhedron::octahedron()),
                Model::new("Icosahedron", Polyhedron::icosahedron()),
            ],
        ),
        (
            "Basic",
            vec![
                Model::new("3-Pyramid = Tetrahedron", Polyhedron::pyramid(3).poly),
                Model::new("4-Pyramid", Polyhedron::pyramid(4).poly),
                Model::new("5-Pyramid", Polyhedron::pyramid(5).poly),
                Model::new("3-Cupola", Polyhedron::cupola(3).poly),
                Model::new("4-Cupola", Polyhedron::cupola(4).poly),
                Model::new("5-Cupola", Polyhedron::cupola(5).poly),
            ],
        ),
        (
            "Prisms & Antiprisms",
            vec![
                Model::new("3-Prism", Polyhedron::prism(3).poly),
                Model::new("4-Prism = Cube", Polyhedron::prism(4).poly),
                Model::new("5-Prism", Polyhedron::prism(5).poly),
                Model::new("6-Prism", Polyhedron::prism(6).poly),
                Model::new("3-Antiprism = Octahedron", Polyhedron::antiprism(3).poly),
                Model::new("4-Antiprism", Polyhedron::antiprism(4).poly),
                Model::new("5-Antiprism", Polyhedron::antiprism(5).poly),
                Model::new("6-Antiprism", Polyhedron::antiprism(6).poly),
            ],
        ),
        (
            "Archimedean",
            vec![
                Model::new("Truncated Tetrahedron", Polyhedron::truncated_tetrahedron()),
                Model::new("Truncated Cube", Polyhedron::truncated_cube()),
                Model::new("Truncated Octahedron", Polyhedron::truncated_octahedron()),
                // Model::new(
                //     "Truncated Dodecahedron",
                //     Polyhedron::truncated_dodecahedron(),
                // ),
                Model::new("Truncated Icosahedron", Polyhedron::truncated_icosahedron()),
                Model::new("Cuboctahedron", Polyhedron::cuboctahedron()),
                Model::new("Rhombicuboctahedron", Polyhedron::rhombicuboctahedron()),
            ],
        ),
        (
            "Toroids",
            vec![
                Model::new("Cake pan", {
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
                Model::new("Cakier pan", {
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
                Model::new("Cakiest pan", {
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
                Model::new("Torturous Tunnel", {
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
                Model::new("Q_3 P_6 Q_3 / P_6", qpq_slash_p(false)),
                Model::new("Q_3 P_6 gQ_3 / P_6", qpq_slash_p(true)),
                Model::new("Q_4^2 / B_4", {
                    let PrismLike {
                        mut poly,
                        bottom_face,
                        top_face,
                    } = Polyhedron::cupola(4);
                    poly.extend_cupola(bottom_face, true);
                    let tunnel = Polyhedron::cuboctahedron();
                    poly.excavate(top_face, &tunnel, tunnel.get_ngon(4), 0, &[]);
                    poly
                }),
                Model::new("Apanar Deltahedron", {
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
                    // Dig tunnel
                    poly.color_edges_added_by(Srgba::BLUE, |poly| {
                        poly.excavate_antiprism(bottom_face);
                        poly.excavate_antiprism(top_face);
                    });
                    // Add pyramids to all faces in the bicupola which still exist
                    for face in faces_to_add_pyramids {
                        if poly.is_face(face) {
                            poly.extend_pyramid(face);
                        }
                    }
                    poly
                }),
            ],
        ),
    ];

    // GUI variables
    let mut show_external_angles = false;
    let mut model_view_settings = ModelViewSettings::default();

    // Create model view
    let mut current_model = Polyhedron::truncated_tetrahedron();
    let mut view = model_view::ModelView::new(
        current_model.clone(),
        model_view_settings.as_render_style(),
        &context,
        window.viewport(),
    );

    // Main loop
    let mut gui = three_d::GUI::new(&context);
    window.render_loop(move |mut frame_input| {
        // Render GUI
        let mut left_panel_width = 0.0;
        let mut right_panel_width = 0.0;
        let mut redraw = gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |egui_context| {
                use three_d::egui::*;
                // Left panel
                let response =
                    SidePanel::left("left-panel")
                        .min_width(200.0)
                        .show(egui_context, |ui| {
                            ScrollArea::vertical().show(ui, |ui| {
                                for (group_name, models) in &model_groups {
                                    ui.heading(*group_name);
                                    for model in models {
                                        if ui.button(&model.name).clicked() {
                                            current_model = model.poly.clone();
                                        }
                                    }
                                    ui.add_space(WIDE_SPACE);
                                }
                            });
                        });
                left_panel_width = response.response.rect.width();
                // Right panel
                let response =
                    SidePanel::right("right-panel")
                        .min_width(250.0)
                        .show(egui_context, |ui| {
                            ui.heading("View Settings");
                            model_view_settings.gui(ui);

                            ui.add_space(WIDE_SPACE);
                            ui.separator();
                            ui.heading("Model Info");
                            model_info_gui(&current_model, &mut show_external_angles, ui);
                        });
                right_panel_width = response.response.rect.width();
            },
        );

        // Calculate remaining viewport
        let wl = (left_panel_width * frame_input.device_pixel_ratio) as i32;
        let wr = (right_panel_width * frame_input.device_pixel_ratio) as i32;
        let width = frame_input.viewport.width as i32 - wl - wr;
        let viewport = Viewport {
            x: wl,
            y: 0,
            width: width.max(1) as u32,
            height: frame_input.viewport.height,
        };

        // Update the 3D view
        redraw |= view.update(&mut frame_input, viewport);
        if redraw {
            let screen = frame_input.screen();
            screen.clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0));
            view.render(
                &current_model,
                model_view_settings.as_render_style(),
                &screen,
            );
            screen.write(|| gui.render());
        }

        FrameOutput {
            swap_buffers: redraw,
            wait_next_event: !redraw,
            ..Default::default()
        }
    });
}

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

            wireframe_edges: false,
            wireframe_verts: false,
        }
    }
}

impl ModelViewSettings {
    pub fn as_render_style(&self) -> RenderStyle {
        let face = match self.style {
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
                    fixed_angle: Some(polyhedron::FixedAngle {
                        unit_angle,
                        push_direction: self.direction,
                        add_crinkle: self.add_crinkle,
                    }),
                }
            }
        };
        RenderStyle {
            face,
            wireframe_edges: self.wireframe_edges,
            wireframe_verts: self.wireframe_verts,
        }
    }

    pub fn gui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Faces");
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
        const DEG_120_REDUCTION: f32 = 0.28867513; // 0.5 * tan(30 deg)
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

fn model_info_gui(polyhedron: &Polyhedron, show_external_angles: &mut bool, ui: &mut egui::Ui) {
    // Faces
    ui.strong(format!("{} faces", polyhedron.faces().count()));
    let faces_by_ngon = polyhedron.faces().into_group_map_by(|f| f.verts().len());
    ui.indent("faces", |ui| {
        for (n, faces) in faces_by_ngon.iter().sorted_by_key(|(n, _)| *n) {
            let count = faces.len();
            ui.label(format!(
                "{} {}{}",
                count,
                ngon_name(*n),
                if count > 1 { "s" } else { "" }
            ));
        }
    });

    // Edges
    ui.add_space(SMALL_SPACE);
    let edges = polyhedron.edges();
    let mut num_open_edges = 0;
    let mut edge_types =
        BTreeMap::<(usize, usize, Srgba), BTreeMap<OrderedFloat<f32>, Vec<&Edge>>>::new();
    for edge in &edges {
        match edge.closed {
            Some(ClosedEdgeData {
                left_face,
                dihedral_angle,
            }) => {
                let left_n = polyhedron.face_order(left_face);
                let right_n = polyhedron.face_order(edge.right_face);
                let mut dihedral = Degrees::from(dihedral_angle).0;
                dihedral = (dihedral * 128.0).round() / 128.0; // Round dihedral angle
                                                               // Record this new edge
                let col = edge.color.unwrap_or(polyhedron::DEFAULT_EDGE_COLOR);
                let edge_list: &mut Vec<&Edge> = edge_types
                    .entry((left_n.min(right_n), left_n.max(right_n), col))
                    .or_default()
                    .entry(OrderedFloat(dihedral))
                    .or_default();
                edge_list.push(edge)
            }
            None => num_open_edges += 1,
        }
    }
    ui.strong(format!("{} edges", edges.len()));
    ui.indent("edges", |ui| {
        if num_open_edges > 0 {
            ui.label(format!("{num_open_edges} open"));
        }
        let display_angle = |a: OrderedFloat<f32>| -> String {
            let mut angle = a.0;
            if *show_external_angles {
                angle = 360.0 - angle;
            }
            format!("{:.2}°", angle)
        };
        for ((n1, n2, color), angle_breakdown) in edge_types {
            let color = utils::srgba_to_egui_color(color);
            assert!(!angle_breakdown.is_empty());
            let num_edges = angle_breakdown.values().map(Vec::len).sum::<usize>();
            // Display overall group
            let angle_string = if angle_breakdown.len() == 1 {
                let only_angle = *angle_breakdown.keys().next().unwrap();
                format!(" ({})", display_angle(only_angle))
            } else {
                String::new()
            };
            ui.colored_label(
                color,
                format!(
                    "{} {}-{}{}",
                    num_edges,
                    ngon_name(n1),
                    ngon_name(n2),
                    angle_string,
                ),
            );
            // Add a angle breakdown if needed
            if angle_breakdown.len() > 1 {
                ui.indent("", |ui| {
                    for (angle, edges) in angle_breakdown {
                        ui.colored_label(
                            color,
                            format!("{}x {}", edges.len(), display_angle(angle)),
                        );
                    }
                });
            }
        }
        ui.checkbox(show_external_angles, "Measure external angles");
    });

    // Vertices
    ui.add_space(SMALL_SPACE);
    ui.strong(format!("{} vertices", polyhedron.vert_positions().len()));
}
