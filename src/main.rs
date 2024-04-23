use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashSet},
};

use itertools::Itertools;
use ordered_float::OrderedFloat;
use polyhedron::{EdgeAngleType, EdgeId, FaceIdx, Polyhedron, VertIdx};
use three_d::{egui::RichText, *};

use crate::{
    model::ModelId,
    model_tree::ModelTree,
    polyhedron::{ClosedEdgeData, Edge},
    utils::ngon_name,
};

use model::{Model, OwUnitGeometry};

mod model;
mod model_tree;
mod polyhedron;
mod utils;
mod viewport;

const BIG_SPACE: f32 = 20.0;
const SMALL_SPACE: f32 = 10.0;

fn main() {
    // Create window
    let window = Window::new(WindowSettings {
        title: "Polygon Modeller".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    let mut custom_tree = ModelTree::new_group("Custom Models", []);
    let builtin_tree = ModelTree::builtin();

    let mut current_model_id = builtin_tree
        .get_model_with_name("Christopher")
        .expect("No model with this name found")
        .id();

    // GUI variables
    let mut show_external_angles = false;
    let mut paper_width = 7.5; // cm

    // Create model view
    let mut view = viewport::Viewport::new(&context, window.viewport());

    // Main loop
    let mut gui = three_d::GUI::new(&context);
    window.render_loop(move |mut frame_input| {
        macro_rules! current_model {
            () => {{
                match custom_tree.get_model_with_id(current_model_id) {
                    Some(m) => m,
                    None => builtin_tree.get_model_with_id(current_model_id).unwrap(),
                }
            }};
        }

        // Render GUI
        let mut left_panel_width = 0.0;
        let mut right_panel_width = 0.0;
        let mut edges_to_highlight = HashSet::new();
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
                        .min_width(300.0)
                        .show(egui_context, |ui| {
                            ScrollArea::vertical().show(ui, |ui| {
                                custom_tree.tree_gui(ui, &mut current_model_id);
                                ui.add_space(BIG_SPACE);
                                ui.separator();
                                builtin_tree.tree_gui(ui, &mut current_model_id);
                            })
                        });
                left_panel_width = response.response.rect.width();
                // Right panel
                let response =
                    SidePanel::right("right-panel")
                        .min_width(250.0)
                        .show(egui_context, |ui| {
                            ScrollArea::vertical().show(ui, |ui| {
                                match custom_tree.get_mut_model_with_id(current_model_id) {
                                    Some(model) => {
                                        ui.heading("General Model Settings");
                                        ui.horizontal(|ui| {
                                            ui.label("Name:");
                                            ui.text_edit_singleline(model.name_mut());
                                        });

                                        ui.add_space(SMALL_SPACE);
                                        ui.heading("Geometry View");
                                        model.draw_view_geom_gui(ui);

                                        ui.add_space(SMALL_SPACE);
                                        ui.heading("Colors");
                                        model.draw_colors_gui(ui);
                                    }
                                    None => {
                                        ui.label("Built-in models can't be edited.");
                                        ui.label("Clone the model to edit it:");
                                        if ui.button("Clone model").clicked() {
                                            let mut new_model = current_model!().clone();
                                            current_model_id = ModelId::next_unique();
                                            new_model.set_id(current_model_id);
                                            custom_tree.add(new_model);
                                        }
                                    }
                                }

                                ui.add_space(BIG_SPACE);
                                ui.separator();

                                let current_model = current_model!();
                                ui.heading("Properties");
                                model_properties_gui(
                                    current_model.polyhedron(),
                                    current_model.view_geometry_settings().ow_unit_geometry(),
                                    &mut paper_width,
                                    ui,
                                );
                                ui.add_space(SMALL_SPACE);
                                ui.heading("Geometry Breakdown");
                                edges_to_highlight =
                                    model_geom_gui(current_model, &mut show_external_angles, ui);
                            });
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
            view.render(current_model!(), &edges_to_highlight, &screen);
            screen.write(|| gui.render());
        }

        FrameOutput {
            swap_buffers: redraw,
            wait_next_event: !redraw,
            ..Default::default()
        }
    });
}

fn model_properties_gui(
    polyhedron: &Polyhedron,
    ow_unit_geometry: Option<OwUnitGeometry>,
    paper_width: &mut f32,
    ui: &mut egui::Ui,
) {
    // Faces
    let all_flat = polyhedron.faces().all(|f| f.is_flat(polyhedron));
    let all_regular = polyhedron.faces().all(|f| f.is_regular(polyhedron));
    ui.horizontal(|ui| {
        ui.strong("Faces: ");
        property_label(ui, all_flat, "Flat");
        property_label(ui, all_regular, "Regular");
    });

    // Edges
    let edges = polyhedron.edges();
    let unit_length_edges = match edges.iter().map(|e| e.length).minmax() {
        itertools::MinMaxResult::MinMax(x, y) => f32::abs(y - x) < 0.0001,
        _ => true,
    };
    ui.horizontal(|ui| {
        ui.strong("Edges: ");
        property_label(ui, unit_length_edges, "Uniform length");
    });

    // Size
    let r = polyhedron.outsphere_radius();
    ui.strong("Outsphere diameter:");
    ui.indent("blah blah blah", |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.label(format!("{:.2} edge lengths", r * 2.0));
            if let Some(geom) = ow_unit_geometry {
                let real_diameter = r * 2.0 * geom.spine_length_factor * *paper_width;
                ui.label(format!(", or {:.2}cm if folded from ", real_diameter));
                ui.add(
                    egui::DragValue::new(paper_width)
                        .fixed_decimals(1)
                        .speed(0.05),
                );
                ui.label(format!("x{:.2}cm paper.", *paper_width * geom.paper_aspect));
            }
        });
    });
}

fn property_label(ui: &mut egui::Ui, value: bool, label: &str) {
    let mut text = RichText::new(label);
    if !value {
        text = text.strikethrough();
    }
    ui.label(text);
}

fn model_geom_gui(
    model: &Model,
    show_external_angles: &mut bool,
    ui: &mut egui::Ui,
) -> HashSet<EdgeId> {
    let poly = model.polyhedron();

    // Faces
    ui.strong(format!("{} faces", poly.faces().count()));
    ui.indent("faces", |ui| {
        let faces_by_ngon = poly.faces().into_group_map_by(|f| f.verts().len());
        for (n, faces) in faces_by_ngon.iter().sorted_by_key(|(n, _)| *n) {
            let count = faces.len();
            ui.label(format!(
                "{}x {}{}",
                count,
                ngon_name(*n),
                if count > 1 { "s" } else { "" }
            ));
        }
    });

    let mut edges_to_highlight = HashSet::new();

    // Edges
    ui.add_space(SMALL_SPACE);
    let edges = poly.edges();
    let mut num_open_edges = 0;
    let mut edge_types = BTreeMap::<EdgeFaceType, BTreeMap<EdgeSubType, Vec<&Edge>>>::new();
    for edge in &edges {
        match edge.closed {
            Some(ClosedEdgeData { left_face, .. }) => {
                // Get face data
                let left_n = poly.face_order(left_face);
                let right_n = poly.face_order(edge.right_face);
                let col_left = poly.get_edge_side_color(edge.bottom_vert, edge.top_vert);
                let col_right = poly.get_edge_side_color(edge.top_vert, edge.bottom_vert);
                // Categorise this new edge
                let (face_type, face_cmp) = EdgeFaceType::new(left_n, right_n, col_left, col_right);
                let sub_type = EdgeSubType::new(edge, face_cmp, poly, &edges);
                // Add this new edge to its corresponding category
                let edge_list: &mut Vec<&Edge> = edge_types
                    .entry(face_type)
                    .or_default()
                    .entry(sub_type)
                    .or_default();
                edge_list.push(edge)
            }
            None => num_open_edges += 1,
        }
    }
    ui.strong(format!("{} edges", edges.len()));
    ui.indent("edges", |ui| {
        if num_open_edges > 0 {
            ui.label(format!("{num_open_edges}x open"));
        }
        for (face_type, angle_breakdown) in edge_types {
            assert!(!angle_breakdown.is_empty());
            let num_edges = angle_breakdown.values().map(Vec::len).sum::<usize>();
            // Display overall group (e.g. "60 triangle-pentagon (152.0°)")
            let response = ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                // Count
                ui.label(format!("{}x ", num_edges));
                // Ngon pair (e.g. "triangle-decagon")
                ui.colored_label(
                    model.get_color(face_type.left_color),
                    ngon_name(face_type.left_order),
                );
                ui.label("-");
                ui.colored_label(
                    model.get_color(face_type.right_color),
                    ngon_name(face_type.right_order),
                );
                // Angle
                if angle_breakdown.len() == 1 {
                    let sub_type = *angle_breakdown.keys().next().unwrap();
                    ui.label(format!(" (all {})", sub_type.show(*show_external_angles)));
                }
            });
            if response.response.hovered() {
                // Highlight these edges if hovered
                edges_to_highlight = angle_breakdown.values().flatten().map(|e| e.id()).collect();
            }
            // Add a angle breakdown if needed
            if angle_breakdown.len() > 1 {
                ui.indent("", |ui| {
                    for (sub_type, edges) in angle_breakdown {
                        let response = ui.label(format!(
                            "{}x {}",
                            edges.len(),
                            sub_type.show(*show_external_angles),
                        ));
                        if response.hovered() {
                            // Highlight these edges if hovered
                            edges_to_highlight = edges.iter().map(|e| e.id()).collect();
                        }
                    }
                });
            }
        }
        ui.checkbox(show_external_angles, "Measure external angles");
    });

    // Vertices
    ui.add_space(SMALL_SPACE);
    ui.strong(format!("{} vertices", poly.verts().len()));

    // Return which edges to highlight
    edges_to_highlight
}

/// Top-level edge classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeFaceType<'poly> {
    left_order: usize,
    right_order: usize,
    left_color: Option<&'poly str>,
    right_color: Option<&'poly str>,
}

impl<'poly> EdgeFaceType<'poly> {
    /// Create a new `EdgeFaceType`, ensuring that left < right
    fn new(
        left_order: usize,
        right_order: usize,
        left_color: Option<&'poly str>,
        right_color: Option<&'poly str>,
    ) -> (Self, Ordering) {
        // Normalize faces so that left < right
        let mut left = (left_order, left_color);
        let mut right = (right_order, right_color);
        let ordering = left.cmp(&right);
        if ordering.is_gt() {
            std::mem::swap(&mut left, &mut right);
        }

        let ty = Self {
            left_order: left.0,
            right_order: right.0,
            left_color: left.1,
            right_color: right.1,
        };
        (ty, ordering)
    }
}

/// Second-level edge classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeSubType {
    dihedral_angle: OrderedFloat<f32>,
    left_type: EdgeAngleType,
    spine_type: EdgeAngleType,
    right_type: EdgeAngleType,
}

impl EdgeSubType {
    fn new(edge: &Edge, face_cmp: Ordering, poly: &Polyhedron, edges: &[Edge]) -> Self {
        // Round dihedral angle
        let mut dihedral = Degrees::from(edge.dihedral_angle().unwrap()).0;
        dihedral = (dihedral * 128.0).round() / 128.0;

        let edge_angle_type = |face_idx: FaceIdx, vert_idx: VertIdx| -> EdgeAngleType {
            // Find the next vertex round the face
            let face = poly.get_face(face_idx);
            let (_, other_vert) = face
                .verts()
                .iter()
                .circular_tuple_windows()
                .find(|(v1, _v2)| **v1 == vert_idx)
                .unwrap();
            // Find the edge going between `vert_idx` and `other_vert`
            let edge = edges
                .iter()
                .find(|e| e.has_verts(vert_idx, *other_vert))
                .unwrap();
            edge.angle_type()
        };

        let mut left_face = edge.closed.as_ref().unwrap().left_face;
        let mut right_face = edge.right_face;
        let mut left_vert = edge.bottom_vert;
        let mut right_vert = edge.top_vert;
        // Normalize face display order, so that the smaller face is always on the left
        if face_cmp.is_lt() {
            std::mem::swap(&mut left_face, &mut right_face);
            std::mem::swap(&mut left_vert, &mut right_vert);
        }

        // Get left/right angle types
        let mut left_type = edge_angle_type(left_face, left_vert);
        let mut right_type = edge_angle_type(right_face, right_vert);
        // If both faces have the same order, then normalize the angle types
        if face_cmp.is_eq() && left_type > right_type {
            std::mem::swap(&mut left_type, &mut right_type);
        }

        Self {
            dihedral_angle: OrderedFloat(dihedral),

            left_type,
            spine_type: edge.angle_type(),
            right_type,
        }
    }

    fn show(self, show_external_angles: bool) -> String {
        let mut angle = self.dihedral_angle.0;
        if show_external_angles {
            angle = 360.0 - angle;
        }
        format!(
            "{}{}{} @ {:.2}°",
            self.left_type.as_char(),
            self.spine_type.as_char(),
            self.right_type.as_char(),
            angle,
        )
    }
}
