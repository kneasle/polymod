use std::collections::BTreeMap;

use itertools::Itertools;
use ordered_float::OrderedFloat;
use polyhedron::Polyhedron;
use three_d::{egui::RichText, *};
use utils::OrderedRgba;

use crate::{
    model::ModelId,
    model_tree::ModelTree,
    polyhedron::{ClosedEdgeData, Edge},
    utils::ngon_name,
};

use model::Model;

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
        .get_model_with_name("Prism-Extended Cuboctahedron")
        .unwrap()
        .id();

    // GUI variables
    let mut show_external_angles = false;

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
                                        ui.heading("View Settings");
                                        model.draw_view_gui(ui);
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
                                model_properties_gui(current_model.polyhedron(), ui);
                                ui.add_space(SMALL_SPACE);
                                ui.heading("Geometry");
                                model_geom_gui(
                                    current_model.polyhedron(),
                                    &mut show_external_angles,
                                    ui,
                                );
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
            view.render(current_model!(), &screen);
            screen.write(|| gui.render());
        }

        FrameOutput {
            swap_buffers: redraw,
            wait_next_event: !redraw,
            ..Default::default()
        }
    });
}

fn model_properties_gui(polyhedron: &Polyhedron, ui: &mut egui::Ui) {
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
}

fn property_label(ui: &mut egui::Ui, value: bool, label: &str) {
    let mut text = RichText::new(label);
    if !value {
        text = text.strikethrough();
    }
    ui.label(text);
}

// TODO: Make this a method
fn model_geom_gui(poly: &Polyhedron, show_external_angles: &mut bool, ui: &mut egui::Ui) {
    // Faces
    ui.strong(format!("{} faces", poly.faces().count()));
    ui.indent("faces", |ui| {
        let faces_by_ngon = poly.faces().into_group_map_by(|f| f.verts().len());
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
    let edges = poly.edges();
    let mut num_open_edges = 0;
    let mut edge_types = BTreeMap::<
        ((usize, OrderedRgba), (usize, OrderedRgba)),
        BTreeMap<OrderedFloat<f32>, Vec<&Edge>>,
    >::new();
    for edge in &edges {
        match edge.closed {
            Some(ClosedEdgeData {
                left_face,
                dihedral_angle,
            }) => {
                let left_n = poly.face_order(left_face);
                let right_n = poly.face_order(edge.right_face);
                let mut dihedral = Degrees::from(dihedral_angle).0;
                dihedral = (dihedral * 128.0).round() / 128.0; // Round dihedral angle

                let col_left = poly.edge_side_color(edge.bottom_vert, edge.top_vert);
                let col_right = poly.edge_side_color(edge.top_vert, edge.bottom_vert);
                let mut left = (left_n, OrderedRgba(col_left));
                let mut right = (right_n, OrderedRgba(col_right));
                if left > right {
                    std::mem::swap(&mut left, &mut right);
                }
                // Record this new edge
                let edge_list: &mut Vec<&Edge> = edge_types
                    .entry((left, right))
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
        for (key, angle_breakdown) in edge_types {
            let ((n1, OrderedRgba(col1)), (n2, OrderedRgba(col2))) = key;
            assert!(!angle_breakdown.is_empty());
            let num_edges = angle_breakdown.values().map(Vec::len).sum::<usize>();
            // Display overall group (e.g. "60 triangle-pentagon (152.0°)")
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                // Count
                ui.label(format!("{} ", num_edges));
                // Ngon pair (e.g. "triangle-decagon")
                ui.colored_label(col1, ngon_name(n1));
                ui.label("-");
                ui.colored_label(col2, ngon_name(n2));
                // Angle
                if angle_breakdown.len() == 1 {
                    let only_angle = *angle_breakdown.keys().next().unwrap();
                    ui.label(format!(" ({})", display_angle(only_angle)));
                }
            });
            // Add a angle breakdown if needed
            if angle_breakdown.len() > 1 {
                ui.indent("", |ui| {
                    for (angle, edges) in angle_breakdown {
                        ui.label(format!("{}x {}", edges.len(), display_angle(angle)));
                    }
                });
            }
        }
        ui.checkbox(show_external_angles, "Measure external angles");
    });

    // Vertices
    ui.add_space(SMALL_SPACE);
    ui.strong(format!("{} vertices", poly.vert_positions().len()));
}
