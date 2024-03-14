use std::collections::BTreeMap;

use itertools::Itertools;
use ordered_float::OrderedFloat;
use polyhedron::{FaceIdx, Polyhedron, PrismLike};
use three_d::{egui::RichText, *};

use crate::{
    polyhedron::{ClosedEdgeData, Edge},
    utils::ngon_name,
};

use model::Model;

mod model;
mod model_view;
mod polyhedron;
mod utils;

const WIDE_SPACE: f32 = 20.0;
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

    // Base models
    let mut model_groups = [
        (
            "Platonic",
            vec![
                Model::new("Tetrahedron", Polyhedron::tetrahedron()),
                Model::new("Cube", Polyhedron::cube()),
                Model::new("Octahedron", Polyhedron::octahedron()),
                Model::new("Dodecahedron", Polyhedron::dodecahedron()),
                Model::new("Icosahedron", Polyhedron::icosahedron()),
            ],
        ),
        (
            "Archimedean",
            vec![
                // Tetrahedral
                Model::new("Truncated Tetrahedron", Polyhedron::truncated_tetrahedron()),
                // Cuboctahedral
                Model::new("Truncated Cube", Polyhedron::truncated_cube()),
                Model::new("Truncated Octahedron", Polyhedron::truncated_octahedron()),
                Model::new("Cuboctahedron", Polyhedron::cuboctahedron()),
                Model::new("Snub Cube", Polyhedron::snub_cube()),
                Model::new("Rhombicuboctahedron", Polyhedron::rhombicuboctahedron()),
                Model::new(
                    "Great Rhombicuboctahedron",
                    Polyhedron::great_rhombicuboctahedron(),
                ),
                // Icosahedral
                Model::new(
                    "Truncated Dodecahedron",
                    Polyhedron::truncated_dodecahedron(),
                ),
                Model::new("Truncated Icosahedron", Polyhedron::truncated_icosahedron()),
                Model::new("Icosidodecahedron", Polyhedron::icosidodecahedron()),
                Model::new("Snub Dodecahedron", Polyhedron::snub_dodecahedron()),
                Model::new(
                    "Rhombicosidodecahedron",
                    Polyhedron::rhombicosidodecahedron(),
                ),
                Model::new(
                    "Great Rhombicosidodecahedron",
                    Polyhedron::great_rhombicosidodecahedron(),
                ),
            ],
        ),
        (
            "Pyramids & Cupolas",
            vec![
                Model::new("3-Pyramid = Tetrahedron", Polyhedron::pyramid(3).poly),
                Model::new("4-Pyramid", Polyhedron::pyramid(4).poly),
                Model::new("5-Pyramid", Polyhedron::pyramid(5).poly),
                Model::new("3-Cupola", Polyhedron::cupola(3).poly),
                Model::new("4-Cupola", Polyhedron::cupola(4).poly),
                Model::new("5-Cupola", Polyhedron::cupola(5).poly),
                Model::new("Rotunda", Polyhedron::rotunda().poly),
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
            "Toroids",
            vec![
                Model::new("Flying saucer", {
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
                Model::new("Oriental Hat", {
                    let PrismLike {
                        mut poly,
                        bottom_face,
                        top_face: _,
                    } = Polyhedron::rotunda();
                    let bottom_face = poly.excavate_cupola(bottom_face, false);
                    poly.excavate_antiprism(bottom_face);
                    poly
                }),
                Model::new("Bob", {
                    let mut poly = Polyhedron::truncated_cube();
                    let face = poly.ngons(8).nth(2).unwrap();
                    let next = poly.excavate_cupola(face, true);
                    let next = poly.excavate_prism(next);
                    poly.excavate_cupola(next, false);
                    poly
                }),
                Model::new("Gyrated Bob", {
                    let mut poly = Polyhedron::truncated_cube();
                    let face = poly.ngons(8).nth(2).unwrap();
                    let next = poly.excavate_cupola(face, false);
                    let next = poly.excavate_prism(next);
                    poly.excavate_cupola(next, false);
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
                    poly.excavate(top_face, &tunnel, tunnel.get_ngon(4), 0);
                    poly
                }),
                Model::new("K_3 / 3Q_3 (S_3)", {
                    let mut poly = Polyhedron::truncated_octahedron();
                    // Excavate cupolas (TODO: Do this by symmetry)
                    let mut inner_face = FaceIdx::new(0);
                    for face_idx in [0, 2, 4, 6] {
                        inner_face = poly.excavate_cupola(FaceIdx::new(face_idx), true);
                    }
                    // Excavate central octahedron
                    poly.excavate_antiprism(inner_face);
                    poly
                }),
                Model::new("K_4 (tunnel octagons)", {
                    let mut poly = Polyhedron::great_rhombicuboctahedron();
                    let mut inner_face = FaceIdx::new(0);
                    for octagon in poly.ngons(8).collect_vec() {
                        inner_face = poly.excavate_cupola(octagon, false);
                    }
                    let inner = Polyhedron::rhombicuboctahedron();
                    poly.excavate(inner_face, &inner, inner.get_ngon(4), 0);
                    poly
                }),
                Model::new("K_4 (tunnel hexagons)", {
                    let mut poly = Polyhedron::great_rhombicuboctahedron();
                    let mut inner_face = FaceIdx::new(0);
                    for hexagon in poly.ngons(6).collect_vec() {
                        inner_face = poly.excavate_cupola(hexagon, true);
                    }
                    let inner = Polyhedron::rhombicuboctahedron();
                    poly.excavate(inner_face, &inner, inner.get_ngon(3), 0);
                    poly
                }),
                Model::new("K_4 (tunnel cubes)", {
                    let mut poly = Polyhedron::great_rhombicuboctahedron();
                    let mut inner_face = FaceIdx::new(0);
                    for square in poly.ngons(4).collect_vec() {
                        inner_face = poly.excavate_prism(square);
                    }
                    let inner = Polyhedron::rhombicuboctahedron();
                    let face = inner.ngons(4).last().unwrap();
                    poly.excavate(inner_face, &inner, face, 0);
                    poly
                }),
                Model::new("K_5 (cupola/antiprism)", {
                    let mut poly = Polyhedron::great_rhombicosidodecahedron();
                    let mut inner_face = FaceIdx::new(0);
                    for decagon in poly.ngons(10).collect_vec() {
                        let next = poly.excavate_cupola(decagon, true);
                        inner_face = poly.excavate_antiprism(next);
                    }
                    let inner = Polyhedron::rhombicosidodecahedron();
                    poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
                    poly
                }),
                Model::new("K_5 (rotunda)", {
                    let mut poly = Polyhedron::great_rhombicosidodecahedron();
                    let mut inner_face = FaceIdx::new(0);
                    for decagon in poly.ngons(10).collect_vec() {
                        inner_face = poly.excavate_rotunda(decagon, true);
                    }
                    let inner = Polyhedron::rhombicosidodecahedron();
                    poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
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
                    let edges_to_color = poly.get_edges_added_by(|poly| {
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

    // Create model view
    let mut current_group_idx = 0;
    let mut current_model_idx = 4;
    macro_rules! current_model {
        () => {
            model_groups[current_group_idx].1[current_model_idx]
        };
    }
    let mut view = model_view::ModelView::new(&context, window.viewport());

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
                                for (group_idx, (group_name, models)) in
                                    model_groups.iter().enumerate()
                                {
                                    ui.heading(*group_name);
                                    for (model_idx, model) in models.iter().enumerate() {
                                        if ui.button(&model.name).clicked() {
                                            current_group_idx = group_idx;
                                            current_model_idx = model_idx;
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
                            ScrollArea::vertical().show(ui, |ui| {
                                let current_model = &mut current_model!();

                                ui.heading("View Settings");
                                current_model.draw_view_gui(ui);

                                ui.add_space(WIDE_SPACE);
                                ui.separator();
                                ui.heading("Model Properties");
                                model_properties_gui(&current_model.poly, ui);

                                ui.add_space(WIDE_SPACE);
                                ui.separator();
                                ui.heading("Model Geometry");
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
            view.render(&current_model!(), &screen);
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
    ui.strong("Faces");
    ui.indent("HI", |ui| {
        property_label(ui, all_flat, "Flat");
        property_label(ui, all_regular, "Regular");
    });

    // Edges
    let edges = polyhedron.edges();
    let unit_length_edges = match edges.iter().map(|e| e.length).minmax() {
        itertools::MinMaxResult::MinMax(x, y) => f32::abs(y - x) < 0.0001,
        _ => true,
    };
    ui.add_space(SMALL_SPACE);
    ui.strong("Edges");
    ui.indent(0, |ui| {
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
fn model_geom_gui(model: &Model, show_external_angles: &mut bool, ui: &mut egui::Ui) {
    // Faces
    ui.strong(format!("{} faces", model.poly.faces().count()));
    let faces_by_ngon = model.poly.faces().into_group_map_by(|f| f.verts().len());
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
    let edges = model.poly.edges();
    let mut num_open_edges = 0;
    let mut edge_types =
        BTreeMap::<(usize, usize, Srgba), BTreeMap<OrderedFloat<f32>, Vec<&Edge>>>::new();
    for edge in &edges {
        match edge.closed {
            Some(ClosedEdgeData {
                left_face,
                dihedral_angle,
            }) => {
                let left_n = model.poly.face_order(left_face);
                let right_n = model.poly.face_order(edge.right_face);
                let mut dihedral = Degrees::from(dihedral_angle).0;
                dihedral = (dihedral * 128.0).round() / 128.0; // Round dihedral angle
                                                               // Record this new edge
                let col = model.edge_side_color(edge.bottom_vert, edge.top_vert);
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
    ui.strong(format!("{} vertices", model.poly.vert_positions().len()));
}
