use std::collections::BTreeMap;

use itertools::Itertools;
use polyhedron::{Polyhedron, PrismLike};
use three_d::*;

use crate::{polyhedron::ClosedEdgeData, utils::ngon_name};

mod model_view;
mod polyhedron;
mod utils;

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
                    poly.excavate_antiprism(bottom_face);
                    poly.excavate_antiprism(top_face);
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

    // Create model view
    let mut current_model = Polyhedron::cuboctahedron();
    let mut view = model_view::ModelView::new(current_model.clone(), &context, window.viewport());

    // Main loop
    let mut show_external_angles = false;
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
                                    ui.add_space(20.0);
                                }
                            });
                        });
                left_panel_width = response.response.rect.width();
                // Right panel
                let response = SidePanel::right("right-panel")
                    .min_width(250.0)
                    .show(egui_context, |ui| {
                        draw_right_panel(&current_model, &mut show_external_angles, ui)
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
            height: frame_input.viewport.height as u32,
        };

        // Update the 3D view
        redraw |= view.update(&mut frame_input, viewport);
        if redraw {
            let screen = frame_input.screen();
            screen.clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0));
            view.render(&current_model, &screen);
            screen.write(|| gui.render());
        }

        FrameOutput {
            swap_buffers: redraw,
            wait_next_event: !redraw,
            ..Default::default()
        }
    });
}

fn draw_right_panel(polyhedron: &Polyhedron, show_external_angles: &mut bool, ui: &mut egui::Ui) {
    ui.add_space(20.0);

    ui.heading("Model Info");
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
    struct EdgeType {
        count: usize,
        min_dihedral: f32, // Degrees
        max_dihedral: f32, // Degrees
    }
    ui.add_space(10.0);
    let edges = polyhedron.edges();
    let mut num_open_edges = 0;
    let mut edge_types = BTreeMap::<(usize, usize), EdgeType>::new();
    for edge in &edges {
        match edge.closed {
            Some(ClosedEdgeData {
                left_face,
                dihedral_angle,
            }) => {
                let left_n = polyhedron.face_order(left_face);
                let right_n = polyhedron.face_order(edge.right_face);
                let mut dihedral = Degrees::from(dihedral_angle).0;
                if *show_external_angles {
                    dihedral = 360.0 - dihedral;
                }
                // Record this new edge
                let edge_type = edge_types
                    .entry((left_n.min(right_n), left_n.max(right_n)))
                    .or_insert_with(|| EdgeType {
                        count: 0,
                        min_dihedral: 360.0,
                        max_dihedral: 0.0,
                    });
                edge_type.count += 1;
                edge_type.min_dihedral = edge_type.min_dihedral.min(dihedral);
                edge_type.max_dihedral = edge_type.max_dihedral.max(dihedral);
            }
            None => num_open_edges += 1,
        }
    }
    ui.strong(format!("{} edges", edges.len()));
    ui.indent("edges", |ui| {
        if num_open_edges > 0 {
            ui.label(format!("{num_open_edges} open"));
        }
        for ((n1, n2), ty) in edge_types {
            let has_angle_range = (ty.min_dihedral - ty.max_dihedral).abs() > 0.001;
            let angle_string = if has_angle_range {
                format!("{:.2}°-{:.2}°", ty.min_dihedral, ty.max_dihedral)
            } else {
                format!("{:.2}°", ty.min_dihedral)
            };
            ui.label(format!(
                "{} {}-{} ({})",
                ty.count,
                ngon_name(n1),
                ngon_name(n2),
                angle_string,
            ));
        }
        ui.checkbox(show_external_angles, "Measure external angles");
    });

    // Vertices
    ui.add_space(10.0);
    ui.strong(format!("{} vertices", polyhedron.verts().len()));
}
