use model::PolyModel;
use three_d::*;

mod model;
mod model_view;

fn main() {
    // Create window
    let window = Window::new(WindowSettings {
        title: "Polygon Modeller".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    // Base models
    let model_groups = [
        (
            "Platonic",
            vec![
                model::Model::new("Tetrahedron", PolyModel::tetrahedron()),
                model::Model::new("Cube", PolyModel::cube()),
                model::Model::new("Octahedron", PolyModel::octahedron()),
            ],
        ),
        (
            "Basic",
            vec![
                model::Model::new("3-Pyramid = Tetrahedron", PolyModel::pyramid(3)),
                model::Model::new("4-Pyramid", PolyModel::pyramid(4)),
                model::Model::new("5-Pyramid", PolyModel::pyramid(5)),
                model::Model::new("3-Cupola", PolyModel::cupola(3)),
                model::Model::new("4-Cupola", PolyModel::cupola(4)),
                model::Model::new("5-Cupola", PolyModel::cupola(5)),
            ],
        ),
        (
            "Prisms & Antiprisms",
            vec![
                model::Model::new("3-Prism", PolyModel::prism(3)),
                model::Model::new("4-Prism = Cube", PolyModel::prism(4)),
                model::Model::new("5-Prism", PolyModel::prism(5)),
                model::Model::new("6-Prism", PolyModel::prism(6)),
                model::Model::new("3-Antiprism = Octahedron", PolyModel::antiprism(3)),
                model::Model::new("4-Antiprism", PolyModel::antiprism(4)),
                model::Model::new("5-Antiprism", PolyModel::antiprism(5)),
                model::Model::new("6-Antiprism", PolyModel::antiprism(6)),
            ],
        ),
        (
            "Archimedian",
            vec![model::Model::new(
                "Cuboctahedron",
                PolyModel::cuboctahedron(),
            )],
        ),
    ];

    // Create model view
    let mut current_model = PolyModel::tetrahedron();
    let mut view = model_view::ModelView::new(current_model.clone(), &context, window.viewport());

    // Main loop
    let mut gui = three_d::GUI::new(&context);
    window.render_loop(move |mut frame_input| {
        // Render GUI
        let mut panel_width = 0.0;
        let mut redraw = gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |egui_context| {
                use three_d::egui::*;
                let response = SidePanel::left("left-panel").show(egui_context, |ui| {
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
                panel_width = response.response.rect.width();
            },
        );

        // Calculate remaining viewport
        let w = (panel_width * frame_input.device_pixel_ratio) as u32;
        let viewport = Viewport {
            x: w as i32,
            y: 0,
            width: frame_input.viewport.width - w,
            height: frame_input.viewport.height,
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
            wait_next_event: true,
            ..Default::default()
        }
    });
}
