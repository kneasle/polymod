use itertools::Itertools;
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
    let models = [
        ("Tetrahedron", PolyModel::tetrahedron()),
        ("Cube", PolyModel::cube()),
        ("Octahedron", PolyModel::octahedron()),
        ("Cuboctahedron", PolyModel::cuboctahedron()),
        ("4-Pyramid", PolyModel::pyramid(4)),
        ("5-Pyramid", PolyModel::pyramid(5)),
        ("3-Prism", PolyModel::prism(3)),
        ("5-Prism", PolyModel::prism(5)),
        ("6-Prism", PolyModel::prism(6)),
        ("3-Cupola", PolyModel::cupola(3)),
        ("4-Cupola", PolyModel::cupola(4)),
        ("5-Cupola", PolyModel::cupola(5)),
    ];
    let models = models
        .into_iter()
        .map(|(name, poly)| crate::model::Model {
            name: name.to_owned(),
            poly,
        })
        .collect_vec();

    // Create model view
    let mut current_model = 0;
    let mut view = model_view::ModelView::new(
        models[current_model].poly.clone(),
        &context,
        window.viewport(),
    );

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
                    ui.heading("PolyMod");
                    for (idx, model) in models.iter().enumerate() {
                        if ui.button(&model.name).clicked() {
                            current_model = idx;
                        }
                    }
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
            view.render(&models[current_model].poly, &screen);
            screen.write(|| gui.render());
        }

        FrameOutput {
            swap_buffers: redraw,
            wait_next_event: true,
            ..Default::default()
        }
    });
}