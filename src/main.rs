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

    // Create model view
    let model = PolyModel::cuboctahedron();
    let mut view = model_view::ModelView::new(model.clone(), &context, window.viewport());

    // Main loop
    let mut gui = three_d::GUI::new(&context);
    let mut checked = false;
    let mut n = 0;
    window.render_loop(move |mut frame_input| {
        // Render GUI
        let mut panel_width = 0.0;
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |egui_context| {
                use three_d::egui::*;
                let response = SidePanel::left("left-panel").show(egui_context, |ui| {
                    ui.heading("PolyMod");
                    if ui.button("Say hello").clicked() {
                        println!("Hello!");
                    }
                    ui.checkbox(&mut checked, "Check?");
                    if checked {
                        println!("HI {n}");
                        n += 1;
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

        let redraw = view.update(&mut frame_input, viewport);
        if redraw {
            let screen = frame_input.screen();
            screen.clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0));
            view.render(&model, &screen);
            screen.write(|| gui.render());
        }

        // FrameOutput {
        //     swap_buffers: redraw,
        //     wait_next_event: true,
        //     ..Default::default()
        // }

        FrameOutput::default()
    });
}
