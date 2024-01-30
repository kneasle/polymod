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
    window.render_loop(move |mut frame_input| {
        let redraw = view.update(&mut frame_input);
        if redraw {
            view.render(&model, frame_input.screen());
        }
        FrameOutput {
            swap_buffers: redraw,
            wait_next_event: true,
            ..Default::default()
        }
    });
}
