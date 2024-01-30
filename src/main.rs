use std::time::Instant;

use model::PolyModel;
use three_d::*;

mod model;

fn main() {
    let model = PolyModel::cuboctahedron();

    // Create window
    let window = Window::new(WindowSettings {
        title: "Wireframe!".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    // Camera
    let target = vec3(0.0f32, 0.0, 0.0);
    let scene_radius = 6.0f32;
    let mut camera = Camera::new_perspective(
        window.viewport(),
        target + scene_radius * vec3(0.0, 0.0, 1.0),
        target,
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(*camera.target(), 0.1 * scene_radius, 100.0 * scene_radius);

    // Materials
    let mut face_material = PhysicalMaterial::new_opaque(
        &context,
        &CpuMaterial {
            albedo: Srgba::new_opaque(200, 200, 200),
            roughness: 0.7,
            metallic: 0.8,
            ..Default::default()
        },
    );
    face_material.render_states.cull = Cull::Back;
    let mut wireframe_material = PhysicalMaterial::new_opaque(
        &context,
        &CpuMaterial {
            albedo: Srgba::new_opaque(200, 50, 50),
            roughness: 0.7,
            metallic: 0.8,
            ..Default::default()
        },
    );
    wireframe_material.render_states.cull = Cull::Back;

    // Meshes
    let face_mesh = Gm::new(Mesh::new(&context, &model.face_mesh()), face_material);

    // Lights
    let ambient = AmbientLight::new(&context, 0.7, Srgba::WHITE);
    let directional0 = DirectionalLight::new(&context, 2.0, Srgba::WHITE, &vec3(-1.0, -1.0, -1.0));
    let directional1 = DirectionalLight::new(&context, 2.0, Srgba::WHITE, &vec3(1.0, 1.0, 1.0));

    // Main loop
    window.render_loop(move |mut frame_input| {
        let mut redraw = frame_input.first_frame;
        redraw |= camera.set_viewport(frame_input.viewport);
        redraw |= control.handle_events(&mut camera, &mut frame_input.events);

        if redraw {
            frame_input
                .screen()
                .clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0))
                .render(
                    &camera,
                    face_mesh.into_iter(),
                    &[&ambient, &directional0, &directional1],
                );
        }

        FrameOutput {
            swap_buffers: redraw,
            wait_next_event: true,
            ..Default::default()
        }
    });
}
