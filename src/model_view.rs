use std::ops::Deref;

use three_d::*;

/// The 3D viewport used to display a model
pub(crate) struct ModelView {
    context: Context,

    camera: Camera,
    control: OrbitControl,

    face_material: PhysicalMaterial,
    wireframe_material: PhysicalMaterial,
}

impl ModelView {
    pub fn new(context: &Context, viewport: Viewport) -> Self {
        // Camera
        let target = vec3(0.0f32, 0.0, 0.0);
        let scene_radius = 6.0f32;
        let camera = Camera::new_perspective(
            viewport,
            target + scene_radius * vec3(0.0, 0.0, 1.0),
            target,
            vec3(0.0, 1.0, 0.0),
            degrees(45.0),
            0.1,
            1000.0,
        );
        let control = OrbitControl::new(*camera.target(), 0.1 * scene_radius, 100.0 * scene_radius);

        // Materials
        let mut face_material = PhysicalMaterial::new_opaque(
            context,
            &CpuMaterial {
                albedo: Srgba::WHITE,
                roughness: 0.7,
                metallic: 0.0,
                ..Default::default()
            },
        );
        face_material.render_states.cull = Cull::Back;
        let mut wireframe_material = PhysicalMaterial::new_opaque(
            context,
            &CpuMaterial {
                albedo: Srgba::WHITE,
                roughness: 0.7,
                metallic: 0.8,
                ..Default::default()
            },
        );
        wireframe_material.render_states.cull = Cull::Back;

        Self {
            context: context.clone(),

            camera,
            control,

            face_material,
            wireframe_material,
        }
    }

    pub fn update(&mut self, frame_input: &mut FrameInput, viewport: Viewport) -> bool {
        let mut redraw = frame_input.first_frame;
        redraw |= self.camera.set_viewport(viewport);
        redraw |= self
            .control
            .handle_events(&mut self.camera, &mut frame_input.events);
        redraw
    }

    pub fn render(&mut self, model: &crate::Model, target: &RenderTarget) {
        // Lights
        let ambient = AmbientLight::new(&self.context, 0.7, Srgba::WHITE);
        let directional0 =
            DirectionalLight::new(&self.context, 2.0, Srgba::WHITE, &vec3(-1.0, -1.0, -1.0));
        let directional1 =
            DirectionalLight::new(&self.context, 2.0, Srgba::WHITE, &vec3(1.0, 1.0, 1.0));
        let lights = [&ambient as &dyn Light, &directional0, &directional1];

        // Meshes
        // TODO: Add some caching to not send these to the GPU every frame
        let mut meshes: Vec<Box<dyn Object>> = Vec::new();
        if let Some(cpu_mesh) = model.face_mesh() {
            let mesh = Mesh::new(&self.context, &cpu_mesh);
            meshes.push(Box::new(Gm::new(mesh, &self.face_material)));
        }
        meshes.push(Box::new(Gm::new(
            model.edge_mesh(&self.context),
            &self.wireframe_material,
        )));
        meshes.push(Box::new(Gm::new(
            model.vertex_mesh(&self.context),
            &self.wireframe_material,
        )));

        target.render(&self.camera, meshes.iter().map(Deref::deref), &lights);
    }
}
