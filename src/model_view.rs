use three_d::*;

use crate::polyhedron::Polyhedron;

/// The 3D viewport used to display a model
pub(crate) struct ModelView {
    context: Context,
    mesh_cache: cache::MeshCache,

    camera: Camera,
    control: OrbitControl,

    face_material: PhysicalMaterial,
    wireframe_material: PhysicalMaterial,
}

impl ModelView {
    pub fn new(model: Polyhedron, context: &Context, viewport: Viewport) -> Self {
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
                albedo: Srgba::new_opaque(200, 200, 200),
                roughness: 0.7,
                metallic: 0.8,
                ..Default::default()
            },
        );
        face_material.render_states.cull = Cull::Back;
        let mut wireframe_material = PhysicalMaterial::new_opaque(
            context,
            &CpuMaterial {
                albedo: Srgba::new_opaque(200, 50, 50),
                roughness: 0.7,
                metallic: 0.8,
                ..Default::default()
            },
        );
        wireframe_material.render_states.cull = Cull::Back;

        Self {
            context: context.clone(),
            mesh_cache: cache::MeshCache::new(model, context),

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

    pub fn render(&mut self, model: &Polyhedron, target: &RenderTarget) {
        // Lights
        let ambient = AmbientLight::new(&self.context, 0.7, Srgba::WHITE);
        let directional0 =
            DirectionalLight::new(&self.context, 2.0, Srgba::WHITE, &vec3(-1.0, -1.0, -1.0));
        let directional1 =
            DirectionalLight::new(&self.context, 2.0, Srgba::WHITE, &vec3(1.0, 1.0, 1.0));
        let lights = [&ambient as &dyn Light, &directional0, &directional1];

        // Meshes
        let meshes = self.mesh_cache.get(model, &self.context);
        let face_mesh = Gm::new(&meshes.face_mesh, &self.face_material);
        let edge_mesh = Gm::new(&meshes.edge_mesh, &self.wireframe_material);
        let vertex_mesh = Gm::new(&meshes.vertex_mesh, &self.wireframe_material);

        target.render(
            &self.camera,
            [&face_mesh as &dyn Object, &vertex_mesh, &edge_mesh],
            &lights,
        );
    }
}

mod cache {
    use crate::polyhedron::{Meshes, Polyhedron, RenderStyle};
    use three_d::*;

    /// Caches the [`Mesh`]es for the model rendered in the last frame.  This means if the same
    /// model is rendered in consecutive frames then we can avoid resending the mesh data to the
    /// GPU.
    pub(super) struct MeshCache {
        model: Polyhedron,
        meshes: Meshes,
    }

    impl MeshCache {
        pub(super) fn new(model: Polyhedron, context: &Context) -> Self {
            let style = RenderStyle::OwLike {
                side_ratio: 0.25,
                fixed_angle: None,
            };
            Self {
                meshes: model.meshes(style, context),
                model,
            }
        }

        pub(super) fn get<'s>(&'s mut self, model: &Polyhedron, context: &Context) -> &'s Meshes {
            if &self.model != model {
                *self = Self::new(model.clone(), context);
            }
            &self.meshes
        }
    }
}
