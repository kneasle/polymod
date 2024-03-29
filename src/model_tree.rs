use itertools::Itertools;
use three_d::egui;

use crate::{
    model::{Model, ModelId},
    polyhedron::{FaceIdx, Polyhedron, PrismLike},
};

#[derive(Debug, Clone)]
pub enum ModelTree {
    Model(Model),
    Group {
        name: String,
        children: Vec<ModelTree>,
    },
}

impl ModelTree {
    pub fn first_model(&self) -> &Model {
        match self {
            ModelTree::Model(m) => m,
            ModelTree::Group { name: _, children } => children[0].first_model(),
        }
    }

    pub fn get_model_with_id(&self, id: ModelId) -> Option<&Model> {
        match self {
            ModelTree::Model(m) => (m.id() == id).then_some(m),
            ModelTree::Group { name: _, children } => {
                children.iter().find_map(|m| m.get_model_with_id(id))
            }
        }
    }

    pub fn get_mut_model_with_id(&mut self, id: ModelId) -> Option<&mut Model> {
        match self {
            ModelTree::Model(m) => (m.id() == id).then_some(m),
            ModelTree::Group { name: _, children } => children
                .iter_mut()
                .find_map(|m| m.get_mut_model_with_id(id)),
        }
    }

    pub fn add(&mut self, model: Model) {
        match self {
            ModelTree::Model(_) => panic!("Can't add a model to a non-group"),
            ModelTree::Group { name: _, children } => children.push(Self::Model(model)),
        }
    }

    pub fn tree_gui(&self, ui: &mut egui::Ui, current_model_id: &mut ModelId) {
        match self {
            ModelTree::Group { name, children } => {
                egui::CollapsingHeader::new(name)
                    .default_open(true)
                    .show(ui, |ui| {
                        for child in children {
                            child.tree_gui(ui, current_model_id);
                        }
                    });
            }
            ModelTree::Model(m) => {
                if m.id() == *current_model_id {
                    ui.strong(&m.name);
                } else {
                    if ui.button(&m.name).clicked() {
                        *current_model_id = m.id();
                    }
                }
            }
        }
    }
}

////////////////////
// BUILTIN MODELS //
////////////////////

impl ModelTree {
    pub fn builtin() -> Self {
        let platonic = [
            Model::new("Tetrahedron", Polyhedron::tetrahedron()),
            Model::new("Cube", Polyhedron::cube()),
            Model::new("Octahedron", Polyhedron::octahedron()),
            Model::new("Dodecahedron", Polyhedron::dodecahedron()),
            Model::new("Icosahedron", Polyhedron::icosahedron()),
        ];
        let archimedean_octa = [
            Model::new("Truncated Cube", Polyhedron::truncated_cube()),
            Model::new("Truncated Octahedron", Polyhedron::truncated_octahedron()),
            Model::new("Cuboctahedron", Polyhedron::cuboctahedron()),
            Model::new("Snub Cube", Polyhedron::snub_cube()),
            Model::new("Rhombicuboctahedron", Polyhedron::rhombicuboctahedron()),
            Model::new(
                "Great Rhombicuboctahedron",
                Polyhedron::great_rhombicuboctahedron(),
            ),
        ];
        let archimedean_dodeca = [
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
        ];

        let pyramidlikes = [
            Model::new("3-Pyramid = Tetrahedron", Polyhedron::pyramid(3).poly),
            Model::new("4-Pyramid", Polyhedron::pyramid(4).poly),
            Model::new("5-Pyramid", Polyhedron::pyramid(5).poly),
            Model::new("3-Cupola", Polyhedron::cupola(3).poly),
            Model::new("4-Cupola", Polyhedron::cupola(4).poly),
            Model::new("5-Cupola", Polyhedron::cupola(5).poly),
            Model::new("Rotunda", Polyhedron::rotunda().poly),
        ];
        let prismlikes = [
            Model::new("3-Prism", Polyhedron::prism(3).poly),
            Model::new("4-Prism = Cube", Polyhedron::prism(4).poly),
            Model::new("5-Prism", Polyhedron::prism(5).poly),
            Model::new("6-Prism", Polyhedron::prism(6).poly),
            Model::new("3-Antiprism = Octahedron", Polyhedron::antiprism(3).poly),
            Model::new("4-Antiprism", Polyhedron::antiprism(4).poly),
            Model::new("5-Antiprism", Polyhedron::antiprism(5).poly),
            Model::new("6-Antiprism", Polyhedron::antiprism(6).poly),
        ];

        Self::Group {
            name: "Built-in Models".to_owned(),
            children: [
                Self::new_group("Platonic", platonic),
                Self::Group {
                    name: "Archimedean".to_owned(),
                    children: [
                        Self::Model(Model::new(
                            "Truncated Tetrahedron",
                            Polyhedron::truncated_tetrahedron(),
                        )),
                        Self::new_group("Octahedral", archimedean_octa),
                        Self::new_group("Dodecahedral", archimedean_dodeca),
                    ]
                    .to_vec(),
                },
                Self::new_group("Pyramids and Cupolae", pyramidlikes),
                Self::new_group("Prisms and Antiprisms", prismlikes),
                Self::toroids(),
            ]
            .to_vec(),
        }
    }

    pub fn toroids() -> Self {
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

        let toroids = [
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
            {
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
                let (edges_to_color, _) = poly.get_edges_added_by(|poly| {
                    poly.excavate_antiprism(bottom_face);
                    poly.excavate_antiprism(top_face);
                });
                // Add pyramids to all faces in the bicupola which still exist
                for face in faces_to_add_pyramids {
                    if poly.is_face(face) {
                        poly.extend_pyramid(face);
                    }
                }
                // Create the model, and color the edges
                let mut model = Model::new("Apanar Deltahedron", poly);
                let blue = model.add_color(egui::Rgba::from_rgb(0.2, 0.35, 1.0));
                for (v1, v2) in edges_to_color {
                    model.set_full_edge_color(v1, v2, blue);
                }
                model
            },
        ];
        Self::new_group("Toroids", toroids)
    }

    pub fn new_group(name: &str, models: impl IntoIterator<Item = Model>) -> Self {
        Self::Group {
            name: name.to_owned(),
            children: models.into_iter().map(ModelTree::Model).collect_vec(),
        }
    }
}
