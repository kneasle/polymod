use itertools::Itertools;
use three_d::egui::{self, Rgba};

use crate::{
    model::{Model, ModelId},
    polyhedron::{FaceIdx, Polyhedron, PrismLike, Pyramid},
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
    pub fn get_model_with_id(&self, id: ModelId) -> Option<&Model> {
        self.flatten().into_iter().find(|m| m.id() == id)
    }

    pub fn get_mut_model_with_id(&mut self, id: ModelId) -> Option<&mut Model> {
        self.flatten_mut().into_iter().find(|m| m.id() == id)
    }

    pub fn get_model_with_name<'s>(&'s self, name: &str) -> Option<&'s Model> {
        self.flatten().into_iter().find(|m| m.name() == name)
    }

    pub fn flatten_mut(&mut self) -> Vec<&mut Model> {
        let mut models = Vec::new();
        self.flatten_recursive_mut(&mut models);
        models
    }

    fn flatten_recursive_mut<'s>(&'s mut self, vec: &mut Vec<&'s mut Model>) {
        match self {
            ModelTree::Model(m) => vec.push(m),
            ModelTree::Group { name: _, children } => {
                for child in children {
                    child.flatten_recursive_mut(vec);
                }
            }
        }
    }

    pub fn flatten(&self) -> Vec<&Model> {
        let mut models = Vec::new();
        self.flatten_recursive(&mut models);
        models
    }

    fn flatten_recursive<'s>(&'s self, vec: &mut Vec<&'s Model>) {
        match self {
            ModelTree::Model(m) => vec.push(m),
            ModelTree::Group { name: _, children } => {
                for child in children {
                    child.flatten_recursive(vec);
                }
            }
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
                    ui.strong(m.name());
                } else {
                    if ui.button(m.name()).clicked() {
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
    pub fn new_group(name: &str, models: impl IntoIterator<Item = Model>) -> Self {
        Self::Group {
            name: name.to_owned(),
            children: models.into_iter().map(ModelTree::Model).collect_vec(),
        }
    }

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
            Model::new("7-Prism", Polyhedron::prism(7).poly),
            Model::new("8-Prism", Polyhedron::prism(8).poly),
            Model::new("3-Antiprism = Octahedron", Polyhedron::antiprism(3).poly),
            Model::new("4-Antiprism", Polyhedron::antiprism(4).poly),
            Model::new("5-Antiprism", Polyhedron::antiprism(5).poly),
            Model::new("6-Antiprism", Polyhedron::antiprism(6).poly),
            Model::new("7-Antiprism", Polyhedron::antiprism(7).poly),
            Model::new("8-Antiprism", Polyhedron::antiprism(8).poly),
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
            Model::new("Stephanie", {
                // Start with a coloured truncated dodecahedron
                let mut poly = Polyhedron::truncated_dodecahedron();
                // Excavate using cupolae and antiprisms to form the tunnels
                let mut inner_face = FaceIdx::new(0);
                for decagon in poly.ngons(10).collect_vec() {
                    let next = poly.excavate_cupola(decagon, false);
                    inner_face = poly.excavate_antiprism(next);
                }
                // Excavate the central cavity, and color these edges
                let inner = Polyhedron::dodecahedron();
                poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
                poly
            }),
            Model::new("Stephanie (Colouring A)", {
                // Start with a coloured truncated dodecahedron
                let mut poly = Polyhedron::truncated_dodecahedron();
                let outer_col = poly.add_color(egui::Rgba::from_rgb(0.2, 0.35, 1.0));
                poly.colour_all_edges(outer_col);
                // Excavate using cupolae and antiprisms to form the tunnels
                let mut inner_face = FaceIdx::new(0);
                for decagon in poly.ngons(10).collect_vec() {
                    let next = poly.excavate_cupola(decagon, false);
                    inner_face = poly.excavate_antiprism(next);
                }
                // Excavate the central cavity, and color these edges
                let mut inner = Polyhedron::dodecahedron();
                let inner_col = inner.add_color(egui::Rgba::from_rgb(1.0, 0.2, 0.35));
                inner.colour_all_edges(inner_col);
                poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
                poly
            }),
            Model::new("Stephanie (Colouring B)", {
                // Start with a coloured truncated dodecahedron
                let mut poly = Polyhedron::truncated_dodecahedron();
                let blue = poly.add_color(egui::Rgba::from_rgb(0.2, 0.35, 1.0));
                for tri in poly.ngons(3).collect_vec() {
                    poly.color_face(tri, blue);
                }
                // Excavate using cupolae and antiprisms to form the tunnels
                let mut inner_face = FaceIdx::new(0);
                for decagon in poly.ngons(10).collect_vec() {
                    let next = poly.excavate_cupola(decagon, false);
                    inner_face = poly.excavate_antiprism(next);
                }
                // Excavate the central cavity, and color these edges
                let inner = Polyhedron::dodecahedron();
                poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
                poly
            }),
            Model::new("Football", {
                let mut poly = Polyhedron::truncated_icosahedron();
                let black = poly.add_color(Rgba::BLACK);
                for face in poly.ngons(5).collect_vec() {
                    poly.color_face(face, black);
                }
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
                // Dig tunnel, and colour it blue
                let blue = poly.add_color(egui::Rgba::from_rgb(0.2, 0.35, 1.0));
                poly.color_edges_added_by(
                    |poly| {
                        poly.excavate_antiprism(bottom_face);
                        poly.excavate_antiprism(top_face);
                    },
                    blue,
                );
                // Add pyramids to all faces in the bicupola which still exist
                for face in faces_to_add_pyramids {
                    if poly.is_face(face) {
                        poly.extend_pyramid(face);
                    }
                }
                poly
            }),
            Model::new("Christopher", prism_extended_cuboctahedron()),
        ];
        Self::new_group("Toroids", toroids)
    }
}

fn prism_extended_cuboctahedron() -> Polyhedron {
    // Make the pyramids out of which the cuboctahedron's faces will be constructed
    let make_pyramid = |n: usize| -> (Polyhedron, Vec<FaceIdx>) {
        let Pyramid {
            mut poly,
            base_face,
        } = Polyhedron::pyramid(n);
        // Colour the base face
        let blue = poly.add_color(egui::Rgba::from_rgb(0.2, 0.35, 1.0));
        let verts = &poly.get_face(base_face).verts().to_vec();
        for (v1, v2) in verts.iter().copied().circular_tuple_windows() {
            poly.set_half_edge_color(v2, v1, blue);
        }
        // Get the side faces
        let side_faces = poly
            .faces_enumerated()
            .map(|(id, _)| id)
            .filter(|id| *id != base_face)
            .collect_vec();
        (poly, side_faces)
    };
    let (tri_pyramid, tri_pyramid_side_faces) = make_pyramid(3);
    let (quad_pyramid, quad_pyramid_side_faces) = make_pyramid(4);

    // Recursively build the polyhedron
    let mut poly = quad_pyramid.clone();
    let mut faces_to_expand = quad_pyramid_side_faces
        .iter()
        .map(|idx| (*idx, 3))
        .collect_vec();
    while let Some((face_to_extend, order_of_new_pyramid)) = faces_to_expand.pop() {
        // Get which pyramid we're going to add
        let (pyramid, side_faces, next_pyramid_order) = match order_of_new_pyramid {
            3 => (&tri_pyramid, &tri_pyramid_side_faces, 4),
            4 => (&quad_pyramid, &quad_pyramid_side_faces, 3),
            _ => unreachable!(),
        };

        // Add prism and pyramid
        if !poly.is_face(face_to_extend) {
            continue; // Face has already been connected to something
        }
        let opposite_face = poly.extend_prism(face_to_extend);
        if !poly.is_face(opposite_face) {
            continue; // Connecting to something which already exists
        }
        let face_mapping = poly.extend(opposite_face, pyramid, side_faces[0], 2);

        // Add new faces to extend
        for &source_side_face in side_faces {
            faces_to_expand.push((face_mapping[source_side_face], next_pyramid_order));
        }
    }

    // Centre the polyhedron and return it
    poly.make_centred();
    poly
}
