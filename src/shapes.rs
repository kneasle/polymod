#![allow(clippy::unnecessary_to_owned)]

use itertools::Itertools;
use three_d::{Deg, Vec3};

use crate::{
    model::{Model, OwUnit},
    polyhedron::{Cube, EdgeId, FaceIdx, Polyhedron, PrismLike, Pyramid},
    utils::{Axis, Side},
    COLOR_THEME,
};

/// Create all the shapes created so far
pub fn all() -> Vec<Model> {
    let mut all_models = Vec::new();
    add_origamis(&mut all_models);
    add_shapes(&mut all_models);
    all_models
}

////////////////////
// ORIGAMI MODELS //
////////////////////

pub fn add_origamis(all_models: &mut Vec<Model>) {
    let models = [
        Model::new("Bi Cube", bi_cube()).with_ow_unit(OwUnit::CustomDeg90, 1, 1),
        // Truncated tetrahedron
        // Cuboctahedron
        Model::new("Flying Saucer", flying_saucer())
            .with_ow_unit(OwUnit::CustomDeg90, 3, 2)
            .with_push(Side::Out, true)
            .with_default_color(COLOR_THEME.green),
        // Snub cube
        Model::new("Terribly Torturous Tunnel", aplanar_deltahedron()).with_ow_unit(
            OwUnit::Deg60,
            3,
            2,
        ),
        // Cuboctahedral rings
        // Icosidodecahedral rings
        // Octahedral rings
        // Dodecahedron
        // Tri prism
        // Hex prism
        // Oct prism
        Model::new("Christopher", christopher()).with_ow_unit(OwUnit::CustomDeg90, 3, 2),
        Model::new("Amethyst", q4_q4_slash_b4())
            .with_ow_unit(OwUnit::Custom3468, 3, 2)
            .with_default_color(COLOR_THEME.mauve),
        // Torturous tunnel
        Model::new("Chamfered RGB Cube", colored_great_rhombicuboctahedron())
            .with_ow_unit(OwUnit::Ow68, 1, 1)
            .with_default_color(crate::COLOR_THEME.surface0),
    ];

    let ideas = [Model::new("Robin", robin()).with_ow_unit(OwUnit::Custom468, 3, 2)];

    for mut model in models {
        model.full_name = format!("Origami\\{}", model.full_name);
        all_models.push(model);
    }
    for mut model in ideas {
        model.full_name = format!("Origami\\Ideas\\{}", model.full_name);
        all_models.push(model);
    }
}

fn colored_great_rhombicuboctahedron() -> Polyhedron {
    let mut poly = Polyhedron::great_rhombicuboctahedron();
    let mut color_face = |normal: Vec3, color: &str| {
        let face_idx = poly.get_face_with_normal(normal);
        for (v1, v2) in poly
            .get_face(face_idx)
            .verts()
            .to_vec()
            .into_iter()
            .circular_tuple_windows()
        {
            poly.set_full_edge_color(EdgeId::new(v1, v2), color);
        }
    };
    color_face(Vec3::unit_x(), "X");
    color_face(-Vec3::unit_x(), "X");
    color_face(Vec3::unit_y(), "Y");
    color_face(-Vec3::unit_y(), "Y");
    color_face(Vec3::unit_z(), "Z");
    color_face(-Vec3::unit_z(), "Z");
    poly
}

fn christopher() -> Polyhedron {
    let tri_pyramid = PrismExtensionSection::pyramid(3, Some("Triangles"));
    let quad_pyramid = PrismExtensionSection::pyramid(4, Some("Squares"));
    let poly = prism_extended_cuboctahedron(&tri_pyramid, &quad_pyramid);
    poly
}

fn aplanar_deltahedron() -> Polyhedron {
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
    // Dig tunnel, and color it blue
    poly.color_edges_added_by("Tunnel", |poly| {
        poly.excavate_antiprism(bottom_face);
        poly.excavate_antiprism(top_face);
    });
    // Add pyramids to all faces in the bicupola which still exist
    for face in faces_to_add_pyramids {
        if poly.is_face(face) {
            poly.extend_pyramid(face);
        }
    }
    poly
}

fn bi_cube() -> Polyhedron {
    let mut poly = Polyhedron::cube_poly();
    for edge in poly.edges() {
        let direction = poly.vert_pos(edge.top_vert) - poly.vert_pos(edge.bottom_vert);
        let axis = Axis::exact_axis(direction).unwrap();
        poly.set_full_edge_color(edge.id(), axis.name());
    }
    poly
}

fn robin() -> Polyhedron {
    // Create a `PrismExtensionSection` for a cupola
    const COLOR_NAME: &str = "Cupolae";
    let mut cupola = Polyhedron::cupola_with_top(4).poly;
    cupola.color_all_edges(COLOR_NAME);
    let cupola_section = PrismExtensionSection {
        side_faces: cupola.ngons(3),
        poly: cupola,
    };
    // Combine it with triangular prisms to build the model
    let tri_pyramid = PrismExtensionSection::pyramid(3, None);
    let mut poly = prism_extended_cuboctahedron(&cupola_section, &tri_pyramid);
    // Add extra colours along the prisms
    for e in poly.edges() {
        let is_about_60_degrees = e
            .dihedral_angle()
            .is_some_and(|Deg(a)| (a - 60.0).abs() < 0.001);
        if is_about_60_degrees {
            poly.set_full_edge_color(e.id(), COLOR_NAME);
        }
    }
    // Redo colouring by axis
    for face_idx in poly.face_indices() {
        if poly.face_order(face_idx) == 3 {
            // Colour each edge of the triangle differently
            let edges_round_face = poly.get_face(face_idx).verts().to_vec();
            for (v1, v2) in edges_round_face.into_iter().circular_tuple_windows() {
                let midpoint = (poly.vert_pos(v1) + poly.vert_pos(v2)) / 2.0;
                let nearest_axis_name = Axis::nearest_to(midpoint).name();
                poly.set_half_edge_color(v2, v1, nearest_axis_name);
            }
        } else {
            // Color each other face by its nearest axis
            let nearest_axis = Axis::nearest_to(poly.face_centroid(face_idx));
            poly.color_face(face_idx, nearest_axis.name());
        }
    }
    poly
}

#[derive(Debug, Clone)]
struct PrismExtensionSection {
    poly: Polyhedron,
    side_faces: Vec<FaceIdx>,
}

impl PrismExtensionSection {
    fn pyramid(n: usize, color_name: Option<&str>) -> Self {
        let Pyramid {
            mut poly,
            base_face,
        } = Polyhedron::pyramid(n);
        // Color the base face
        if let Some(color_name) = color_name {
            let verts = &poly.get_face(base_face).verts().to_vec();
            for (v1, v2) in verts.iter().copied().circular_tuple_windows() {
                poly.set_half_edge_color(v2, v1, color_name);
            }
        }
        // Get the side faces
        let side_faces = poly
            .faces_enumerated()
            .map(|(id, _)| id)
            .filter(|id| *id != base_face)
            .collect_vec();
        Self { poly, side_faces }
    }
}

fn prism_extended_cuboctahedron(
    section_a: &PrismExtensionSection,
    section_b: &PrismExtensionSection,
) -> Polyhedron {
    /// Which section should be created on the other end of a pyramid
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SectionType {
        A,
        B,
    }

    // Recursively build the polyhedron
    let mut poly = section_a.poly.clone();
    let mut faces_to_expand = section_a
        .side_faces
        .iter()
        .map(|idx| (*idx, SectionType::B)) // The other sides will all be Bs
        .collect_vec();
    while let Some((face_to_extend, next_section_type)) = faces_to_expand.pop() {
        let (next_section, other_section_type) = match next_section_type {
            SectionType::A => (&section_a, SectionType::B),
            SectionType::B => (&section_b, SectionType::A),
        };

        // Add prism and pyramid
        if !poly.is_face(face_to_extend) {
            continue; // Face has already been connected from the other side
        }
        let opposite_face = poly.extend_prism(face_to_extend);
        if !poly.is_face(opposite_face) {
            continue; // Connecting to a section which already exists
        }
        let face_mapping = poly.extend(
            opposite_face,
            &next_section.poly,
            next_section.side_faces[0],
            0,
        );

        // Add new faces to extend
        for &source_side_face in &next_section.side_faces {
            faces_to_expand.push((face_mapping[source_side_face], other_section_type));
        }
    }

    // Centre the polyhedron and return it
    poly.make_centred();
    poly
}

////////////
// SHAPES //
////////////

pub fn add_shapes(all_models: &mut Vec<Model>) {
    let platonic = [
        ("Tetrahedron", Polyhedron::tetrahedron()),
        ("Cube", Polyhedron::cube_poly()),
        ("Octahedron", Polyhedron::octahedron()),
        ("Dodecahedron", Polyhedron::dodecahedron()),
        ("Icosahedron", Polyhedron::icosahedron()),
    ];
    let archimedean_tetra = [("Truncated Tetrahedron", Polyhedron::truncated_tetrahedron())];
    let archimedean_octa = [
        ("Truncated Cube", Polyhedron::truncated_cube()),
        ("Truncated Octahedron", Polyhedron::truncated_octahedron()),
        ("Cuboctahedron", Polyhedron::cuboctahedron()),
        ("Snub Cube", Polyhedron::snub_cube()),
        ("Rhombicuboctahedron", Polyhedron::rhombicuboctahedron()),
        (
            "Great Rhombicuboctahedron",
            Polyhedron::great_rhombicuboctahedron(),
        ),
    ];
    let archimedean_dodeca = [
        (
            "Truncated Dodecahedron",
            Polyhedron::truncated_dodecahedron(),
        ),
        ("Truncated Icosahedron", Polyhedron::truncated_icosahedron()),
        ("Icosidodecahedron", Polyhedron::icosidodecahedron()),
        ("Snub Dodecahedron", Polyhedron::snub_dodecahedron()),
        (
            "Rhombicosidodecahedron",
            Polyhedron::rhombicosidodecahedron(),
        ),
        (
            "Great Rhombicosidodecahedron",
            Polyhedron::great_rhombicosidodecahedron(),
        ),
    ];

    let pyramidlikes = [
        ("3-Pyramid = Tetrahedron", Polyhedron::pyramid(3).poly),
        ("4-Pyramid", Polyhedron::pyramid(4).poly),
        ("5-Pyramid", Polyhedron::pyramid(5).poly),
        ("3-Cupola", Polyhedron::cupola(3).poly),
        ("4-Cupola", Polyhedron::cupola(4).poly),
        ("5-Cupola", Polyhedron::cupola(5).poly),
        ("Rotunda", Polyhedron::rotunda().poly),
    ];
    let prismlikes = [
        ("3-Prism", Polyhedron::prism(3).poly),
        ("4-Prism = Cube", Polyhedron::prism(4).poly),
        ("5-Prism", Polyhedron::prism(5).poly),
        ("6-Prism", Polyhedron::prism(6).poly),
        ("7-Prism", Polyhedron::prism(7).poly),
        ("8-Prism", Polyhedron::prism(8).poly),
        ("3-Antiprism = Octahedron", Polyhedron::antiprism(3).poly),
        ("4-Antiprism", Polyhedron::antiprism(4).poly),
        ("5-Antiprism", Polyhedron::antiprism(5).poly),
        ("6-Antiprism", Polyhedron::antiprism(6).poly),
        ("7-Antiprism", Polyhedron::antiprism(7).poly),
        ("8-Antiprism", Polyhedron::antiprism(8).poly),
    ];

    let groups = [
        (r"Platonic", platonic.to_vec()),
        (r"Archimedean", archimedean_tetra.to_vec()),
        (r"Archimedean\Octahedral", archimedean_octa.to_vec()),
        (r"Archimedean\Dodecahedral", archimedean_dodeca.to_vec()),
        (r"Pyramids and Cupolae", pyramidlikes.to_vec()),
        (r"Prisms and Antiprisms", prismlikes.to_vec()),
    ];

    for (group_name, models) in groups {
        for (model_name, poly) in models {
            all_models.push(Model::new(full_builtin_name(group_name, model_name), poly));
        }
    }
    for mut model in toroids() {
        model.full_name = full_builtin_name("Toroids", &model.full_name);
        all_models.push(model);
    }
}

fn toroids() -> Vec<Model> {
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
        Model::new("Flying saucer", flying_saucer()),
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
        {
            let mut poly = Polyhedron::truncated_cube();
            let face = poly.get_face_with_normal(Vec3::unit_x());
            poly.color_faces_added_by("Tunnel", |poly| {
                let next = poly.excavate_cupola(face, true);
                let next = poly.excavate_prism(next);
                poly.excavate_cupola(next, false);
            });
            Model::new("Bob", poly)
        },
        {
            let mut poly = Polyhedron::truncated_cube();
            let face = poly.get_face_with_normal(Vec3::unit_x());
            poly.color_faces_added_by("Tunnel", |poly| {
                let next = poly.excavate_cupola(face, false);
                let next = poly.excavate_prism(next);
                poly.excavate_cupola(next, false);
            });

            Model::new("Gyrated Bob", poly)
        },
        {
            let mut poly = Polyhedron::truncated_cube();
            // Extend +x and -x faces with cupolae
            let face = poly.get_face_with_normal(Vec3::unit_x());
            let back_face = poly.get_face_with_normal(-Vec3::unit_x());
            poly.extend_cupola(back_face, true);
            let next = poly.extend_cupola(face, true);
            // Tunnel with B_4 (P_4) B_4
            poly.color_faces_added_by("Tunnel", |poly| {
                let next = poly.excavate_cuboctahedron(next);
                let next = poly.excavate_prism(next);
                poly.excavate_cuboctahedron(next);
            });

            Model::new("Dumbell", poly)
        },
        Model::new("Q_3 P_6 Q_3 / P_6", qpq_slash_p(false)),
        Model::new("Q_3 P_6 gQ_3 / P_6", qpq_slash_p(true)),
        Model::new("Q_4^2 / B_4", q4_q4_slash_b4()),
        {
            let mut poly = Polyhedron::truncated_octahedron();
            // Excavate cupolas (TODO: Do this by symmetry)
            let mut inner_face = FaceIdx::new(0);
            poly.color_faces_added_by("Tunnels", |poly| {
                for face_idx in [0, 2, 4, 6] {
                    inner_face = poly.excavate_cupola(FaceIdx::new(face_idx), true);
                }
            });
            // Excavate central octahedron
            poly.color_faces_added_by("Centre", |poly| {
                poly.excavate_antiprism(inner_face);
            });
            Model::new("K_3 / 3Q_3 (S_3)", poly)
        },
        Model::new("K_4 (tunnel octagons)", {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut inner_face = FaceIdx::new(0);
            for octagon in poly.ngons(8) {
                inner_face = poly.excavate_cupola(octagon, false);
            }
            let inner = Polyhedron::rhombicuboctahedron();
            poly.color_faces_added_by("Inner", |poly| {
                poly.excavate(inner_face, &inner, inner.get_ngon(4), 0);
            });
            poly
        }),
        Model::new("K_4 (tunnel hexagons)", {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut inner_face = FaceIdx::new(0);
            for hexagon in poly.ngons(6) {
                inner_face = poly.excavate_cupola(hexagon, true);
            }
            let inner = Polyhedron::rhombicuboctahedron();
            poly.color_faces_added_by("Inner", |poly| {
                poly.excavate(inner_face, &inner, inner.get_ngon(3), 0);
            });
            poly
        }),
        Model::new("K_4 (tunnel cubes)", {
            let mut poly = Polyhedron::great_rhombicuboctahedron();
            let mut inner_face = FaceIdx::new(0);
            for square in poly.ngons(4) {
                inner_face = poly.excavate_prism(square);
            }
            let inner = Polyhedron::rhombicuboctahedron();
            let face = *inner.ngons(4).last().unwrap();
            poly.color_faces_added_by("Inner", |poly| {
                poly.excavate(inner_face, &inner, face, 0);
            });
            poly
        }),
        {
            let mut poly = Polyhedron::great_rhombicosidodecahedron();
            for face_idx in poly.face_indices() {
                if poly.face_order(face_idx) != 10 {
                    poly.color_face(face_idx, "Outer");
                }
            }
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, true);
                inner_face = poly.excavate_antiprism(next);
            }
            let mut inner = Polyhedron::rhombicosidodecahedron();
            for face_idx in inner.face_indices() {
                if inner.face_order(face_idx) != 5 {
                    inner.color_face(face_idx, "Inner");
                }
            }
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
            Model::new("K_5 (cupola/antiprism)", poly)
        },
        Model::new("K_5 (rotunda)", {
            let mut poly = Polyhedron::great_rhombicosidodecahedron();
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                inner_face = poly.excavate_rotunda(decagon, true);
            }
            let inner = Polyhedron::rhombicosidodecahedron();
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
            poly
        }),
        Model::new("Stephanie", {
            // Start with a colored truncated dodecahedron
            let mut poly = Polyhedron::truncated_dodecahedron();
            // Excavate using cupolae and antiprisms to form the tunnels
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, false);
                inner_face = poly.excavate_antiprism(next);
            }
            // Excavate the central cavity, and color these edges
            let inner = Polyhedron::dodecahedron();
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);
            poly
        }),
        {
            // Start with a colored truncated dodecahedron
            let mut poly = Polyhedron::truncated_dodecahedron();
            poly.color_all_edges("Outer");
            // Excavate using cupolae and antiprisms to form the tunnels
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, false);
                inner_face = poly.excavate_antiprism(next);
            }
            // Excavate the central cavity, and color these edges
            let mut inner = Polyhedron::dodecahedron();
            inner.color_all_edges("Inner");
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);

            Model::new("Stephanie (Coloring A)", poly)
        },
        {
            // Start with a colored truncated dodecahedron
            let mut poly = Polyhedron::truncated_dodecahedron();
            for tri in poly.ngons(3) {
                poly.color_face(tri, "Outer");
            }
            // Excavate using cupolae and antiprisms to form the tunnels
            let mut inner_face = FaceIdx::new(0);
            for decagon in poly.ngons(10) {
                let next = poly.excavate_cupola(decagon, false);
                inner_face = poly.excavate_antiprism(next);
            }
            // Excavate the central cavity, and color these edges
            let inner = Polyhedron::dodecahedron();
            poly.excavate(inner_face, &inner, inner.get_ngon(5), 0);

            Model::new("Stephanie (Coloring B)", poly)
        },
        {
            let mut poly = Polyhedron::truncated_icosahedron();
            for face in poly.ngons(5) {
                poly.color_face(face, "Pentagons");
            }

            Model::new("Football", poly)
        },
        Model::new("Cube Box (Color A)", cube_box_col_a(false)),
        Model::new("Cube Box (Color B)", cube_box_col_a(true)),
    ];

    toroids.to_vec()
}

fn q4_q4_slash_b4() -> Polyhedron {
    let PrismLike {
        mut poly,
        bottom_face,
        top_face,
    } = Polyhedron::cupola(4);
    poly.extend_cupola(bottom_face, true);
    poly.excavate_cuboctahedron(top_face);
    poly
}

fn flying_saucer() -> Polyhedron {
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
}

fn cube_box_col_a(use_concave_color: bool) -> Polyhedron {
    // Start with a central cube, which we'll use for the bottom-back-left corner
    let Cube {
        poly: cube,
        left,
        right,
        top,
        bottom,
        front,
        back,
    } = Polyhedron::cube();

    // Original cube becomes bottom-back-left central
    let mut poly = cube.clone();
    macro_rules! extend_colored {
        ($face: expr) => {{
            extend_prism_with_axis_color(&mut poly, $face)
        }};
    }
    extend_colored!(left);
    extend_colored!(back);
    extend_colored!(bottom);
    // Bottom-back-right
    let new_right = extend_colored!(right);
    let bbr_face_map = poly.extend(new_right, &cube, left, 0);
    extend_colored!(bbr_face_map[back]);
    extend_colored!(bbr_face_map[right]);
    extend_colored!(bbr_face_map[bottom]);
    // Top-back-left
    let new_top = extend_colored!(top);
    let tbl_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tbl_face_map[back]);
    extend_colored!(tbl_face_map[left]);
    extend_colored!(tbl_face_map[top]);
    // Bottom-front-left
    let new_front = extend_colored!(front);
    let bfl_face_map = poly.extend(new_front, &cube, back, 0);
    extend_colored!(bfl_face_map[front]);
    extend_colored!(bfl_face_map[left]);
    extend_colored!(bfl_face_map[bottom]);
    // Top-back-right
    let new_top = extend_colored!(bbr_face_map[top]);
    let tbr_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tbr_face_map[left]); // Links to top-back-left
    extend_colored!(tbr_face_map[back]);
    extend_colored!(tbr_face_map[right]);
    extend_colored!(tbr_face_map[top]);
    // Top-front-left
    let new_top = extend_colored!(bfl_face_map[top]);
    let tfl_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tfl_face_map[back]); // Links to top-back-left
    extend_colored!(tfl_face_map[front]);
    extend_colored!(tfl_face_map[left]);
    extend_colored!(tfl_face_map[top]);
    // Bottom-front-right
    let new_right = extend_colored!(bfl_face_map[right]);
    let bfr_face_map = poly.extend(new_right, &cube, left, 0);
    extend_colored!(bfr_face_map[back]); // Links to bottom-back-right
    extend_colored!(bfr_face_map[front]);
    extend_colored!(bfr_face_map[right]);
    extend_colored!(bfr_face_map[bottom]);
    // Top-front-right
    let new_top = extend_colored!(bfr_face_map[top]);
    let tfr_face_map = poly.extend(new_top, &cube, bottom, 0);
    extend_colored!(tfr_face_map[back]); // Links to top-back-right
    extend_colored!(tfr_face_map[left]); // Links to top-front-left
    extend_colored!(tfr_face_map[front]);
    extend_colored!(tfr_face_map[right]);
    extend_colored!(tfr_face_map[top]);

    // Color concave edges black
    if use_concave_color {
        for e in poly.edges() {
            if e.is_concave() {
                poly.reset_full_edge_color(e.top_vert, e.bottom_vert);
            }
        }
    }

    poly.make_centred();
    poly
}

fn extend_prism_with_axis_color(poly: &mut Polyhedron, face: FaceIdx) -> FaceIdx {
    // Determine which axis the face is in
    let normal = poly.get_face(face).normal(poly);
    let color = Axis::exact_axis(normal).unwrap().name();
    poly.color_faces_added_by(color, |poly| poly.extend_prism(face))
}

fn full_builtin_name(group_name: &str, model_name: &str) -> String {
    format!("Built-in\\{}\\{}", group_name, model_name)
}
