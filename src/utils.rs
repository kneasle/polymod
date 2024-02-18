use three_d::{egui, Angle, InnerSpace, Radians, Srgba, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Out,
    In,
}

pub fn ngon_name(n: usize) -> String {
    let name = match n {
        0..=2 => panic!("No n-gon of size {n}"),
        3 => "triangle",
        4 => "quad",
        5 => "pentagon",
        6 => "hexagon",
        7 => "heptagon",
        8 => "octagon",
        9 => "nonagon",
        10 => "decagon",
        _ => return format!("{n}-gon"),
    };
    name.to_owned()
}

/// Normalize a given vector `v` such that its length is `1` when projected perpendicular to
/// the given `direction`.
pub fn normalize_perpendicular_to(v: Vec3, direction: Vec3) -> Vec3 {
    let dist_along_edge_squared = v.project_on(direction).magnitude2();
    let perpendicular_distance = f32::sqrt(1.0 - dist_along_edge_squared);
    let normalized = v / perpendicular_distance;
    normalized
}

/// Compute the interior angle of a spherical triangles whos side lengths are `a`, `b` and `c`.
/// This calculates the interior angle opposite to the side of length `a`.
pub fn angle_in_spherical_triangle(a: Radians, b: Radians, c: Radians) -> Radians {
    // Formula is based on the Spherical Cosine Law
    let cos_angle = (a.cos() - b.cos() * c.cos()) / (b.sin() * c.sin());
    Radians::acos(cos_angle)
}

pub fn srgba_to_egui_color(c: Srgba) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c.r, c.g, c.b, c.a)
}

pub fn lerp3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}
