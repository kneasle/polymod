use std::cmp::Ordering;

use ordered_float::OrderedFloat;
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
    v / perpendicular_distance
}

/// Compute the interior angle of a spherical triangles whos side lengths are `a`, `b` and `c`.
/// This calculates the interior angle opposite to the side of length `a`.
pub fn angle_in_spherical_triangle(a: Radians, b: Radians, c: Radians) -> Radians {
    // Formula is based on the Spherical Cosine Law
    let cos_angle = (a.cos() - b.cos() * c.cos()) / (b.sin() * c.sin());
    Radians::acos(cos_angle)
}

pub fn darken_color(c: egui::Rgba, factor: f32) -> egui::Rgba {
    egui::Rgba::from_rgba_premultiplied(c.r() * factor, c.g() * factor, c.b() * factor, c.a())
}

pub fn lerp_color(a: egui::Rgba, b: egui::Rgba, factor: f32) -> egui::Rgba {
    a * (1.0 - factor) + b * factor
}

pub fn egui_color_to_srgba(c: egui::Rgba) -> Srgba {
    let [r, g, b, a] = c.to_rgba_unmultiplied();
    Srgba {
        r: (r * 255.0) as u8,
        g: (g * 255.0) as u8,
        b: (b * 255.0) as u8,
        a: (a * 255.0) as u8,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OrderedRgba(pub egui::Rgba);

impl OrderedRgba {
    fn as_ordered_floats(self) -> [OrderedFloat<f32>; 4] {
        self.0.to_array().map(OrderedFloat)
    }
}

impl PartialEq for OrderedRgba {
    fn eq(&self, other: &Self) -> bool {
        self.as_ordered_floats() == other.as_ordered_floats()
    }
}

impl Eq for OrderedRgba {}

impl PartialOrd for OrderedRgba {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedRgba {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ordered_floats().cmp(&other.as_ordered_floats())
    }
}

pub fn lerp3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}
