use std::cmp::Ordering;

use ordered_float::OrderedFloat;
use three_d::{
    egui::{self, Color32},
    Angle, InnerSpace, Radians, Srgba, Vec3,
};

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

#[derive(Debug, Clone, Copy)]
pub struct PolygonGeom {
    pub angle: f32,
    pub in_radius: f32,
    pub out_radius: f32,
}

impl PolygonGeom {
    pub fn new(n: usize) -> Self {
        let angle = std::f32::consts::PI * 2.0 / n as f32;
        let in_radius = 1.0 / (2.0 * f32::tan(angle / 2.0));
        let out_radius = 1.0 / (2.0 * f32::sin(angle / 2.0));
        Self {
            angle,
            in_radius,
            out_radius,
        }
    }

    pub fn point(&self, i: usize) -> (f32, f32) {
        self.offset_point(i, 0.0)
    }

    pub fn offset_point(&self, i: usize, offset: f32) -> (f32, f32) {
        let a = self.angle * (i as f32 + offset);
        let x = a.sin() * self.out_radius;
        let y = a.cos() * self.out_radius;
        (x, y)
    }
}

pub fn darken_color(c: Color32, factor: f32) -> Color32 {
    lerp_color(Color32::BLACK, c, factor)
}

pub fn lerp_color(a: Color32, b: Color32, factor: f32) -> Color32 {
    let lerp = |a: u8, b: u8| -> u8 {
        let lerped_f32 = (a as f32) * (1.0 - factor) + (b as f32) * factor;
        lerped_f32 as u8
    };
    Color32::from_rgba_premultiplied(
        lerp(a.r(), b.r()),
        lerp(a.g(), b.g()),
        lerp(a.b(), b.b()),
        lerp(a.a(), b.a()),
    )
}

pub fn egui_color_to_srgba(c: Color32) -> Srgba {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    Srgba { r, g, b, a }
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
