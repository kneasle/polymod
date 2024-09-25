//! Module storing a set of standard colours used in the GUI and model view
//!
//! Colors are the 'Mocha' theme from:
//! https://github.com/catppuccin/catppuccin?tab=readme-ov-file#-palette

#![expect(dead_code)]

use three_d::egui::Color32;

pub const BASE: Color32 = Color32::from_rgb(30, 30, 46);
pub const MANTLE: Color32 = Color32::from_rgb(24, 24, 37);
pub const CRUST: Color32 = Color32::from_rgb(17, 17, 27);

pub const TEXT: Color32 = Color32::from_rgb(205, 214, 244);
pub const SUBTEXT1: Color32 = Color32::from_rgb(186, 194, 222);
pub const SUBTEXT0: Color32 = Color32::from_rgb(166, 173, 200);
pub const OVERLAY2: Color32 = Color32::from_rgb(147, 153, 178);
pub const OVERLAY1: Color32 = Color32::from_rgb(127, 132, 156);
pub const OVERLAY0: Color32 = Color32::from_rgb(108, 112, 134);
pub const SURFACE2: Color32 = Color32::from_rgb(88, 91, 112);
pub const SURFACE1: Color32 = Color32::from_rgb(69, 71, 90);
pub const SURFACE0: Color32 = Color32::from_rgb(49, 50, 68);

pub const ROSEWATER: Color32 = Color32::from_rgb(245, 224, 220);
pub const FLAMINGO: Color32 = Color32::from_rgb(242, 205, 205);
pub const PINK: Color32 = Color32::from_rgb(245, 194, 231);

pub const RED: Color32 = Color32::from_rgb(243, 139, 168);
pub const MAROON: Color32 = Color32::from_rgb(235, 160, 172);
pub const PEACH: Color32 = Color32::from_rgb(250, 179, 135);
pub const YELLOW: Color32 = Color32::from_rgb(249, 226, 175);
pub const GREEN: Color32 = Color32::from_rgb(166, 227, 161);
pub const TEAL: Color32 = Color32::from_rgb(148, 226, 213);
pub const SKY: Color32 = Color32::from_rgb(137, 220, 235);
pub const SAPPHIRE: Color32 = Color32::from_rgb(116, 199, 236);
pub const BLUE: Color32 = Color32::from_rgb(137, 180, 250);
pub const LAVENDER: Color32 = Color32::from_rgb(180, 190, 254);
pub const MAUVE: Color32 = Color32::from_rgb(203, 166, 247);
