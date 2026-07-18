//! ASCII visualization + invariant checks for NNUE distance / corridor fields.
//!
//! Run: `titanium fields [moves...]` or `python training/visualize_fields.py e2 e8 ...`

use crate::titanium::dist::{
    fill_ace_dist_from_pawn, fill_ace_dist_to_goal, fill_choke_points, fill_contested,
    fill_corridor_delta, fill_path_crossing,
};
use crate::titanium::game::GameState;
use crate::util::grid::square_index;

/// All per-cell geometry the HalfPW net consumes (ACE cell index, row 0 = top).
/// Field names match `field_planes.rs` / `training/field_planes.py`.
#[derive(Clone, Debug)]
pub struct NnueFields {
    pub d0_scalar: u8,
    pub d1_scalar: u8,
    pub goal_inv_p0: [u8; 81],
    pub goal_inv_p1: [u8; 81],
    pub pawn_fwd_p0: [u8; 81],
    pub pawn_fwd_p1: [u8; 81],
    pub corridor_delta_p0: [u8; 81],
    pub corridor_delta_p1: [u8; 81],
    /// Optional: path pinch / merge points (viz only — not in net eval).
    pub choke0: [u8; 81],
    pub choke1: [u8; 81],
    pub path_cross_p0: [u8; 81],
    pub path_cross_p1: [u8; 81],
    pub contested: [u8; 81],
}

pub fn compute_nnue_fields(g: &GameState) -> NnueFields {
    let mut goal_inv_p0 = [255u8; 81];
    let mut goal_inv_p1 = [255u8; 81];
    fill_ace_dist_to_goal(g, 0, &mut goal_inv_p0);
    fill_ace_dist_to_goal(g, 1, &mut goal_inv_p1);
    let d0_scalar = goal_inv_p0[g.pawn[0]];
    let d1_scalar = goal_inv_p1[g.pawn[1]];

    let mut pawn_fwd_p0 = [255u8; 81];
    let mut pawn_fwd_p1 = [255u8; 81];
    fill_ace_dist_from_pawn(g, g.pawn[0], &mut pawn_fwd_p0);
    fill_ace_dist_from_pawn(g, g.pawn[1], &mut pawn_fwd_p1);

    let mut corridor_delta_p0 = [255u8; 81];
    let mut corridor_delta_p1 = [255u8; 81];
    fill_corridor_delta(
        &pawn_fwd_p0,
        &goal_inv_p0,
        d0_scalar,
        &mut corridor_delta_p0,
    );
    fill_corridor_delta(
        &pawn_fwd_p1,
        &goal_inv_p1,
        d1_scalar,
        &mut corridor_delta_p1,
    );

    let mut choke0 = [0u8; 81];
    let mut choke1 = [0u8; 81];
    fill_choke_points(g, &pawn_fwd_p0, &goal_inv_p0, d0_scalar, &mut choke0);
    fill_choke_points(g, &pawn_fwd_p1, &goal_inv_p1, d1_scalar, &mut choke1);
    let mut path_cross_p0 = [0u8; 81];
    let mut path_cross_p1 = [0u8; 81];
    fill_path_crossing(g, &pawn_fwd_p0, &goal_inv_p0, d0_scalar, &mut path_cross_p0);
    fill_path_crossing(g, &pawn_fwd_p1, &goal_inv_p1, d1_scalar, &mut path_cross_p1);
    let mut contested = [0u8; 81];
    fill_contested(&corridor_delta_p0, &corridor_delta_p1, &mut contested);

    NnueFields {
        d0_scalar,
        d1_scalar,
        goal_inv_p0,
        goal_inv_p1,
        pawn_fwd_p0,
        pawn_fwd_p1,
        corridor_delta_p0,
        corridor_delta_p1,
        choke0,
        choke1,
        path_cross_p0,
        path_cross_p1,
        contested,
    }
}

fn cell_char(v: u8, on_shortest: bool) -> char {
    if v == 255 {
        return '·';
    }
    if on_shortest {
        return match v {
            0..=9 => b"0123456789"[v as usize] as char,
            _ => '+',
        };
    }
    match v {
        0..=9 => b"0123456789"[v as usize] as char,
        10..=19 => (b'a' + (v - 10)) as char,
        _ => '*',
    }
}

fn render_grid(
    g: &GameState,
    field: &[u8; 81],
    highlight_delta: Option<&[u8; 81]>,
    title: &str,
    out: &mut String,
) {
    out.push_str(title);
    out.push('\n');
    for row in (0..9u8).rev() {
        out.push_str(&format!("{row} "));
        for col in 0..9u8 {
            let sq = square_index(row, col) as usize;
            let ch = if g.pawn[0] == sq {
                '@'
            } else if g.pawn[1] == sq {
                '&'
            } else {
                let on_path = highlight_delta
                    .map(|d| d[sq] != 255 && d[sq] <= 1)
                    .unwrap_or(false);
                cell_char(field[sq], on_path)
            };
            out.push(ch);
            out.push(' ');
        }
        out.push('\n');
    }
    out.push('\n');
}

fn render_choke_grid(g: &GameState, field: &[u8; 81], title: &str, out: &mut String) {
    out.push_str(title);
    out.push('\n');
    for row in (0..9u8).rev() {
        out.push_str(&format!("{row} "));
        for col in 0..9u8 {
            let sq = square_index(row, col) as usize;
            let ch = if g.pawn[0] == sq {
                '@'
            } else if g.pawn[1] == sq {
                '&'
            } else if field[sq] != 0 {
                let v = field[sq];
                if v >= 16 {
                    '!' // fully forced (0 continuations)
                } else {
                    b"0123456789"[(v * 10 / 16).min(9) as usize] as char
                }
            } else {
                '.'
            };
            out.push(ch);
            out.push(' ');
        }
        out.push('\n');
    }
    out.push('\n');
}

fn render_cross_grid(g: &GameState, field: &[u8; 81], title: &str, out: &mut String) {
    out.push_str(title);
    out.push('\n');
    for row in (0..9u8).rev() {
        out.push_str(&format!("{row} "));
        for col in 0..9u8 {
            let sq = square_index(row, col) as usize;
            let ch = if g.pawn[0] == sq {
                '@'
            } else if g.pawn[1] == sq {
                '&'
            } else if field[sq] == 255 {
                '·'
            } else if field[sq] == 0 {
                '.'
            } else if field[sq] < 10 {
                (b'0' + field[sq]) as char
            } else {
                '+'
            };
            out.push(ch);
            out.push(' ');
        }
        out.push('\n');
    }
    out.push('\n');
}

/// Multi-path summary: cells on/near shortest-route corridor (delta ≤ 1).
fn corridor_summary(delta: &[u8; 81], player: usize) -> String {
    let mut d0 = 0usize;
    let mut d1 = 0usize;
    let mut wider = 0usize;
    for &v in delta {
        if v == 255 {
            continue;
        }
        if v == 0 {
            d0 += 1;
        } else if v == 1 {
            d1 += 1;
        } else {
            wider += 1;
        }
    }
    format!(
        "P{player}: delta=0 (on shortest family): {d0} cells | delta=1: {d1} | delta≥2: {wider}"
    )
}

pub fn render_fields_text(g: &GameState, fields: &NnueFields) -> String {
    let mut out = String::new();
    out.push_str("Quoridor NNUE field planes (ACE coords, row 8 = bottom)\n");
    out.push_str("@ = P0 pawn  & = P1 pawn  · = unreachable  digits = distance\n");
    out.push_str(
        "Corridor delta: 0 = on some shortest route | 1 = one tempo longer route | 2+ = further off\n\n",
    );
    out.push_str(&format!(
        "Scalars: P0 shortest={}  P1 shortest={}  turn={}  walls P0/P1={}/{}\n\n",
        fields.d0_scalar, fields.d1_scalar, g.turn, g.wl[0], g.wl[1]
    ));

    render_grid(
        g,
        &fields.goal_inv_p0,
        Some(&fields.corridor_delta_p0),
        "P0 goal_inv_p0 (inverse BFS → P0 goal row 0)",
        &mut out,
    );
    render_grid(
        g,
        &fields.goal_inv_p1,
        Some(&fields.corridor_delta_p1),
        "P1 goal_inv_p1 (inverse BFS → P1 goal row 8)",
        &mut out,
    );
    render_grid(
        g,
        &fields.pawn_fwd_p0,
        Some(&fields.corridor_delta_p0),
        "P0 pawn_fwd_p0 (forward steps from pawn)",
        &mut out,
    );
    render_grid(
        g,
        &fields.pawn_fwd_p1,
        Some(&fields.corridor_delta_p1),
        "P1 pawn_fwd_p1 (forward steps from pawn)",
        &mut out,
    );
    render_grid(
        g,
        &fields.corridor_delta_p0,
        None,
        "P0 corridor_delta_p0 (from+to−shortest; 0 = on some shortest route)",
        &mut out,
    );
    render_grid(
        g,
        &fields.corridor_delta_p1,
        None,
        "P1 corridor_delta_p1 (from+to−shortest; 0 = on some shortest route)",
        &mut out,
    );
    render_choke_grid(
        g,
        &fields.choke0,
        "P0 choke_p0 (forcedness 1/(1+continuations); ! = dead-end)",
        &mut out,
    );
    render_choke_grid(
        g,
        &fields.choke1,
        "P1 choke_p1 (forcedness 1/(1+continuations); ! = dead-end)",
        &mut out,
    );
    render_cross_grid(
        g,
        &fields.path_cross_p0,
        "P0 path_cross_p0 (route count through cell)",
        &mut out,
    );
    render_cross_grid(
        g,
        &fields.path_cross_p1,
        "P1 path_cross_p1 (route count through cell)",
        &mut out,
    );
    render_choke_grid(
        g,
        &fields.contested,
        "contested (1/(1+delta_p0+delta_p1); both routes matter)",
        &mut out,
    );

    out.push_str(&corridor_summary(&fields.corridor_delta_p0, 0));
    out.push('\n');
    out.push_str(&corridor_summary(&fields.corridor_delta_p1, 1));
    out.push('\n');
    out
}

/// Invariant checks — returns human-readable error strings (empty = OK).
pub fn validate_fields(g: &GameState, f: &NnueFields) -> Vec<String> {
    let mut errs = Vec::new();

    if f.goal_inv_p0[g.pawn[0]] != f.d0_scalar {
        errs.push(format!(
            "P0 scalar mismatch: d0={} but goal_inv_p0[pawn]={}",
            f.d0_scalar, f.goal_inv_p0[g.pawn[0]]
        ));
    }
    if f.goal_inv_p1[g.pawn[1]] != f.d1_scalar {
        errs.push(format!(
            "P1 scalar mismatch: d1={} but goal_inv_p1[pawn]={}",
            f.d1_scalar, f.goal_inv_p1[g.pawn[1]]
        ));
    }
    if f.pawn_fwd_p0[g.pawn[0]] != 0 {
        errs.push("P0 pawn should have 0 forward steps".into());
    }
    if f.pawn_fwd_p1[g.pawn[1]] != 0 {
        errs.push("P1 pawn should have 0 forward steps".into());
    }
    if f.corridor_delta_p0[g.pawn[0]] != 0 {
        errs.push(format!(
            "P0 pawn corridor_delta_p0 should be 0, got {}",
            f.corridor_delta_p0[g.pawn[0]]
        ));
    }
    if f.corridor_delta_p1[g.pawn[1]] != 0 {
        errs.push(format!(
            "P1 pawn corridor_delta_p1 should be 0, got {}",
            f.corridor_delta_p1[g.pawn[1]]
        ));
    }

    for col in 0..9u8 {
        let p0_goal = square_index(0, col) as usize;
        if f.goal_inv_p0[p0_goal] != 0 {
            errs.push(format!(
                "P0 goal row cell {p0_goal} should have goal_inv_p0=0"
            ));
        }
        let p1_goal = square_index(8, col) as usize;
        if f.goal_inv_p1[p1_goal] != 0 {
            errs.push(format!(
                "P1 goal row cell {p1_goal} should have goal_inv_p1=0"
            ));
        }
    }

    for sq in 0..81usize {
        for (player, from, to, shortest, delta) in [
            (
                0,
                &f.pawn_fwd_p0,
                &f.goal_inv_p0,
                f.d0_scalar,
                &f.corridor_delta_p0,
            ),
            (
                1,
                &f.pawn_fwd_p1,
                &f.goal_inv_p1,
                f.d1_scalar,
                &f.corridor_delta_p1,
            ),
        ] {
            let fr = from[sq];
            let inv = to[sq];
            if fr == 255 || inv == 255 {
                if delta[sq] != 255 {
                    errs.push(format!(
                        "P{player} sq {sq}: unreachable but delta={}",
                        delta[sq]
                    ));
                }
                continue;
            }
            let expect = (u16::from(fr) + u16::from(inv)).saturating_sub(u16::from(shortest)) as u8;
            if delta[sq] != expect {
                errs.push(format!(
                    "P{player} sq {sq}: delta={d} expected {expect} (from={fr}+to={inv}-{shortest})",
                    d = delta[sq],
                ));
            }
        }
    }

    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::algebraic_to_move_id;

    fn pos(moves: &[&str]) -> GameState {
        let mut g = GameState::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
        }
        g
    }

    #[test]
    fn startpos_fields_sane() {
        let g = GameState::new();
        let f = compute_nnue_fields(&g);
        assert_eq!(validate_fields(&g, &f), vec![] as Vec<String>);
        assert!(
            f.corridor_delta_p0.iter().filter(|&&v| v == 0).count() > 1,
            "multiple P0 route cells"
        );
    }

    #[test]
    fn midgame_fields_sane() {
        let g = pos(&["e2", "e8", "e3", "e7", "d3h", "f5v"]);
        let f = compute_nnue_fields(&g);
        assert_eq!(validate_fields(&g, &f), vec![] as Vec<String>);
    }

    /// Walls must change distance fields — parallel flood reads wall topology via DirMasks.
    #[test]
    fn walls_change_shortest_and_corridor() {
        let open = pos(&["e2", "e8", "e3", "e7"]);
        let walled = pos(&["e2", "e8", "e3", "e7", "d4h"]);
        let f_open = compute_nnue_fields(&open);
        let f_wall = compute_nnue_fields(&walled);
        assert_eq!(validate_fields(&walled, &f_wall), vec![] as Vec<String>);
        assert!(
            f_wall.d0_scalar > f_open.d0_scalar || f_wall.d1_scalar > f_open.d1_scalar,
            "wall should lengthen at least one shortest path"
        );
        // Asymmetric detour: some cells gain delta=1 (alternate route one tempo slower).
        assert!(
            f_wall.corridor_delta_p0.iter().filter(|&&v| v == 1).count()
                > f_open.corridor_delta_p0.iter().filter(|&&v| v == 1).count(),
            "wall should create alternate-path (delta=1) cells"
        );
    }

    /// Net inputs: both players get goal field, pawn field, corridor field (6 plane sets).
    #[test]
    fn nnue_has_both_players_all_planes() {
        let g = pos(&["e2", "e8", "e3", "e7", "d3h", "f5v"]);
        let f = compute_nnue_fields(&g);
        assert!(f.goal_inv_p0.iter().any(|&v| v != 255));
        assert!(f.goal_inv_p1.iter().any(|&v| v != 255));
        assert!(f.pawn_fwd_p0.iter().any(|&v| v != 255));
        assert!(f.pawn_fwd_p1.iter().any(|&v| v != 255));
        assert!(f.corridor_delta_p0.iter().any(|&v| v == 0));
        assert!(f.corridor_delta_p1.iter().any(|&v| v == 0));
        assert!(
            f.corridor_delta_p0.iter().any(|&v| v == 1),
            "multi-path band"
        );
        assert!(
            f.corridor_delta_p1.iter().any(|&v| v == 1),
            "multi-path band"
        );
    }

    #[test]
    #[ignore]
    fn dump_midgame_viz() {
        let g = pos(&["e2", "e8", "e3", "e7", "d3h", "f5v", "c2h"]);
        let f = compute_nnue_fields(&g);
        eprintln!("{}", render_fields_text(&g, &f));
    }
}
