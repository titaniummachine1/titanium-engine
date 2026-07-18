//! Experimental wall-ignorance forced-loss certificate (Titanium v15 experimental).
//!
//! Feature-gated via `TITANIUM_WALL_IGNORE_LOSS_CERT` (default off).

use crate::core::board::{Board, Player};
use crate::titanium::cert_bridge::{paths_overlap, titanium_game_from_board};
use crate::titanium::game::GameState;
use crate::titanium::race::RACE_WIN_FLOOR;
use crate::titanium::wall_ignore_corridor::{
    detect_zero_delay_corridor, shortest_distance, CorridorScratch, RunnerGuarantee,
};
use std::sync::atomic::{AtomicU64, Ordering};

pub const FEATURE_ENV: &str = "TITANIUM_WALL_IGNORE_LOSS_CERT";
pub const TRACE_ENV: &str = "TITANIUM_WALL_IGNORE_CERT_TRACE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceInteraction {
    NonInteracting,
    Deterministic,
    Volatile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertSource {
    WallIgnoranceCorridor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WallIgnoreVerdict {
    pub winner: usize,
    pub winner_terminal_ply: u16,
    pub loser_terminal_ply: u16,
    pub source: CertSource,
    pub interaction: RaceInteraction,
    pub race_minimax_used: bool,
}

#[derive(Default, Debug)]
pub struct WallIgnoreStats {
    pub detector_calls: u64,
    pub corridors_found: u64,
    pub certificates_emitted: u64,
    pub path_edge_checks: u64,
    pub detector_nanos: u64,
}

pub static WALL_IGNORE_STATS: WallIgnoreStatsAtomic = WallIgnoreStatsAtomic::new();

pub struct WallIgnoreStatsAtomic {
    pub detector_calls: AtomicU64,
    pub corridors_found: AtomicU64,
    pub certificates_emitted: AtomicU64,
    pub path_edge_checks: AtomicU64,
    pub detector_nanos: AtomicU64,
}

impl WallIgnoreStatsAtomic {
    pub const fn new() -> Self {
        Self {
            detector_calls: AtomicU64::new(0),
            corridors_found: AtomicU64::new(0),
            certificates_emitted: AtomicU64::new(0),
            path_edge_checks: AtomicU64::new(0),
            detector_nanos: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> WallIgnoreStats {
        WallIgnoreStats {
            detector_calls: self.detector_calls.load(Ordering::Relaxed),
            corridors_found: self.corridors_found.load(Ordering::Relaxed),
            certificates_emitted: self.certificates_emitted.load(Ordering::Relaxed),
            path_edge_checks: self.path_edge_checks.load(Ordering::Relaxed),
            detector_nanos: self.detector_nanos.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.detector_calls.store(0, Ordering::Relaxed);
        self.corridors_found.store(0, Ordering::Relaxed);
        self.certificates_emitted.store(0, Ordering::Relaxed);
        self.path_edge_checks.store(0, Ordering::Relaxed);
        self.detector_nanos.store(0, Ordering::Relaxed);
    }
}

#[inline]
pub fn wall_ignore_loss_cert_enabled() -> bool {
    std::env::var(FEATURE_ENV)
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[inline]
pub fn wall_ignore_cert_trace_enabled() -> bool {
    std::env::var(TRACE_ENV)
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[inline]
pub fn earliest_terminal_ply(side: usize, side_to_move: usize, distance: u8) -> u16 {
    if distance == 0 {
        return 0;
    }
    let moves_first = side == side_to_move;
    2 * distance as u16 - u16::from(moves_first)
}

pub struct CertScratch {
    pub corridor: CorridorScratch,
}

impl Default for CertScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl CertScratch {
    pub fn new() -> Self {
        Self {
            corridor: CorridorScratch::new(),
        }
    }
}

fn classify_race_interaction(
    g: &GameState,
    _guarantee: &RunnerGuarantee,
    _loser: usize,
) -> RaceInteraction {
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);
    if paths_overlap(g, &d0, &d1) {
        let adj = crate::titanium::cert_bridge::turn_adjusted_tempo_advantage(g);
        if adj.abs() >= 2 {
            RaceInteraction::Deterministic
        } else {
            RaceInteraction::Volatile
        }
    } else {
        RaceInteraction::NonInteracting
    }
}

fn direct_wall_ignore_verdict(
    g: &GameState,
    winner: usize,
    guarantee: &RunnerGuarantee,
) -> Option<WallIgnoreVerdict> {
    let loser = 1 - winner;
    let winner_ply = earliest_terminal_ply(winner, g.turn, guarantee.max_own_moves_to_goal);
    let loser_dist = shortest_distance(g, loser);
    if loser_dist == 255 {
        return None;
    }
    let loser_ply = earliest_terminal_ply(loser, g.turn, loser_dist);
    if winner_ply >= loser_ply {
        return None;
    }
    Some(WallIgnoreVerdict {
        winner,
        winner_terminal_ply: winner_ply,
        loser_terminal_ply: loser_ply,
        source: CertSource::WallIgnoranceCorridor,
        interaction: RaceInteraction::NonInteracting,
        race_minimax_used: false,
    })
}

fn trace_rejection(
    reason: &str,
    g: &GameState,
    winner: Option<usize>,
    interaction: Option<RaceInteraction>,
) {
    if !wall_ignore_cert_trace_enabled() {
        return;
    }
    eprintln!(
        "wall_ignore_certificate: reject={reason} turn={} p0={} p1={} wl=({}, {}) winner={winner:?} interaction={interaction:?}",
        g.turn, g.pawn[0], g.pawn[1], g.wl[0], g.wl[1]
    );
}

fn trace_verdict(
    g: &GameState,
    guarantee: &RunnerGuarantee,
    verdict: &WallIgnoreVerdict,
    interaction: RaceInteraction,
) {
    if !wall_ignore_cert_trace_enabled() {
        return;
    }
    eprintln!(
        "wall_ignore_certificate: winner={} loser={} path={:?} edges={:?} w_dist={} w_ply={} l_dist={} l_ply={} interaction={interaction:?} race_minimax={} verdict=ForcedWin",
        verdict.winner,
        1 - verdict.winner,
        guarantee.path,
        guarantee.protected_edges,
        guarantee.max_own_moves_to_goal,
        verdict.winner_terminal_ply,
        shortest_distance(g, 1 - verdict.winner),
        verdict.loser_terminal_ply,
        verdict.race_minimax_used,
    );
}

/// Core detector + race check on a [`GameState`].
pub fn try_wall_ignorance_loss_cert(
    g: &mut GameState,
    scratch: &mut CertScratch,
    force_enable: bool,
) -> Option<WallIgnoreVerdict> {
    if !force_enable && !wall_ignore_loss_cert_enabled() {
        return None;
    }
    if g.winner() >= 0 {
        return None;
    }

    let t0 = std::time::Instant::now();
    WALL_IGNORE_STATS
        .detector_calls
        .fetch_add(1, Ordering::Relaxed);

    for winner in [0usize, 1] {
        let Some(guarantee) = detect_zero_delay_corridor(g, winner, &mut scratch.corridor) else {
            continue;
        };
        WALL_IGNORE_STATS
            .corridors_found
            .fetch_add(1, Ordering::Relaxed);
        WALL_IGNORE_STATS
            .path_edge_checks
            .fetch_add(guarantee.protected_edges.len() as u64, Ordering::Relaxed);

        let loser = 1 - winner;
        let interaction = classify_race_interaction(g, &guarantee, loser);

        let verdict = match interaction {
            RaceInteraction::NonInteracting => direct_wall_ignore_verdict(g, winner, &guarantee),
            RaceInteraction::Deterministic | RaceInteraction::Volatile => {
                trace_rejection("not-non-interacting", g, Some(winner), Some(interaction));
                None
            }
        };

        if let Some(ref v) = verdict {
            if v.winner_terminal_ply >= v.loser_terminal_ply {
                trace_rejection("equal-or-later-winner", g, Some(winner), Some(interaction));
                continue;
            }
            trace_verdict(g, &guarantee, v, interaction);
            WALL_IGNORE_STATS
                .certificates_emitted
                .fetch_add(1, Ordering::Relaxed);
            WALL_IGNORE_STATS
                .detector_nanos
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return Some(v.clone());
        }
    }

    WALL_IGNORE_STATS
        .detector_nanos
        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    None
}

/// Board-facing entry (converts to throwaway [`GameState`]).
pub fn try_wall_ignore_cert_board(board: &Board, force_enable: bool) -> Option<WallIgnoreVerdict> {
    let mut g = titanium_game_from_board(board);
    let mut scratch = CertScratch::new();
    try_wall_ignorance_loss_cert(&mut g, &mut scratch, force_enable)
}

/// Proven-outcome bound from the side-to-move perspective.
///
/// `winner_terminal_ply` is a guaranteed arrival estimate, not exact DTM, so
/// this must stay outside both the mate and exact-race score bands.
#[inline]
pub fn cert_score_from_stm(verdict: &WallIgnoreVerdict, stm: usize) -> i32 {
    if verdict.winner == stm {
        RACE_WIN_FLOOR
    } else {
        -RACE_WIN_FLOOR
    }
}

#[inline]
pub fn cert_score_from_player(verdict: &WallIgnoreVerdict, player: Player) -> i32 {
    cert_score_from_stm(verdict, player as usize)
}

/// Compare with an existing winner-side certificate; debug builds assert agreement.
pub fn assert_agrees_with_existing(existing_winner: Player, verdict: &WallIgnoreVerdict) {
    debug_assert_eq!(
        existing_winner as usize, verdict.winner,
        "wall-ignore cert disagrees with existing certificate"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game_with_pawns(p0: usize, p1: usize, turn: usize, wl: (i32, i32)) -> GameState {
        let mut g = GameState::new();
        g.pawn = [p0, p1];
        g.turn = turn;
        g.wl = [wl.0, wl.1];
        g
    }

    /// Column-4 corridor fixture with configurable wall counts.
    fn corridor_game(wl0: i32, wl1: i32) -> GameState {
        let mut g = crate::titanium::wall_ignore_corridor::build_column_four_corridor_fixture();
        g.wl = [wl0, wl1];
        g
    }

    #[test]
    fn one_tempo_forced_loss_white_wins() {
        let g = corridor_game(10, 10);
        let mut scratch = CertScratch::new();
        let v = try_wall_ignorance_loss_cert(&mut g.clone(), &mut scratch, true).expect("cert");
        assert_eq!(v.winner, 0);
        assert!(v.winner_terminal_ply < v.loser_terminal_ply);
    }

    #[test]
    fn arbitrary_loser_wall_count_invariant() {
        for wl1 in [0, 1, 3, 7, 10] {
            let mut g = corridor_game(5, wl1);
            let mut scratch = CertScratch::new();
            let v =
                try_wall_ignorance_loss_cert(&mut g, &mut scratch, true).expect("cert wl1={wl1}");
            assert_eq!(v.winner, 0);
            assert!(v.winner_terminal_ply < v.loser_terminal_ply);
        }
    }

    #[test]
    fn arbitrary_winner_wall_count_invariant() {
        for wl0 in [0, 1, 5, 10] {
            let mut g = corridor_game(wl0, 10);
            let mut scratch = CertScratch::new();
            let v =
                try_wall_ignorance_loss_cert(&mut g, &mut scratch, true).expect("cert wl0={wl0}");
            assert_eq!(v.winner, 0);
        }
    }

    #[test]
    fn winner_zero_walls_still_certifies() {
        let mut g = corridor_game(0, 10);
        let mut scratch = CertScratch::new();
        assert!(try_wall_ignorance_loss_cert(&mut g, &mut scratch, true).is_some());
    }

    #[test]
    fn equal_arrival_no_direct_certificate() {
        // Equal distance 4 both sides, white to move → both ply 7.
        let mut g = game_with_pawns(5 * 9 + 1, 5 * 9 + 7, 0, (10, 10));
        let mut scratch = CertScratch::new();
        assert!(
            try_wall_ignorance_loss_cert(&mut g, &mut scratch, true).is_none(),
            "equal terminal ply must not direct-certify"
        );
    }

    #[test]
    fn candidate_winner_later_no_certificate() {
        // White far, black close — no forced white win.
        let mut g = game_with_pawns(8 * 9 + 4, 2 * 9 + 4, 0, (10, 10));
        let mut scratch = CertScratch::new();
        assert!(try_wall_ignorance_loss_cert(&mut g, &mut scratch, true).is_none());
    }

    #[test]
    fn side_to_move_tempo_differs() {
        let g_stm0 = corridor_game(10, 10);
        let mut g_stm1 = corridor_game(10, 10);
        g_stm1.turn = 1;
        let mut s0 = CertScratch::new();
        let mut s1 = CertScratch::new();
        let v0 = try_wall_ignorance_loss_cert(&mut g_stm0.clone(), &mut s0, true).expect("w stm");
        let v1 = try_wall_ignorance_loss_cert(&mut g_stm1, &mut s1, true);
        assert_eq!(v0.winner, 0);
        if let Some(v1) = v1 {
            assert_ne!(
                v0.winner, v1.winner,
                "side-to-move flip must change which side the certificate favors"
            );
        }
    }

    #[test]
    fn shared_path_no_raw_distance_certificate() {
        let g = game_with_pawns(6 * 9 + 4, 5 * 9 + 4, 0, (0, 0));
        let mut scratch = CertScratch::new();
        assert!(try_wall_ignorance_loss_cert(&mut g.clone(), &mut scratch, true).is_none());
    }

    #[test]
    fn feature_disabled_returns_none_without_env() {
        let g = corridor_game(10, 10);
        let mut scratch = CertScratch::new();
        let prev = std::env::var(FEATURE_ENV).ok();
        std::env::remove_var(FEATURE_ENV);
        assert!(try_wall_ignorance_loss_cert(&mut g.clone(), &mut scratch, false).is_none());
        if let Some(v) = prev {
            std::env::set_var(FEATURE_ENV, v);
        }
    }

    #[test]
    fn earliest_terminal_ply_examples() {
        assert_eq!(earliest_terminal_ply(0, 0, 1), 1);
        assert_eq!(earliest_terminal_ply(0, 1, 1), 2);
        assert_eq!(earliest_terminal_ply(0, 0, 4), 7);
        assert_eq!(earliest_terminal_ply(0, 1, 4), 8);
    }

    #[test]
    fn cert_score_is_a_bound_and_does_not_fake_dtm_ordering() {
        let fast = WallIgnoreVerdict {
            winner: 0,
            winner_terminal_ply: 2,
            loser_terminal_ply: 5,
            source: CertSource::WallIgnoranceCorridor,
            interaction: RaceInteraction::NonInteracting,
            race_minimax_used: false,
        };
        let slow = WallIgnoreVerdict {
            winner: 0,
            winner_terminal_ply: 5,
            loser_terminal_ply: 8,
            source: CertSource::WallIgnoranceCorridor,
            interaction: RaceInteraction::NonInteracting,
            race_minimax_used: false,
        };
        assert_eq!(cert_score_from_stm(&fast, 0), RACE_WIN_FLOOR);
        assert_eq!(cert_score_from_stm(&slow, 0), RACE_WIN_FLOOR);
        assert_eq!(cert_score_from_stm(&fast, 1), -RACE_WIN_FLOOR);
        assert!(fast.winner_terminal_ply < slow.winner_terminal_ply);
    }
}
