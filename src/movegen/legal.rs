//! Legal move generation — pawn jumps + wall placements with path validation.

use crate::core::board::{Board, Move, Player, WallOrientation};
use crate::movegen::o1::{
    generate_pawn_moves_lean_lut, generate_pawn_moves_o1, wall_masks, wall_physically_legal_o1,
};
use crate::movegen::pawn_bits::{
    generate_pawn_moves_bitboard_with_masks, generate_pawn_moves_shift_slice,
};
use crate::path::masks::DirMasks;
use crate::path::parallel::{pawn_bit, pbff_wall_legal, wall_delta, WallGrids};
use crate::path::BfsScratch;
use crate::util::grid::{can_step, has_wall};

const DIRS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// Upper bound on legal moves in any Quoridor position (startpos ≈ 131).
pub const MAX_LEGAL_MOVES: usize = 140;

/// Pawn-generation strategy — production uses [`PawnGenMode::O1Lookup`]; other modes for benches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PawnGenMode {
    /// ~4× `can_step` per node — no mask table.
    Scalar,
    /// Full-board `DirMasks::from_board` + bitmask axis logic.
    BitboardFreshDirMasks,
    /// Reuse `BfsScratch::dir_masks` — incorrect if stale after wall trials.
    BitboardCachedDirMasks,
    /// Blind bit shift + `can_step` wall check — no `DirMasks`.
    ShiftCanStep,
    /// Offline `PAWN_LEGAL` tables. **Production default** — fastest at perft(4)
    /// in both default and `target-cpu=native` (PEXT) builds, verified correct
    /// against the oracle. (Was research-only on `movgen-o1-lookup`; promoted
    /// once it beat shift/scalar at perft(4) with and without BMI2.)
    O1Lookup,
    /// Lean LUT: skip the table when no enemy is adjacent (ek=0 → ShiftCanStep),
    /// use O1 table only for jump/lateral special cases (ek≠0).
    O1LeanLut,
}

impl Default for PawnGenMode {
    fn default() -> Self {
        Self::O1Lookup
    }
}

fn generate_pawn_moves_with_mode(
    board: &Board,
    scratch: &mut BfsScratch,
    out: &mut [Move],
    mode: PawnGenMode,
) -> usize {
    match mode {
        PawnGenMode::Scalar => generate_pawn_moves_scalar_for(board, board.side_to_move, out),
        PawnGenMode::BitboardFreshDirMasks => {
            let masks = DirMasks::from_board(board);
            generate_pawn_moves_bitboard_with_masks(board, &masks, out)
        }
        PawnGenMode::BitboardCachedDirMasks => {
            let masks = scratch.dir_masks(board);
            generate_pawn_moves_bitboard_with_masks(board, &masks, out)
        }
        PawnGenMode::ShiftCanStep => generate_pawn_moves_shift_slice(board, out),
        PawnGenMode::O1Lookup => generate_pawn_moves_o1(board, out),
        PawnGenMode::O1LeanLut => generate_pawn_moves_lean_lut(board, out),
    }
}

pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    let mut copy = board.clone();
    let mut out = Vec::new();
    let mut scratch = BfsScratch::new();
    generate_legal_moves_into(&mut copy, &mut out, &mut scratch);
    out
}

/// Hot-path API — stack buffer in perft, zero heap allocs per node.
pub fn generate_legal_moves_slice(
    board: &mut Board,
    out: &mut [Move],
    scratch: &mut BfsScratch,
) -> usize {
    if board.is_terminal().is_some() {
        return 0;
    }

    generate_legal_moves_slice_mode(board, out, scratch, PawnGenMode::default())
}

/// Legal moves with a selectable pawn generator — wall path logic unchanged.
pub fn generate_legal_moves_slice_mode(
    board: &mut Board,
    out: &mut [Move],
    scratch: &mut BfsScratch,
    mode: PawnGenMode,
) -> usize {
    if board.is_terminal().is_some() {
        return 0;
    }

    let mut n = generate_pawn_moves_with_mode(board, scratch, out, mode);
    if board.walls_remaining[board.side_to_move as usize] > 0 {
        n += generate_wall_moves_slice(board, &mut out[n..], scratch);
    }
    debug_assert!(n <= MAX_LEGAL_MOVES);
    n
}

/// Pawn moves only — no wall enumeration, no BFS wall trials (mobility / pawn-only perft).
pub fn generate_pawn_moves_slice_mode(
    board: &Board,
    out: &mut [Move],
    scratch: &mut BfsScratch,
    mode: PawnGenMode,
) -> usize {
    if board.is_terminal().is_some() {
        return 0;
    }
    generate_pawn_moves_with_mode(board, scratch, out, mode)
}

/// Reuses `out` buffer and `scratch` BFS pool — board restored after wall trials.
pub fn generate_legal_moves_into(board: &mut Board, out: &mut Vec<Move>, scratch: &mut BfsScratch) {
    out.clear();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, scratch);
    out.extend_from_slice(&buf[..n]);
}

pub fn generate_pawn_moves(board: &Board) -> Vec<Move> {
    let mut out = Vec::with_capacity(4);
    generate_pawn_moves_into(board, &mut out);
    out
}

pub fn generate_pawn_moves_into(board: &Board, out: &mut Vec<Move>) {
    let mut buf = [Move::Pawn { row: 0, col: 0 }; 8];
    let n = generate_pawn_moves_slice(board, &mut buf);
    out.extend_from_slice(&buf[..n]);
}

pub(crate) fn generate_pawn_moves_slice(board: &Board, out: &mut [Move]) -> usize {
    generate_pawn_moves_scalar_for(board, board.side_to_move, out)
}

/// Pawn moves for an arbitrary player — no board clone, no wall generation.
/// Hot path for mobility eval: counting pawn moves must never trigger the
/// full legal movegen (which BFS-validates every wall placement).
pub(crate) fn generate_pawn_moves_for(board: &Board, player: Player, out: &mut [Move]) -> usize {
    generate_pawn_moves_scalar_for(board, player, out)
}

/// Scalar pawn moves — kept for mobility eval and differential tests vs bitboard.
pub(crate) fn generate_pawn_moves_scalar_for(
    board: &Board,
    player: Player,
    out: &mut [Move],
) -> usize {
    let side = player as usize;
    let (fr, fc) = board.pawns[side];
    let (or, oc) = board.pawns[1 - side];
    let mut n = 0usize;

    for (dr, dc) in DIRS {
        if !can_step(board, fr, fc, dr, dc) {
            continue;
        }
        let nr = (fr as i8 + dr) as u8;
        let nc = (fc as i8 + dc) as u8;

        if (nr, nc) != (or, oc) {
            out[n] = Move::Pawn { row: nr, col: nc };
            n += 1;
            continue;
        }

        if can_step(board, nr, nc, dr, dc) {
            let jr = (nr as i8 + dr) as u8;
            let jc = (nc as i8 + dc) as u8;
            out[n] = Move::Pawn { row: jr, col: jc };
            n += 1;
            continue;
        }

        let perp = if dr != 0 {
            [(0i8, 1i8), (0, -1)]
        } else {
            [(1, 0), (-1, 0)]
        };
        for (pdr, pdc) in perp {
            if can_step(board, nr, nc, pdr, pdc) {
                let sr = (nr as i8 + pdr) as u8;
                let sc = (nc as i8 + pdc) as u8;
                out[n] = Move::Pawn { row: sr, col: sc };
                n += 1;
            }
        }
    }
    n
}

pub fn generate_wall_moves(board: &Board) -> Vec<Move> {
    let mut copy = board.clone();
    let mut out = Vec::with_capacity(64);
    let mut scratch = BfsScratch::new();
    generate_wall_moves_into(&mut copy, &mut out, &mut scratch);
    out
}

pub fn generate_wall_moves_into(board: &mut Board, out: &mut Vec<Move>, scratch: &mut BfsScratch) {
    let mut buf = [Move::Wall {
        row: 0,
        col: 0,
        orientation: WallOrientation::Horizontal,
    }; MAX_LEGAL_MOVES];
    let n = generate_wall_moves_slice(board, &mut buf, scratch);
    out.extend_from_slice(&buf[..n]);
}

/// Path-valid wall placements (both players retain routes). Ignores wall budget / side to move.
pub fn count_geometric_legal_walls(board: &mut Board, scratch: &mut BfsScratch) -> usize {
    let mut buf = [Move::Wall {
        row: 0,
        col: 0,
        orientation: WallOrientation::Horizontal,
    }; MAX_LEGAL_MOVES];
    generate_wall_moves_slice(board, &mut buf, scratch)
}

/// Board geometry that determines path-valid wall slots (ignores side / wall budget).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GeometricWallKey {
    horizontal_walls: u64,
    vertical_walls: u64,
    p0: (u8, u8),
    p1: (u8, u8),
}

impl GeometricWallKey {
    #[inline]
    pub fn from_board(board: &Board) -> Self {
        Self {
            horizontal_walls: board.horizontal_walls,
            vertical_walls: board.vertical_walls,
            p0: board.pawns[0],
            p1: board.pawns[1],
        }
    }
}

pub struct GeometricWallCache {
    key: GeometricWallKey,
    moves: [Move; MAX_LEGAL_MOVES],
    len: usize,
}

impl GeometricWallCache {
    pub fn wall_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

/// Who requested the cached geometric wall set (for profiling).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometricWallCacheRole {
    Eval,
    Movegen,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct GeometricWallCacheStats {
    pub hits_eval: u64,
    pub misses_eval: u64,
    pub hits_movegen: u64,
    pub misses_movegen: u64,
    pub wall_generation_calls: u64,
}

/// Path-valid wall slots, generating once per [`GeometricWallKey`].
pub fn geometric_wall_len_cached(
    cache: &mut Option<GeometricWallCache>,
    board: &mut Board,
    scratch: &mut BfsScratch,
    role: GeometricWallCacheRole,
    stats: Option<&mut GeometricWallCacheStats>,
) -> usize {
    let key = GeometricWallKey::from_board(board);
    if cache.as_ref().is_some_and(|c| c.key == key) {
        if let Some(s) = stats {
            match role {
                GeometricWallCacheRole::Eval => s.hits_eval += 1,
                GeometricWallCacheRole::Movegen => s.hits_movegen += 1,
            }
        }
        return cache.as_ref().unwrap().len;
    }
    if let Some(s) = stats {
        match role {
            GeometricWallCacheRole::Eval => s.misses_eval += 1,
            GeometricWallCacheRole::Movegen => s.misses_movegen += 1,
        }
        s.wall_generation_calls += 1;
    }
    let mut moves = [Move::Wall {
        row: 0,
        col: 0,
        orientation: WallOrientation::Horizontal,
    }; MAX_LEGAL_MOVES];
    let len = generate_wall_moves_slice(board, &mut moves, scratch);
    *cache = Some(GeometricWallCache { key, moves, len });
    len
}

#[inline]
fn copy_geometric_walls_cached(
    cache: &GeometricWallCache,
    out: &mut [Move],
) -> usize {
    debug_assert!(out.len() >= cache.len);
    out[..cache.len].copy_from_slice(&cache.moves[..cache.len]);
    cache.len
}

/// Legal movegen with a shared geometric-wall cache (pawn moves + budget-gated walls).
pub fn generate_legal_moves_slice_cached(
    cache: &mut Option<GeometricWallCache>,
    board: &mut Board,
    out: &mut [Move],
    scratch: &mut BfsScratch,
    stats: Option<&mut GeometricWallCacheStats>,
) -> usize {
    if board.is_terminal().is_some() {
        return 0;
    }

    let mut n = generate_pawn_moves_with_mode(board, scratch, out, PawnGenMode::default());
    if board.walls_remaining[board.side_to_move as usize] > 0 {
        geometric_wall_len_cached(
            cache,
            board,
            scratch,
            GeometricWallCacheRole::Movegen,
            stats,
        );
        let cached = cache.as_ref().expect("filled by geometric_wall_len_cached");
        n += copy_geometric_walls_cached(cached, &mut out[n..]);
    }
    debug_assert!(n <= MAX_LEGAL_MOVES);
    n
}

fn generate_wall_moves_slice(
    board: &mut Board,
    out: &mut [Move],
    _scratch: &mut BfsScratch,
) -> usize {
    // Walls: L1 empty ∧ L2 collision → topo flood-skip → L3 bitboard flood when needed.
    // Flood grids are built only if some candidate actually needs L3.
    let masks = wall_masks(board);
    let mut ctx: Option<WallTrialCtx> = None;
    let mut n = 0usize;
    n += collect_wall_orientation(
        board,
        masks.l12_h,
        masks.topo_h,
        WallOrientation::Horizontal,
        &mut out[n..],
        &mut ctx,
    );
    n += collect_wall_orientation(
        board,
        masks.l12_v,
        masks.topo_v,
        WallOrientation::Vertical,
        &mut out[n..],
        &mut ctx,
    );
    n
}

/// L1∧L2 candidates — phase A emits isolated walls; phase B runs L3 flood.
fn collect_wall_orientation(
    board: &Board,
    candidates: u64,
    needs_flood: u64,
    orientation: WallOrientation,
    out: &mut [Move],
    ctx: &mut Option<WallTrialCtx>,
) -> usize {
    let mut n = 0usize;

    let mut isolated = candidates & !needs_flood;
    while isolated != 0 {
        let bit = isolated.trailing_zeros();
        isolated &= isolated - 1;
        out[n] = Move::Wall {
            row: (bit / 8) as u8,
            col: (bit % 8) as u8,
            orientation,
        };
        n += 1;
    }

    let mut heavy = candidates & needs_flood;
    while heavy != 0 {
        let bit = heavy.trailing_zeros();
        heavy &= heavy - 1;
        let row = (bit / 8) as u8;
        let col = (bit % 8) as u8;
        debug_assert!(wall_physically_legal_o1(
            board,
            row,
            col,
            orientation == WallOrientation::Horizontal
        ));
        if ctx
            .get_or_insert_with(|| WallTrialCtx::new(board))
            .wall_keeps_paths_open(row, col, orientation)
        {
            out[n] = Move::Wall {
                row,
                col,
                orientation,
            };
            n += 1;
        }
    }
    n
}

/// Per-node wall-trial state: directional blocked-step grids + pawn flood bits.
struct WallTrialCtx {
    grids: WallGrids,
    p1_bit: u128,
    p2_bit: u128,
}

impl WallTrialCtx {
    fn new(board: &Board) -> Self {
        let (r1, c1) = board.pawn(Player::One);
        let (r2, c2) = board.pawn(Player::Two);
        Self {
            grids: WallGrids::from_board(board),
            p1_bit: pawn_bit(r1, c1),
            p2_bit: pawn_bit(r2, c2),
        }
    }

    /// Speculative trial: place the wall's blocked-edge delta, run binary flood fill
    /// for both players (`pbff_wall_legal`; P2 reuses P1 visited bits), roll back.
    #[inline]
    fn wall_keeps_paths_open(&mut self, row: u8, col: u8, orientation: WallOrientation) -> bool {
        let delta = wall_delta(row, col, orientation);
        self.grids.place(delta);
        let ok = pbff_wall_legal(self.p1_bit, self.p2_bit, &self.grids);
        self.grids.remove(delta);
        ok
    }
}

/// Trial wall placement — both players must still reach goals (website rules oracle).
pub fn wall_path_ok_after_place(
    board: &mut Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> bool {
    let mut ctx = WallTrialCtx::new(board);
    ctx.wall_keeps_paths_open(row, col, orientation)
}

/// Matches scraped `collidesWithExistingWall` — scalar reference for the L2 table.
#[cfg(test)]
fn wall_collides(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    let perpendicular = match orientation {
        WallOrientation::Horizontal => WallOrientation::Vertical,
        WallOrientation::Vertical => WallOrientation::Horizontal,
    };

    if has_wall(board, row, col, orientation) || has_wall(board, row, col, perpendicular) {
        return true;
    }

    match orientation {
        WallOrientation::Horizontal => {
            if col > 0 && has_wall(board, row, col - 1, WallOrientation::Horizontal) {
                return true;
            }
            if col < 7 && has_wall(board, row, col + 1, WallOrientation::Horizontal) {
                return true;
            }
        }
        WallOrientation::Vertical => {
            if row > 0 && has_wall(board, row - 1, col, WallOrientation::Vertical) {
                return true;
            }
            if row < 7 && has_wall(board, row + 1, col, WallOrientation::Vertical) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
pub(crate) fn wall_collides_test(
    board: &Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> bool {
    wall_collides(board, row, col, orientation)
}

/// Matches scraped `canWallBlock` — wall must touch existing topology to matter.
pub fn can_wall_block_topology(
    board: &Board,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> bool {
    let js_col = col + 1;
    let js_row = row + 1;

    let (on_a, on_b) = match orientation {
        // Scraped `sideOnEdge` compared against col 9 (`numCols`) — unreachable for our
        // 0-based slots (rightmost H slot is js_col 8), so right-edge H walls skipped the
        // path flood and trapping walls were accepted (canta game 0 depth 2: 5980 ≠ 5978).
        WallOrientation::Horizontal => (js_col == 1, js_col == 8),
        WallOrientation::Vertical => (js_row == 8, js_row == 1),
    };

    let side_a = on_a || touching_side_a(board, row, col, orientation);
    let side_b = on_b || touching_side_b(board, row, col, orientation);
    let middle = touching_middle(board, row, col, orientation);

    (side_a && side_b) || (side_a && middle) || (side_b && middle)
}

fn touching_side_a(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    match orientation {
        WallOrientation::Horizontal => {
            wall_at_offset(board, row, col, &[(0, -1)], WallOrientation::Vertical)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(1, 0), (0, -1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(-1, 0), (0, -1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, -1), (0, -1)],
                    WallOrientation::Horizontal,
                )
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, -1), (1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, 1), (1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(1, 0), (1, 0)],
                    WallOrientation::Vertical,
                )
        }
    }
}

fn touching_side_b(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    match orientation {
        WallOrientation::Horizontal => {
            wall_at_offset(board, row, col, &[(0, 1)], WallOrientation::Vertical)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(1, 0), (0, 1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(-1, 0), (0, 1)],
                    WallOrientation::Vertical,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, 1), (0, 1)],
                    WallOrientation::Horizontal,
                )
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(-1, 0)], WallOrientation::Horizontal)
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, -1), (-1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(0, 1), (-1, 0)],
                    WallOrientation::Horizontal,
                )
                || wall_at_offset(
                    board,
                    row,
                    col,
                    &[(-1, 0), (-1, 0)],
                    WallOrientation::Vertical,
                )
        }
    }
}

fn touching_middle(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    match orientation {
        WallOrientation::Horizontal => {
            wall_at_offset(board, row, col, &[(1, 0)], WallOrientation::Vertical)
                || wall_at_offset(board, row, col, &[(-1, 0)], WallOrientation::Vertical)
        }
        WallOrientation::Vertical => {
            wall_at_offset(board, row, col, &[(0, -1)], WallOrientation::Horizontal)
                || wall_at_offset(board, row, col, &[(0, 1)], WallOrientation::Horizontal)
        }
    }
}

fn wall_at_offset(
    board: &Board,
    row: u8,
    col: u8,
    offsets: &[(i8, i8)],
    orientation: WallOrientation,
) -> bool {
    let (wr, wc) = apply_offsets(row, col, offsets);
    if wr > 7 || wc > 7 {
        return false;
    }
    has_wall(board, wr, wc, orientation)
}

fn apply_offsets(mut row: u8, mut col: u8, offsets: &[(i8, i8)]) -> (u8, u8) {
    for (dr, dc) in offsets {
        row = (row as i16 + *dr as i16) as u8;
        col = (col as i16 + *dc as i16) as u8;
    }
    (row, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct ShortcutAudit {
        checked: usize,
        strict_isolated: usize,
        topology_fast: usize,
        bff_checks: usize,
    }

    fn audit_shortcuts(board: &Board, audit: &mut ShortcutAudit) {
        let masks = wall_masks(board);
        let mut ctx = WallTrialCtx::new(board);
        for (orientation, candidates, needs_flood) in [
            (WallOrientation::Horizontal, masks.l12_h, masks.topo_h),
            (WallOrientation::Vertical, masks.l12_v, masks.topo_v),
        ] {
            let horizontal = orientation == WallOrientation::Horizontal;
            let mut remaining = candidates;
            while remaining != 0 {
                let slot = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let row = (slot / 8) as u8;
                let col = (slot % 8) as u8;
                let exact = ctx.wall_keeps_paths_open(row, col, orientation);
                let topology_fast = needs_flood & (1u64 << slot) == 0;
                let current = topology_fast || exact;
                if topology_fast {
                    audit.topology_fast += 1;
                } else {
                    audit.bff_checks += 1;
                }
                assert_eq!(
                    current,
                    exact,
                    "topology shortcut mismatch: candidate={row},{col},{orientation:?} h={:#018x} v={:#018x} pawns={:?}",
                    board.horizontal_walls,
                    board.vertical_walls,
                    board.pawns,
                );
                if crate::movegen::wall_masks::wall_is_strictly_isolated(board, slot, horizontal) {
                    audit.strict_isolated += 1;
                    assert!(
                        exact,
                        "isolated counterexample: candidate={row},{col},{orientation:?} h={:#018x} v={:#018x} pawns={:?}",
                        board.horizontal_walls,
                        board.vertical_walls,
                        board.pawns,
                    );
                    assert_eq!(
                        needs_flood & (1u64 << slot),
                        0,
                        "strict isolation must already be covered by topology shortcut"
                    );
                }
                audit.checked += 1;
            }
        }
    }

    fn replay(moves: &[&str]) -> Board {
        let mut board = Board::new();
        for &mv in moves {
            board.apply_algebraic(mv);
        }
        board
    }

    #[test]
    fn isolated_and_topology_shortcuts_match_exact_bff_globally() {
        let mut audit = ShortcutAudit::default();
        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        for _game in 0..64 {
            let mut board = Board::new();
            for _ply in 0..48 {
                audit_shortcuts(&board, &mut audit);
                if board.is_terminal().is_some() {
                    break;
                }
                let mut scratch = BfsScratch::new();
                let mut moves = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
                let n = generate_legal_moves_slice(&mut board, &mut moves, &mut scratch);
                if n == 0 {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let mv = moves[(seed as usize) % n];
                let _ = board.make_move(mv);
            }
        }
        eprintln!(
            "shortcut audit checked={} strict_isolated={} topology_fast={} bff_checks={}",
            audit.checked, audit.strict_isolated, audit.topology_fast, audit.bff_checks
        );
        assert!(
            audit.checked >= 10_000,
            "only checked {} wall candidates",
            audit.checked
        );
    }

    #[test]
    fn isolated_and_topology_shortcuts_match_adversarial_bff() {
        let fixtures: &[&[&str]] = &[
            &[
                "e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e3h", "e5h", "c3h", "c5h", "g3h",
            ],
            &[
                "e2", "e8", "e3", "e7", "e4", "e6", "c3h", "e7h", "e3h", "c7h", "f4", "g7h", "f5",
                "h8h", "f6", "b6v", "g3h", "h7v", "a3h",
            ],
            &[
                "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e4h", "d4", "c4h", "e5v", "a5h", "h8h",
                "d6", "b5v", "f3v", "e7v", "c3h", "d7h", "b2v", "h6h",
            ],
        ];
        let mut audit = ShortcutAudit::default();
        for moves in fixtures {
            audit_shortcuts(&replay(moves), &mut audit);
        }
        assert!(audit.checked > 100);
    }

    #[test]
    fn start_has_three_pawn_moves_for_white() {
        let board = Board::new();
        let pawns = generate_pawn_moves(&board);
        assert_eq!(pawns.len(), 3);
    }

    #[test]
    fn start_has_many_wall_moves() {
        let board = Board::new();
        let walls = generate_wall_moves(&board);
        assert!(walls.len() > 100);
    }
    #[test]
    fn slice_matches_vec_at_startpos() {
        let mut board = Board::new();
        let mut scratch = BfsScratch::new();
        let mut slice_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice(&mut board, &mut slice_buf, &mut scratch);
        let vec_moves = generate_legal_moves(&board);
        assert_eq!(n, vec_moves.len());
        assert_eq!(&slice_buf[..n], vec_moves.as_slice());
        assert!(n <= MAX_LEGAL_MOVES);
    }

    #[test]
    fn wall_trial_leaves_board_unchanged() {
        let mut board = Board::new();
        let before = board.clone();
        let mut scratch = BfsScratch::new();
        let mut moves = Vec::new();
        generate_wall_moves_into(&mut board, &mut moves, &mut scratch);
        assert_eq!(board, before);
    }

    #[test]
    fn zero_wall_budget_never_emits_cached_walls() {
        let mut board = Board::new();
        board.apply_algebraic("e2");
        board.apply_algebraic("e8");
        board.apply_algebraic("e3");
        board.apply_algebraic("e7");
        board.apply_algebraic("e4");
        board.apply_algebraic("e6");
        board.apply_algebraic("c3h");
        // Side to move has no walls left; geometry still has path-valid slots.
        board.walls_remaining[board.side_to_move as usize] = 0;
        let mut scratch = BfsScratch::new();
        let mut cache = None;
        let mut stats = GeometricWallCacheStats::default();
        let geom = geometric_wall_len_cached(
            &mut cache,
            &mut board,
            &mut scratch,
            GeometricWallCacheRole::Eval,
            Some(&mut stats),
        );
        assert!(geom > 0, "fixture should have geometric wall slots");

        let mut out = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice_cached(
            &mut cache,
            &mut board,
            &mut out,
            &mut scratch,
            Some(&mut stats),
        );
        assert!(n > 0);
        assert!(out[..n].iter().all(|m| matches!(m, Move::Pawn { .. })));
        // Budget gate skips wall path entirely — cache is not consulted for movegen.
        assert_eq!(stats.hits_movegen, 0);
        assert_eq!(stats.misses_movegen, 0);
    }

    #[test]
    fn geometric_wall_cache_matches_uncached() {
        let mut board = Board::new();
        board.apply_algebraic("e2");
        board.apply_algebraic("e8");
        board.apply_algebraic("e3");
        board.apply_algebraic("e7");
        board.apply_algebraic("e4");
        board.apply_algebraic("e6");
        board.apply_algebraic("c3h");
        let mut scratch = BfsScratch::new();
        let mut uncached = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n_uncached = generate_legal_moves_slice(&mut board, &mut uncached, &mut scratch);

        let mut cache = None;
        let mut stats = GeometricWallCacheStats::default();
        let mut cached = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n_cached = generate_legal_moves_slice_cached(
            &mut cache,
            &mut board,
            &mut cached,
            &mut scratch,
            Some(&mut stats),
        );
        assert_eq!(n_uncached, n_cached);
        assert_eq!(&uncached[..n_uncached], &cached[..n_cached]);
        assert_eq!(stats.wall_generation_calls, 1);
        assert_eq!(stats.misses_movegen, 1);
        assert_eq!(stats.hits_movegen, 0);

        let count = geometric_wall_len_cached(
            &mut cache,
            &mut board,
            &mut scratch,
            GeometricWallCacheRole::Eval,
            Some(&mut stats),
        );
        let mut wall_only = [Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Horizontal,
        }; MAX_LEGAL_MOVES];
        let n_walls = generate_wall_moves_slice(&mut board, &mut wall_only, &mut scratch);
        assert_eq!(count, n_walls);
        assert_eq!(stats.wall_generation_calls, 1);
        assert_eq!(stats.hits_eval, 1);
    }
}
