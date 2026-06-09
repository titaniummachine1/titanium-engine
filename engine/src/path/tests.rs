#[cfg(test)]
mod naive_reference {
    use crate::core::board::Board;
    use crate::path::masks::DirMasks;
    use crate::util::grid::{can_step, is_goal, square_index, unpack_square};

    const NEIGHBORS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

    pub fn flood_fill_naive(board: &Board, start: u8) -> u128 {
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;

        while head < tail {
            let sq = queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                tail += 1;
            }
        }
        visited
    }

    pub fn can_reach_goal_naive(board: &Board, player: crate::core::board::Player) -> bool {
        use crate::core::board::Player;
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;

        while head < tail {
            let sq = queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            if is_goal(player, r) {
                return true;
            }
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                tail += 1;
            }
        }
        false
    }

    pub fn shortest_distance_naive(
        board: &Board,
        player: crate::core::board::Player,
    ) -> Option<u8> {
        use crate::core::board::Player;
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut depth = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;
        depth[0] = 0;

        while head < tail {
            let sq = queue[head];
            let d = depth[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            if is_goal(player, r) {
                return Some(d);
            }
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                depth[tail] = d + 1;
                tail += 1;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::naive_reference::{
        can_reach_goal_naive, flood_fill_naive, shortest_distance_naive,
    };
    use crate::core::board::{Board, Player, WallOrientation};
    use crate::path::bfs::{can_reach_goal, shortest_distance, BfsScratch};
    use crate::path::flood::flood_fill;
    use crate::path::masks::DirMasks;
    use crate::util::grid::{flood_bit_sq, set_wall, square_index};

    fn assert_bitwise_matches_naive(board: &Board) {
        let masks = DirMasks::from_board(board);
        let mut scratch = BfsScratch::new();

        for sq in 0u8..81 {
            let bitwise = flood_fill(sq, masks);
            let naive = flood_fill_naive(board, sq);
            assert_eq!(bitwise, naive, "reachable mismatch from sq {sq}");
        }

        for player in [Player::One, Player::Two] {
            assert_eq!(
                scratch.can_reach_goal(board, player),
                can_reach_goal_naive(board, player),
            );
            assert_eq!(
                scratch.shortest_distance(board, player),
                shortest_distance_naive(board, player),
            );
        }
    }

    #[test]
    fn start_position_reachable() {
        let board = Board::new();
        assert!(can_reach_goal(&board, Player::One));
        assert_eq!(shortest_distance(&board, Player::One), Some(8));
    }

    #[test]
    fn bitwise_flood_matches_naive_on_startpos() {
        assert_bitwise_matches_naive(&Board::new());
    }

    #[test]
    fn both_reachable_mask_includes_both_pawns() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let mask = scratch.both_reachable_mask(&board);
        assert_ne!(mask & (1u128 << square_index(0, 4)), 0);
        assert_ne!(mask & (1u128 << square_index(8, 4)), 0);
    }

    #[test]
    fn full_barrier_blocks_p1() {
        let mut board = Board::new();
        for c in 0..8u8 {
            set_wall(&mut board, 6, c, WallOrientation::Horizontal, true);
        }
        assert!(!can_reach_goal(&board, Player::One));
    }
}
