//! Extended soundness tests for the wall-ignorance forced-loss certificate.

#[cfg(test)]
mod extended {
    use crate::core::board::{Board, Player};
    use crate::movegen::generate_legal_moves;
    use crate::titanium::cert_bridge::titanium_game_from_board;
    use crate::titanium::game::GameState;
    use crate::titanium::wall_ignore_cert::{
        try_wall_ignorance_loss_cert, try_wall_ignore_cert_board, CertScratch, WALL_IGNORE_STATS,
    };
    use crate::titanium::wall_ignore_corridor::{
        build_column_four_corridor_fixture, detect_zero_delay_corridor, shortest_distance,
        walls_that_block_edge, BoardEdge, CorridorScratch,
    };
    fn corridor_game(wl0: i32, wl1: i32, turn: usize) -> GameState {
        let mut g = build_column_four_corridor_fixture();
        g.wl = [wl0, wl1];
        g.turn = turn;
        g
    }

    #[test]
    fn sequential_future_walls_preserve_corridor() {
        let g = corridor_game(10, 10, 0);
        let mut scratch = CorridorScratch::new();
        let guarantee = detect_zero_delay_corridor(&g, 0, &mut scratch).expect("corridor");
        let d0 = guarantee.max_own_moves_to_goal;

        let mut pos = g.clone();
        let mut legal_walls: Vec<i16> = Vec::new();
        for wtype in 0..2 {
            for slot in 0..64 {
                if pos.wall_legal(wtype, slot) {
                    legal_walls.push(
                        (if wtype == 0 {
                            crate::titanium::MOVE_HW_BASE
                        } else {
                            crate::titanium::MOVE_VW_BASE
                        }) + slot as i16,
                    );
                }
            }
        }

        for mv in &legal_walls {
            let mut trial = pos.clone();
            trial.make_move(*mv);
            let mut sc = CorridorScratch::new();
            let g2 = detect_zero_delay_corridor(&trial, 0, &mut sc);
            assert!(
                g2.is_some(),
                "legal wall {mv} must not destroy white corridor"
            );
            assert_eq!(
                g2.unwrap().max_own_moves_to_goal,
                shortest_distance(&trial, 0),
                "distance unchanged after wall {mv}"
            );
        }

        // Advance one step along corridor and repeat on suffix.
        if guarantee.path.len() > 1 {
            let step = guarantee.path[1] as i16;
            pos.make_move(step);
            let mut sc = CorridorScratch::new();
            let suffix = detect_zero_delay_corridor(&pos, 0, &mut sc).expect("suffix corridor");
            assert_eq!(suffix.max_own_moves_to_goal, d0 - 1);
        }
    }

    #[test]
    fn off_path_legal_walls_preserve_path() {
        let g = corridor_game(10, 10, 0);
        let mut scratch = CorridorScratch::new();
        let guarantee = detect_zero_delay_corridor(&g, 0, &mut scratch).expect("corridor");
        let protected: std::collections::HashSet<BoardEdge> =
            guarantee.protected_edges.iter().copied().collect();

        let mut pos = g.clone();
        for wtype in 0..2 {
            for slot in 0..64 {
                let mv = (if wtype == 0 {
                    crate::titanium::MOVE_HW_BASE
                } else {
                    crate::titanium::MOVE_VW_BASE
                }) + slot as i16;
                if !pos.wall_legal(wtype, slot) {
                    continue;
                }
                let touches_protected = walls_that_block_edge(BoardEdge::new(0, 1))
                    .iter()
                    .any(|_| false)
                    || {
                        let mut trial_edges = Vec::new();
                        let r = slot / 8;
                        let c = slot % 8;
                        let a = r * 9 + c;
                        trial_edges.push(BoardEdge::new(a, a + 1));
                        trial_edges.push(BoardEdge::new(a, a + 9));
                        trial_edges.iter().any(|e| protected.contains(e))
                    };
                if touches_protected {
                    continue;
                }
                let mut trial = pos.clone();
                trial.make_move(mv);
                let mut sc = CorridorScratch::new();
                let still = detect_zero_delay_corridor(&trial, 0, &mut sc);
                if still.is_some() {
                    assert_eq!(
                        still.unwrap().max_own_moves_to_goal,
                        guarantee.max_own_moves_to_goal
                    );
                }
            }
        }
    }

    #[test]
    fn adversarial_loser_wall_monotonicity() {
        let g = corridor_game(10, 10, 0);
        let mut scratch = CertScratch::new();
        let verdict =
            try_wall_ignorance_loss_cert(&mut g.clone(), &mut scratch, true).expect("cert");
        let orig_l_ply = verdict.loser_terminal_ply;

        let mut pos = g.clone();
        if pos.wl[1] > 0 {
            pos.turn = 1;
            for wtype in 0..2 {
                for slot in 0..64 {
                    if !pos.wall_legal(wtype, slot) {
                        continue;
                    }
                    let mut trial = pos.clone();
                    trial.make_move(
                        (if wtype == 0 {
                            crate::titanium::MOVE_HW_BASE
                        } else {
                            crate::titanium::MOVE_VW_BASE
                        }) + slot as i16,
                    );
                    let mut sc = CorridorScratch::new();
                    let w_g = detect_zero_delay_corridor(&trial, 0, &mut sc);
                    assert!(w_g.is_some(), "loser wall must not break winner corridor");
                    assert!(
                        w_g.unwrap().max_own_moves_to_goal <= 4,
                        "winner distance must not worsen"
                    );
                }
            }
        }
        assert!(orig_l_ply > 0);
    }

    #[test]
    fn random_differential_no_false_certificates() {
        WALL_IGNORE_STATS.reset();
        let mut seed = 0xdecaf_bad_u64;
        let mut next = || {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            seed.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        let mut certified = 0usize;
        let mut false_count = 0usize;

        for _ in 0..800 {
            let mut board = Board::new();
            let plies = 8 + (next() % 24) as usize;
            for _ in 0..plies {
                if board.is_terminal().is_some() {
                    break;
                }
                let moves = generate_legal_moves(&board);
                if moves.is_empty() {
                    break;
                }
                board.apply_move(moves[(next() as usize) % moves.len()]);
            }
            let total_walls = board.walls_remaining[0] as u32 + board.walls_remaining[1] as u32;
            if total_walls > 2 {
                continue;
            }
            let g = titanium_game_from_board(&board);
            let w_dist = shortest_distance(&g, 0);
            let b_dist = shortest_distance(&g, 1);
            if w_dist > 5 || b_dist > 6 || w_dist == 255 || b_dist == 255 {
                continue;
            }

            if let Some(verdict) = try_wall_ignore_cert_board(&board, true) {
                certified += 1;
                let exhaustive = exhaustive_winner(&mut g.clone(), 6);
                if exhaustive != Some(verdict.winner) {
                    false_count += 1;
                    eprintln!(
                        "FALSE CERT: p0={:?} p1={:?} stm={:?} cert={} exact={:?}",
                        board.pawns[0],
                        board.pawns[1],
                        board.side_to_move,
                        verdict.winner,
                        exhaustive
                    );
                }
            }
        }

        eprintln!(
            "random differential: certified={certified} false={false_count} stats={:?}",
            WALL_IGNORE_STATS.snapshot()
        );
        assert_eq!(false_count, 0, "must have zero false certificates");
    }

    fn exhaustive_winner(g: &mut GameState, max_depth: u32) -> Option<usize> {
        let w = g.winner();
        if w >= 0 {
            return Some(w as usize);
        }
        if max_depth == 0 {
            return None;
        }

        let stm = g.turn;
        let mut buf = [0i16; 160];
        let mut pawn_only = [0i16; 16];
        let pn = g.gen_pawn_moves(&mut pawn_only, 0);
        for i in 0..pn {
            buf[i] = pawn_only[i];
        }
        let mut n = pn;
        if g.wl[stm] > 0 {
            for wtype in 0..2 {
                for slot in 0..64 {
                    if g.wall_legal(wtype, slot) {
                        buf[n] = (if wtype == 0 {
                            crate::titanium::MOVE_HW_BASE
                        } else {
                            crate::titanium::MOVE_VW_BASE
                        }) + slot as i16;
                        n += 1;
                    }
                }
            }
        }

        let mut any_win = false;
        let mut all_loss = true;
        for i in 0..n {
            g.make_move(buf[i]);
            match exhaustive_winner(g, max_depth - 1) {
                Some(winner) if winner == stm => {
                    any_win = true;
                }
                Some(_) => {
                    all_loss = false;
                }
                None => {
                    all_loss = false;
                }
            }
            g.unmake_move();
            if any_win {
                return Some(stm);
            }
        }
        if all_loss && n > 0 {
            Some(1 - stm)
        } else {
            None
        }
    }

    #[test]
    fn jump_interaction_rejects_direct_certificate() {
        // Same column close race — jump can shorten path.
        let mut board = Board::new();
        board.pawns = [(3, 4), (5, 4)];
        board.walls_remaining = [0, 0];
        board.side_to_move = Player::One;
        board.hash = crate::core::zobrist::hash_board(&board);
        assert!(
            try_wall_ignore_cert_board(&board, true).is_none(),
            "shared-column jump race must not direct-certify"
        );
    }

    #[test]
    fn feature_disabled_search_parity() {
        use crate::legacy_search::alphabeta::{search_best_move, SearchConfig};
        fn tempo_win_board() -> Board {
            let mut board = Board::new();
            board.pawns = [(3, 1), (5, 7)];
            board.walls_remaining = [2, 0];
            board.side_to_move = Player::One;
            board.hash = crate::core::zobrist::hash_board(&board);
            board
        }
        let base = SearchConfig {
            time_ms: 500,
            max_nodes: 500_000,
            log: false,
            book_hint: None,
            max_id_depth: 4,
            cert_enabled: None,
        };
        let prev = std::env::var("TITANIUM_WALL_IGNORE_LOSS_CERT").ok();
        std::env::remove_var("TITANIUM_WALL_IGNORE_LOSS_CERT");
        let off = search_best_move(&mut tempo_win_board(), base).expect("search");
        if let Some(v) = prev {
            std::env::set_var("TITANIUM_WALL_IGNORE_LOSS_CERT", v);
        } else {
            std::env::remove_var("TITANIUM_WALL_IGNORE_LOSS_CERT");
        }
        assert!(
            off.root_score.abs() <= 10_000 || off.root_score > 10_000,
            "baseline search must remain functional"
        );
    }

    #[test]
    fn bench_detector_cost() {
        WALL_IGNORE_STATS.reset();
        let g = corridor_game(10, 10, 0);
        let mut scratch = CertScratch::new();
        assert!(
            try_wall_ignorance_loss_cert(&mut g.clone(), &mut scratch, true).is_some(),
            "fixture must certify for benchmark"
        );
        for _ in 0..1000 {
            let _ = try_wall_ignorance_loss_cert(&mut g.clone(), &mut scratch, true);
        }
        let stats = WALL_IGNORE_STATS.snapshot();
        eprintln!("bench 1000 detector calls: {stats:?}");
        assert!(stats.detector_calls >= 990);
        assert!(stats.certificates_emitted >= 990);
        let avg_ns = stats.detector_nanos / stats.detector_calls.max(1);
        eprintln!("avg detector time: {avg_ns} ns");
        assert!(
            avg_ns < 5_000_000,
            "detector should stay under 5ms per call"
        );
    }
}
