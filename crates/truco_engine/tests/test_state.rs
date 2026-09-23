//! Comprehensive state transition, action masking, trick advancing,
//! envido chains, truco raises, and game termination tests for `BitboardState`.

use truco_engine::actions::{
    ACTION_ENVIDO, ACTION_FALTA_ENVIDO, ACTION_IR_AL_MAZO, ACTION_NO_QUIERO_ENVIDO,
    ACTION_NO_QUIERO_TRUCO, ACTION_PLAY_CARD_0, ACTION_PLAY_CARD_1, ACTION_PLAY_CARD_2,
    ACTION_QUIERO_ENVIDO, ACTION_QUIERO_TRUCO, ACTION_REAL_ENVIDO, ACTION_RETRUCO,
    ACTION_TRUCO, ACTION_VALE_CUATRO,
};
use truco_engine::state::BitboardState;

#[test]
fn test_bitboard_state_size_and_copy() {
    assert_eq!(
        std::mem::size_of::<BitboardState>(),
        32,
        "BitboardState MUST be exactly 32 bytes for AVX2 register packing"
    );

    let hands = [[0, 1, 2], [10, 11, 12]];
    let s1 = BitboardState::new(hands, 0);
    // BitboardState must implement Copy
    let s2 = s1;
    assert_eq!(s1, s2);
}

#[test]
fn test_initial_state_and_legal_actions_mano_0() {
    // Player 0 has [1 Espadas, 2 Espadas, 3 Espadas]
    // Player 1 has [1 Bastos, 2 Bastos, 3 Bastos]
    let hands = [[0, 1, 2], [10, 11, 12]];
    let state = BitboardState::new(hands, 0);

    assert_eq!(state.active_player(), 0);
    assert_eq!(state.mano(), 0);
    assert_eq!(state.current_trick(), 0);
    assert_eq!(state.truco_level(), 1);
    assert!(!state.envido_resolved());
    assert!(!state.is_done());

    let mask = state.legal_actions_mask();

    // Mano at trick 0 before cards played can call:
    assert_ne!(mask & (1 << ACTION_ENVIDO), 0, "ENVIDO must be legal");
    assert_ne!(mask & (1 << ACTION_REAL_ENVIDO), 0, "REAL_ENVIDO must be legal");
    assert_ne!(mask & (1 << ACTION_FALTA_ENVIDO), 0, "FALTA_ENVIDO must be legal");
    assert_ne!(mask & (1 << ACTION_TRUCO), 0, "TRUCO must be legal");
    assert_ne!(mask & (1 << ACTION_PLAY_CARD_0), 0, "PLAY_CARD_0 must be legal");
    assert_ne!(mask & (1 << ACTION_PLAY_CARD_1), 0, "PLAY_CARD_1 must be legal");
    assert_ne!(mask & (1 << ACTION_PLAY_CARD_2), 0, "PLAY_CARD_2 must be legal");
    assert_ne!(mask & (1 << ACTION_IR_AL_MAZO), 0, "IR_AL_MAZO must be legal");

    // Illegal actions:
    assert_eq!(mask & (1 << ACTION_QUIERO_ENVIDO), 0);
    assert_eq!(mask & (1 << ACTION_NO_QUIERO_ENVIDO), 0);
    assert_eq!(mask & (1 << ACTION_QUIERO_TRUCO), 0);
    assert_eq!(mask & (1 << ACTION_NO_QUIERO_TRUCO), 0);
    assert_eq!(mask & (1 << ACTION_RETRUCO), 0);
    assert_eq!(mask & (1 << ACTION_VALE_CUATRO), 0);
}

#[test]
fn test_envido_chain_quiero() {
    // P0 has [7 Espadas (6), 6 Espadas (5), 4 Espadas (3)] -> envido 33
    // P1 has [7 Bastos (16), 5 Bastos (14), 4 Bastos (13)] -> envido 32
    let hands = [[6, 5, 3], [16, 14, 13]];
    let s0 = BitboardState::new(hands, 0);

    // P0 calls ENVIDO
    let s1 = s0.step(ACTION_ENVIDO);
    assert_eq!(s1.active_player(), 1);
    let mask1 = s1.legal_actions_mask();
    assert_ne!(mask1 & (1 << ACTION_QUIERO_ENVIDO), 0);
    assert_ne!(mask1 & (1 << ACTION_NO_QUIERO_ENVIDO), 0);
    assert_ne!(mask1 & (1 << ACTION_ENVIDO), 0); // envido va de nuevo
    assert_ne!(mask1 & (1 << ACTION_REAL_ENVIDO), 0);
    assert_ne!(mask1 & (1 << ACTION_FALTA_ENVIDO), 0);

    // P1 responds QUIERO_ENVIDO
    let s2 = s1.step(ACTION_QUIERO_ENVIDO);
    assert!(s2.envido_resolved());
    assert_eq!(s2.active_player(), 0, "Play resumes with trick leader P0");

    // P0 had 33 vs P1's 32, so P0 won 2 envido points
    assert_eq!(s2.score(), (2, 0));

    // Envido cannot be called again
    let mask2 = s2.legal_actions_mask();
    assert_eq!(mask2 & (1 << ACTION_ENVIDO), 0);
    assert_eq!(mask2 & (1 << ACTION_REAL_ENVIDO), 0);
    assert_eq!(mask2 & (1 << ACTION_FALTA_ENVIDO), 0);
}

#[test]
fn test_envido_chain_no_quiero() {
    let hands = [[6, 5, 3], [16, 14, 13]];
    let s0 = BitboardState::new(hands, 0);

    let s1 = s0.step(ACTION_ENVIDO);
    let s2 = s1.step(ACTION_NO_QUIERO_ENVIDO);

    assert!(s2.envido_resolved());
    assert_eq!(s2.active_player(), 0);
    // P0 gets 1 point when P1 declines initial envido
    assert_eq!(s2.score(), (1, 0));
}

#[test]
fn test_envido_nested_raise_no_quiero() {
    let hands = [[6, 5, 3], [16, 14, 13]];
    let s0 = BitboardState::new(hands, 0);

    // P0: ENVIDO
    let s1 = s0.step(ACTION_ENVIDO);
    // P1: REAL ENVIDO (total stake would be 5 if accepted)
    let s2 = s1.step(ACTION_REAL_ENVIDO);
    assert_eq!(s2.active_player(), 0);

    // P0: NO QUIERO -> P1 wins previous accepted points (2 points from ENVIDO)
    let s3 = s2.step(ACTION_NO_QUIERO_ENVIDO);
    assert!(s3.envido_resolved());
    assert_eq!(s3.score(), (0, 2));
}

#[test]
fn test_truco_chain_quiero_and_retruco() {
    let hands = [[0, 1, 2], [10, 11, 12]];
    let s0 = BitboardState::new(hands, 0);

    // P0 calls TRUCO
    let s1 = s0.step(ACTION_TRUCO);
    assert_eq!(s1.active_player(), 1);
    let mask1 = s1.legal_actions_mask();
    assert_ne!(mask1 & (1 << ACTION_QUIERO_TRUCO), 0);
    assert_ne!(mask1 & (1 << ACTION_NO_QUIERO_TRUCO), 0);
    assert_ne!(mask1 & (1 << ACTION_RETRUCO), 0);

    // P1 responds RETRUCO
    let s2 = s1.step(ACTION_RETRUCO);
    assert_eq!(s2.active_player(), 0);
    let mask2 = s2.legal_actions_mask();
    assert_ne!(mask2 & (1 << ACTION_QUIERO_TRUCO), 0);
    assert_ne!(mask2 & (1 << ACTION_NO_QUIERO_TRUCO), 0);
    assert_ne!(mask2 & (1 << ACTION_VALE_CUATRO), 0);

    // P0 calls QUIERO_TRUCO
    let s3 = s2.step(ACTION_QUIERO_TRUCO);
    assert_eq!(s3.truco_level(), 3, "Retruco accepted brings level to 3");
    assert!(!s3.is_done());
}

#[test]
fn test_truco_chain_no_quiero() {
    let hands = [[0, 1, 2], [10, 11, 12]];
    let s0 = BitboardState::new(hands, 0);

    let s1 = s0.step(ACTION_TRUCO);
    let s2 = s1.step(ACTION_NO_QUIERO_TRUCO);

    assert!(s2.is_done(), "Hand must be done after NO_QUIERO_TRUCO");
    assert_eq!(s2.winner(), Some(0));
    assert_eq!(s2.points_won(), 1);
}

#[test]
fn test_trick_progression_showdown() {
    // P0: 1 Espadas (0, rank 14), 7 Espadas (6, rank 12), 4 Espadas (3, rank 1)
    // P1: 1 Bastos (10, rank 13), 7 Oros (26, rank 11), 4 Bastos (13, rank 1)
    let hands = [[0, 6, 3], [10, 26, 13]];
    let s0 = BitboardState::new(hands, 0);

    // Trick 1:
    // P0 plays 1 Espadas (card index 0)
    let s1 = s0.step(ACTION_PLAY_CARD_0);
    assert_eq!(s1.active_player(), 1);
    // P1 plays 1 Bastos (card index 0)
    let s2 = s1.step(ACTION_PLAY_CARD_0);

    // 1 Espadas (rank 14) beats 1 Bastos (rank 13) -> P0 won Trick 1
    assert_eq!(s2.current_trick(), 1);
    assert_eq!(s2.active_player(), 0, "P0 won trick 1 and leads trick 2");
    assert!(!s2.is_done());

    // Trick 2:
    // P0 plays 7 Espadas (now card index 0 in hand)
    let s3 = s2.step(ACTION_PLAY_CARD_0);
    assert_eq!(s3.active_player(), 1);
    // P1 plays 7 Oros (now card index 0 in hand)
    let s4 = s3.step(ACTION_PLAY_CARD_0);

    // 7 Espadas (rank 12) beats 7 Oros (rank 11) -> P0 wins Trick 2 (2-0)
    assert!(s4.is_done(), "P0 won 2 tricks -> hand must be done");
    assert_eq!(s4.winner(), Some(0));
    assert_eq!(s4.points_won(), 1, "Base truco points = 1");
}

#[test]
fn test_parda_resolution_in_trick_1() {
    // Both players play 3s in trick 1 (tie/parda)
    // P0: 3 Espadas (2, rank 10), 1 Espadas (0, rank 14), 4 Espadas (3, rank 1)
    // P1: 3 Bastos (12, rank 10), 1 Bastos (10, rank 13), 4 Bastos (13, rank 1)
    let hands = [[2, 0, 3], [12, 10, 13]];
    let s0 = BitboardState::new(hands, 0);

    // Trick 1: 3 Espadas vs 3 Bastos (parda)
    let s1 = s0.step(ACTION_PLAY_CARD_0);
    let s2 = s1.step(ACTION_PLAY_CARD_0);

    assert_eq!(s2.current_trick(), 1);
    // On parda in trick 1, trick leader remains mano (0)
    assert_eq!(s2.active_player(), 0);
    assert!(!s2.is_done());

    // Trick 2: P0 plays 1 Espadas (rank 14), P1 plays 1 Bastos (rank 13)
    let s3 = s2.step(ACTION_PLAY_CARD_0);
    let s4 = s3.step(ACTION_PLAY_CARD_0);

    // T1 parda, T2 decided -> T2 winner wins hand immediately!
    assert!(s4.is_done(), "Hand decided after T1 parda and T2 win");
    assert_eq!(s4.winner(), Some(0));
}

#[test]
fn test_ir_al_mazo_immediate_surrender() {
    let hands = [[0, 1, 2], [10, 11, 12]];
    let s0 = BitboardState::new(hands, 0);

    // P0 folds immediately
    let s1 = s0.step(ACTION_IR_AL_MAZO);
    assert!(s1.is_done());
    assert_eq!(s1.winner(), Some(1));
    assert_eq!(s1.points_won(), 1);
}
