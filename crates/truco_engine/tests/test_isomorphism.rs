//! Tests for suit isomorphism and canonical 64-bit information set keys.
//! Verifies that isomorphic hands produce identical canonical keys and that
//! asymmetric cards (1s, 7s) maintain distinct equivalence classes.

use std::collections::HashSet;
use truco_engine::actions::ACTION_PLAY_CARD_0;
use truco_engine::isomorphism::canonicalize_hand;
use truco_engine::state::BitboardState;

#[test]
fn test_pure_suit_isomorphism_generic_cards() {
    // 4, 5, 6 in each of the 4 suits:
    // Espadas: 4e (3), 5e (4), 6e (5)
    // Bastos:  4b (13), 5b (14), 6b (15)
    // Oros:    4o (23), 5o (24), 6o (25)
    // Copas:   4c (33), 5c (34), 6c (35)
    let dummy_opp = [10, 11, 12]; // arbitrary opponent hand

    let s_esp = BitboardState::new([[3, 4, 5], dummy_opp], 0);
    let s_bas = BitboardState::new([[13, 14, 15], dummy_opp], 0);
    let s_oro = BitboardState::new([[23, 24, 25], dummy_opp], 0);
    let s_cop = BitboardState::new([[33, 34, 35], dummy_opp], 0);

    let k_esp = s_esp.canonical_infoset_key(0);
    let k_bas = s_bas.canonical_infoset_key(0);
    let k_oro = s_oro.canonical_infoset_key(0);
    let k_cop = s_cop.canonical_infoset_key(0);

    assert_eq!(
        k_esp, k_bas,
        "Espadas and Bastos hands of 4-5-6 must have identical canonical key"
    );
    assert_eq!(
        k_bas, k_oro,
        "Bastos and Oros hands of 4-5-6 must have identical canonical key"
    );
    assert_eq!(
        k_oro, k_cop,
        "Oros and Copas hands of 4-5-6 must have identical canonical key"
    );
}

#[test]
fn test_asymmetry_preservation_special_cards() {
    let dummy_opp = [10, 11, 12];

    // Hand with 1 de Espadas (0, rank 14) vs 1 de Bastos (10, rank 13) vs 1 de Oros (20, rank 8)
    let s_1e = BitboardState::new([[0, 4, 5], dummy_opp], 0);
    let s_1b = BitboardState::new([[10, 4, 5], dummy_opp], 0);
    let s_1o = BitboardState::new([[20, 4, 5], dummy_opp], 0);

    let k_1e = s_1e.canonical_infoset_key(0);
    let k_1b = s_1b.canonical_infoset_key(0);
    let k_1o = s_1o.canonical_infoset_key(0);

    assert_ne!(
        k_1e, k_1b,
        "1 de Espadas (rank 14) and 1 de Bastos (rank 13) must NOT have same canonical key"
    );
    assert_ne!(
        k_1b, k_1o,
        "1 de Bastos (rank 13) and 1 de Oros (rank 8) must NOT have same canonical key"
    );
    assert_ne!(
        k_1e, k_1o,
        "1 de Espadas and 1 de Oros must NOT have same canonical key"
    );

    // Hand with 7 de Espadas (6, rank 12) vs 7 de Oros (26, rank 11) vs 7 de Copas (36, rank 4)
    let s_7e = BitboardState::new([[6, 4, 5], dummy_opp], 0);
    let s_7o = BitboardState::new([[26, 4, 5], dummy_opp], 0);
    let s_7c = BitboardState::new([[36, 4, 5], dummy_opp], 0);

    let k_7e = s_7e.canonical_infoset_key(0);
    let k_7o = s_7o.canonical_infoset_key(0);
    let k_7c = s_7c.canonical_infoset_key(0);

    assert_ne!(k_7e, k_7o, "7 de Espadas and 7 de Oros must differ");
    assert_ne!(k_7o, k_7c, "7 de Oros and 7 de Copas must differ");
    assert_ne!(k_7e, k_7c, "7 de Espadas and 7 de Copas must differ");
}

#[test]
fn test_hand_permutation_invariance() {
    let dummy_opp = [10, 11, 12];

    // Permutations of [1 de Espadas, 7 de Oros, 3 de Copas]
    let s1 = BitboardState::new([[0, 26, 32], dummy_opp], 0);
    let s2 = BitboardState::new([[26, 32, 0], dummy_opp], 0);
    let s3 = BitboardState::new([[32, 0, 26], dummy_opp], 0);

    assert_eq!(
        s1.canonical_infoset_key(0),
        s2.canonical_infoset_key(0),
        "Key must be invariant to card deal ordering"
    );
    assert_eq!(
        s2.canonical_infoset_key(0),
        s3.canonical_infoset_key(0),
        "Key must be invariant to card deal ordering"
    );
}

#[test]
fn test_canonical_hand_compression_1812_classes() {
    // Across all 9,880 possible 3-card hands from 40 cards,
    // verify canonical hand compression results in exactly 1812 canonical forms.
    let mut canonical_set = HashSet::new();

    for c1 in 0..40u8 {
        for c2 in (c1 + 1)..40u8 {
            for c3 in (c2 + 1)..40u8 {
                let canon = canonicalize_hand(&[c1, c2, c3]);
                canonical_set.insert(canon);
            }
        }
    }

    assert_eq!(
        canonical_set.len(),
        1812,
        "Total canonical hand equivalence classes must match Python reference: exactly 1812"
    );
}

#[test]
fn test_isomorphic_transition_consistency() {
    let dummy_opp = [10, 11, 12];

    // Two isomorphic starting states
    let s_esp = BitboardState::new([[3, 4, 5], dummy_opp], 0);
    let s_bas = BitboardState::new([[13, 14, 15], dummy_opp], 0);

    // Both play their lowest card (index 0)
    let next_esp = s_esp.step(ACTION_PLAY_CARD_0);
    let next_bas = s_bas.step(ACTION_PLAY_CARD_0);

    assert_eq!(
        next_esp.canonical_infoset_key(1),
        next_bas.canonical_infoset_key(1),
        "Child infoset keys for opponent must remain isomorphic"
    );
}
