//! Unit and invariant tests for card representations, ranks, envido calculations,
//! and hand resolution matching the exact logic from Python (`truco_bot.core.rules` and `card.py`).

use truco_engine::card::{
    compare_cards, CARD_ENVIDO, CARD_RANKS, CARD_SUITS,
};
use truco_engine::rules::{calculate_envido, calculate_falta_envido_points, resolve_hand};

#[test]
fn test_card_ranks_hierarchy() {
    // Espadas
    assert_eq!(CARD_RANKS[0], 14, "1 de Espadas must be rank 14");
    assert_eq!(CARD_RANKS[1], 9, "2 de Espadas must be rank 9");
    assert_eq!(CARD_RANKS[2], 10, "3 de Espadas must be rank 10");
    assert_eq!(CARD_RANKS[3], 1, "4 de Espadas must be rank 1");
    assert_eq!(CARD_RANKS[4], 2, "5 de Espadas must be rank 2");
    assert_eq!(CARD_RANKS[5], 3, "6 de Espadas must be rank 3");
    assert_eq!(CARD_RANKS[6], 12, "7 de Espadas must be rank 12");
    assert_eq!(CARD_RANKS[7], 5, "10 de Espadas must be rank 5");
    assert_eq!(CARD_RANKS[8], 6, "11 de Espadas must be rank 6");
    assert_eq!(CARD_RANKS[9], 7, "12 de Espadas must be rank 7");

    // Bastos
    assert_eq!(CARD_RANKS[10], 13, "1 de Bastos must be rank 13");
    assert_eq!(CARD_RANKS[16], 4, "7 de Bastos must be rank 4");

    // Oros
    assert_eq!(CARD_RANKS[20], 8, "1 de Oros must be rank 8");
    assert_eq!(CARD_RANKS[26], 11, "7 de Oros must be rank 11");

    // Copas
    assert_eq!(CARD_RANKS[30], 8, "1 de Copas must be rank 8");
    assert_eq!(CARD_RANKS[36], 4, "7 de Copas must be rank 4");
}

#[test]
fn test_card_suits_and_envido_tables() {
    for c in 0..40u8 {
        let expected_suit = c / 10;
        assert_eq!(CARD_SUITS[c as usize], expected_suit);

        let num_idx = c % 10;
        let expected_envido = match num_idx {
            0 => 1,
            1 => 2,
            2 => 3,
            3 => 4,
            4 => 5,
            5 => 6,
            6 => 7,
            7 | 8 | 9 => 0,
            _ => unreachable!(),
        };
        assert_eq!(CARD_ENVIDO[c as usize], expected_envido);
    }
}

#[test]
fn test_card_comparisons_properties() {
    // 1. Reflexivity: compare(c, c) == 0
    for c in 0..40u8 {
        assert_eq!(compare_cards(c, c), 0, "Card {c} should tie with itself");
    }

    // 2. Antisymmetry: compare(c1, c2) == -compare(c2, c1)
    for c1 in 0..40u8 {
        for c2 in 0..40u8 {
            assert_eq!(
                compare_cards(c1, c2),
                -compare_cards(c2, c1),
                "Antisymmetry violation between {c1} and {c2}"
            );
        }
    }

    // 3. Transitivity
    for c1 in 0..40u8 {
        for c2 in 0..40u8 {
            if compare_cards(c1, c2) > 0 {
                for c3 in 0..40u8 {
                    if compare_cards(c2, c3) > 0 {
                        assert!(
                            compare_cards(c1, c3) > 0,
                            "Transitivity violation: {c1} > {c2} > {c3} but {c1} <= {c3}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_card_comparisons_known_matchups() {
    let ancho_espadas = 0u8;
    let ancho_bastos = 10u8;
    let siete_espadas = 6u8;
    let siete_oros = 26u8;
    let tres_copas = 32u8;
    let tres_bastos = 12u8;
    let dos_espadas = 1u8;
    let ancho_copas = 30u8;
    let ancho_oros = 20u8;
    let doce_espadas = 9u8;
    let diez_bastos = 17u8;
    let siete_copas = 36u8;
    let siete_bastos = 16u8;
    let cuatro_espadas = 3u8;
    let cuatro_copas = 33u8;

    // 1 de Espadas beats everything
    assert_eq!(compare_cards(ancho_espadas, ancho_bastos), 1);
    assert_eq!(compare_cards(ancho_espadas, siete_espadas), 1);
    assert_eq!(compare_cards(ancho_espadas, tres_copas), 1);

    // 1 de Bastos beats 7 de Espadas
    assert_eq!(compare_cards(ancho_bastos, siete_espadas), 1);
    assert_eq!(compare_cards(ancho_bastos, siete_oros), 1);

    // 7 de Espadas beats 7 de Oros
    assert_eq!(compare_cards(siete_espadas, siete_oros), 1);

    // 7 de Oros beats 3s
    assert_eq!(compare_cards(siete_oros, tres_copas), 1);

    // 3s tie with 3s
    assert_eq!(compare_cards(tres_copas, tres_bastos), 0);

    // 3 beats 2
    assert_eq!(compare_cards(tres_copas, dos_espadas), 1);

    // 2 beats 1 de Oros / 1 de Copas
    assert_eq!(compare_cards(dos_espadas, ancho_oros), 1);
    assert_eq!(compare_cards(dos_espadas, ancho_copas), 1);

    // 1 de Oros ties with 1 de Copas
    assert_eq!(compare_cards(ancho_oros, ancho_copas), 0);

    // 1 de Copas beats 12
    assert_eq!(compare_cards(ancho_copas, doce_espadas), 1);

    // 10 beats 7 falso
    assert_eq!(compare_cards(diez_bastos, siete_copas), 1);

    // 7s falsos tie
    assert_eq!(compare_cards(siete_copas, siete_bastos), 0);

    // 7 falso beats 4
    assert_eq!(compare_cards(siete_bastos, cuatro_espadas), 1);

    // 4 ties with 4
    assert_eq!(compare_cards(cuatro_espadas, cuatro_copas), 0);
}

#[test]
fn test_envido_calculation() {
    // Empty hand -> 0
    assert_eq!(calculate_envido(&[]), 0);

    // Single card
    assert_eq!(calculate_envido(&[0]), 1); // 1 de espadas
    assert_eq!(calculate_envido(&[6]), 7); // 7 de espadas
    assert_eq!(calculate_envido(&[7]), 0); // 10 de espadas
    assert_eq!(calculate_envido(&[9]), 0); // 12 de espadas

    // Two cards different suits: max single
    assert_eq!(calculate_envido(&[6, 15]), 7); // 7 de espadas + 6 de bastos -> 7
    assert_eq!(calculate_envido(&[7, 18]), 0); // 10 de espadas + 11 de bastos -> 0

    // Two cards same suit: c1 + c2 + 20
    assert_eq!(calculate_envido(&[6, 5]), 33); // 7 + 6 de espadas -> 33
    assert_eq!(calculate_envido(&[6, 4]), 32); // 7 + 5 de espadas -> 32
    assert_eq!(calculate_envido(&[0, 1]), 23); // 1 + 2 de espadas -> 23
    assert_eq!(calculate_envido(&[7, 8]), 20); // 10 + 11 de espadas -> 20
    assert_eq!(calculate_envido(&[6, 9]), 27); // 7 + 12 de espadas -> 27

    // Three cards all same suit: best two + 20
    assert_eq!(calculate_envido(&[6, 5, 4]), 33); // 7 + 6 + 5 espadas -> 7 + 6 + 20 = 33
    assert_eq!(calculate_envido(&[6, 4, 3]), 32); // 7 + 5 + 4 espadas -> 7 + 5 + 20 = 32
    assert_eq!(calculate_envido(&[7, 8, 9]), 20); // 10 + 11 + 12 espadas -> 20

    // Three cards, two same suit: best pair vs single
    assert_eq!(calculate_envido(&[6, 5, 16]), 33); // 7e, 6e, 7b -> 33
    assert_eq!(calculate_envido(&[7, 8, 16]), 20); // 10e, 11e, 7b -> max(20, 7) = 20
    assert_eq!(calculate_envido(&[7, 8, 26]), 20); // 10e, 11e, 7o -> max(20, 7) = 20

    // Three cards, three different suits: max single
    assert_eq!(calculate_envido(&[6, 15, 24]), 7); // 7e, 6b, 5o -> 7
    assert_eq!(calculate_envido(&[7, 18, 29]), 0); // 10e, 11b, 12o -> 0
}

#[test]
fn test_envido_exhaustive_properties() {
    // Check all 9880 hands of 3 cards
    let mut count = 0;
    for c1 in 0..40u8 {
        for c2 in (c1 + 1)..40u8 {
            for c3 in (c2 + 1)..40u8 {
                let score = calculate_envido(&[c1, c2, c3]);
                count += 1;
                // Envido must be in [0..7] or [20..33]
                let valid_range = (score <= 7) || (score >= 20 && score <= 33);
                assert!(
                    valid_range,
                    "Invalid envido score {score} for hand [{c1}, {c2}, {c3}]"
                );
            }
        }
    }
    assert_eq!(count, 9880);
}

#[test]
fn test_resolve_hand_all_scenarios() {
    // 2-0 immediate
    assert_eq!(resolve_hand(&[1, 1], 0), Some(0));
    assert_eq!(resolve_hand(&[1, 1], 1), Some(0));
    assert_eq!(resolve_hand(&[-1, -1], 0), Some(1));
    assert_eq!(resolve_hand(&[-1, -1], 1), Some(1));

    // 2-1 split
    assert_eq!(resolve_hand(&[1, -1, 1], 0), Some(0));
    assert_eq!(resolve_hand(&[-1, 1, -1], 0), Some(1));
    assert_eq!(resolve_hand(&[1, -1, 1], 1), Some(0));
    assert_eq!(resolve_hand(&[-1, 1, -1], 1), Some(1));

    // First won, second parda -> first winner wins
    assert_eq!(resolve_hand(&[1, 0], 0), Some(0));
    assert_eq!(resolve_hand(&[1, 0], 1), Some(0));
    assert_eq!(resolve_hand(&[-1, 0], 0), Some(1));
    assert_eq!(resolve_hand(&[-1, 0], 1), Some(1));

    // First parda, second won -> second winner wins
    assert_eq!(resolve_hand(&[0, 1], 0), Some(0));
    assert_eq!(resolve_hand(&[0, 1], 1), Some(0));
    assert_eq!(resolve_hand(&[0, -1], 0), Some(1));
    assert_eq!(resolve_hand(&[0, -1], 1), Some(1));

    // First won, second lost, third parda -> first trick winner wins
    assert_eq!(resolve_hand(&[1, -1, 0], 0), Some(0));
    assert_eq!(resolve_hand(&[1, -1, 0], 1), Some(0));
    assert_eq!(resolve_hand(&[-1, 1, 0], 0), Some(1));
    assert_eq!(resolve_hand(&[-1, 1, 0], 1), Some(1));

    // T1 parda, T2 parda, T3 decided -> T3 winner wins
    assert_eq!(resolve_hand(&[0, 0, 1], 0), Some(0));
    assert_eq!(resolve_hand(&[0, 0, 1], 1), Some(0));
    assert_eq!(resolve_hand(&[0, 0, -1], 0), Some(1));
    assert_eq!(resolve_hand(&[0, 0, -1], 1), Some(1));

    // All parda -> mano wins
    assert_eq!(resolve_hand(&[0, 0, 0], 0), Some(0));
    assert_eq!(resolve_hand(&[0, 0, 0], 1), Some(1));

    // Incomplete hands return None
    assert_eq!(resolve_hand(&[], 0), None);
    assert_eq!(resolve_hand(&[1], 0), None);
    assert_eq!(resolve_hand(&[-1], 0), None);
    assert_eq!(resolve_hand(&[0], 0), None);
    assert_eq!(resolve_hand(&[1, -1], 0), None);
    assert_eq!(resolve_hand(&[-1, 1], 0), None);
    assert_eq!(resolve_hand(&[0, 0], 0), None);
}

#[test]
fn test_calculate_falta_envido() {
    assert_eq!(calculate_falta_envido_points(0, 0, 30), 30);
    assert_eq!(calculate_falta_envido_points(10, 20, 30), 10);
    assert_eq!(calculate_falta_envido_points(28, 25, 30), 2);
    assert_eq!(calculate_falta_envido_points(15, 12, 15), 0);
}
