//! Game rules: envido calculation, hand resolution, and falta envido point determination.

use crate::card::{CARD_ENVIDO, CARD_SUITS};

pub fn calculate_envido(cards: &[u8]) -> u8 {
    if cards.is_empty() {
        return 0;
    }
    if cards.len() == 1 {
        return CARD_ENVIDO[cards[0] as usize];
    }

    // ponytail: stack-allocated suits buffer for 0..3 cards; zero heap allocation
    let mut suit_cards: [[u8; 3]; 4] = [[0; 3]; 4];
    let mut suit_counts: [usize; 4] = [0; 4];
    let mut max_single = 0u8;

    for &c in cards {
        let s = CARD_SUITS[c as usize] as usize;
        let e = CARD_ENVIDO[c as usize];
        if e > max_single {
            max_single = e;
        }
        suit_cards[s][suit_counts[s]] = e;
        suit_counts[s] += 1;
    }

    let mut max_suited = 0u8;
    for s in 0..4 {
        match suit_counts[s] {
            2 => {
                let val = 20 + suit_cards[s][0] + suit_cards[s][1];
                if val > max_suited {
                    max_suited = val;
                }
            }
            3 => {
                let mut v = [suit_cards[s][0], suit_cards[s][1], suit_cards[s][2]];
                v.sort_unstable();
                let val = 20 + v[1] + v[2];
                if val > max_suited {
                    max_suited = val;
                }
            }
            _ => {}
        }
    }

    if max_suited > 0 {
        max_suited.max(max_single)
    } else {
        max_single
    }
}

pub fn resolve_hand(trick_results: &[i8], mano: u8) -> Option<u8> {
    let p0 = trick_results.iter().filter(|&&r| r == 1).count();
    let p1 = trick_results.iter().filter(|&&r| r == -1).count();

    if p0 >= 2 {
        return Some(0);
    }
    if p1 >= 2 {
        return Some(1);
    }

    if trick_results.len() >= 2 {
        // T1 won, T2 parda -> T1 winner wins
        if trick_results[0] != 0 && trick_results[1] == 0 {
            return Some(if trick_results[0] == 1 { 0 } else { 1 });
        }
        // T1 parda, T2 won -> T2 winner wins
        if trick_results[0] == 0 && trick_results[1] != 0 {
            return Some(if trick_results[1] == 1 { 0 } else { 1 });
        }
    }

    if trick_results.len() == 3 {
        // T3 parda -> T1 winner wins
        if trick_results[2] == 0 && trick_results[0] != 0 {
            return Some(if trick_results[0] == 1 { 0 } else { 1 });
        }
        // T1 parda, T2 parda, T3 decided -> T3 winner wins
        if trick_results[0] == 0 && trick_results[1] == 0 && trick_results[2] != 0 {
            return Some(if trick_results[2] == 1 { 0 } else { 1 });
        }
        // All parda -> mano wins
        if trick_results.iter().filter(|&&r| r == 0).count() == 3 {
            return Some(mano);
        }
    }

    None
}

#[inline(always)]
pub fn calculate_falta_envido_points(score_p0: u8, score_p1: u8, max_score: u8) -> u8 {
    // ponytail: saturating sub prevents underflow when already at max score
    max_score.saturating_sub(score_p0.max(score_p1))
}
