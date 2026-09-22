//! Suit isomorphism normalizer and canonical 64-bit infoset hash.
//! Compresses all 9,880 3-card hands to exactly 1,812 canonical equivalence classes.

use crate::card::{card_number, CARD_ENVIDO, CARD_RANKS, CARD_SUITS};

pub fn canonicalize_hand(hand: &[u8]) -> [u8; 3] {
    if hand.is_empty() {
        return [255, 255, 255];
    }

    let mut counts = [0u8; 4];
    for &c in hand {
        if c < 40 {
            counts[CARD_SUITS[c as usize] as usize] += 1;
        }
    }

    let mut present = [0usize; 4];
    let mut num_present = 0;
    for s in 0..4 {
        if counts[s] > 0 {
            present[num_present] = s;
            num_present += 1;
        }
    }

    // Sort present suits by (-count, -max_rank, -max_envido, suit)
    let active_suits = &mut present[..num_present];
    active_suits.sort_unstable_by(|&s1, &s2| {
        let cnt1 = counts[s1];
        let cnt2 = counts[s2];
        if cnt1 != cnt2 {
            return cnt2.cmp(&cnt1);
        }

        let max_r1 = hand
            .iter()
            .filter(|&&c| c < 40 && CARD_SUITS[c as usize] as usize == s1)
            .map(|&c| CARD_RANKS[c as usize])
            .max()
            .unwrap_or(0);
        let max_r2 = hand
            .iter()
            .filter(|&&c| c < 40 && CARD_SUITS[c as usize] as usize == s2)
            .map(|&c| CARD_RANKS[c as usize])
            .max()
            .unwrap_or(0);
        if max_r1 != max_r2 {
            return max_r2.cmp(&max_r1);
        }

        let max_e1 = hand
            .iter()
            .filter(|&&c| c < 40 && CARD_SUITS[c as usize] as usize == s1)
            .map(|&c| CARD_ENVIDO[c as usize])
            .max()
            .unwrap_or(0);
        let max_e2 = hand
            .iter()
            .filter(|&&c| c < 40 && CARD_SUITS[c as usize] as usize == s2)
            .map(|&c| CARD_ENVIDO[c as usize])
            .max()
            .unwrap_or(0);
        if max_e1 != max_e2 {
            return max_e2.cmp(&max_e1);
        }

        s1.cmp(&s2)
    });

    let mut suit_map = [0u8; 4];
    for (i, &s) in active_suits.iter().enumerate() {
        suit_map[s] = i as u8;
    }

    // ponytail: stack buffer for canonical cards
    let valid_count = hand.iter().filter(|&&c| c < 40).count();
    let mut canon = [(0u8, 0u8, 0u8, 0u8); 3];
    let mut idx = 0;
    for &c in hand {
        if c < 40 {
            let ns = suit_map[CARD_SUITS[c as usize] as usize];
            let r = CARD_RANKS[c as usize];
            let e = CARD_ENVIDO[c as usize];
            let n = card_number(c);
            canon[idx] = (r, e, n, ns);
            idx += 1;
        }
    }

    let active_canon = &mut canon[..valid_count];
    active_canon.sort_unstable_by(|a, b| {
        // Python sort key: (-c.truco_rank, -c.envido_value, c.number, c.suit.value)
        if a.0 != b.0 {
            b.0.cmp(&a.0)
        } else if a.1 != b.1 {
            b.1.cmp(&a.1)
        } else if a.2 != b.2 {
            a.2.cmp(&b.2)
        } else {
            a.3.cmp(&b.3)
        }
    });

    let mut res = [255u8; 3];
    for (i, item) in active_canon.iter().enumerate() {
        res[i] = item.3 * 14 + (item.0 - 1);
    }
    res
}
