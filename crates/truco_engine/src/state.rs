//! BitboardState: exactly 32 bytes zero-allocation copy state machine.

use crate::actions::*;
use crate::card::{compare_cards, CARD_RANKS};
use crate::isomorphism::canonicalize_hand;
use crate::rules::{calculate_envido, calculate_falta_envido_points, resolve_hand};

const FLAG_ENVIDO_RESOLVED: u8 = 1 << 0;
const FLAG_PENDING_ENVIDO: u8 = 1 << 1;
const FLAG_PENDING_TRUCO: u8 = 1 << 2;
const FLAG_IS_HAND_DONE: u8 = 1 << 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct BitboardState {
    pub hands: [[u8; 3]; 2],     // 6 bytes (0..39 or 255 if played)
    pub trick_cards: [u8; 2],    // 2 bytes (cards in current trick, 255 if empty)
    pub trick_results: [i8; 3],  // 3 bytes (1 for P0, -1 for P1, 0 for parda, 127 if unplayed)
    pub trick_leader: u8,        // 1 byte
    pub current_trick: u8,       // 1 byte
    pub active_player: u8,       // 1 byte
    pub mano: u8,                // 1 byte
    pub score_p0: u8,            // 1 byte
    pub score_p1: u8,            // 1 byte
    pub max_score: u8,           // 1 byte
    pub envido_chain: [u8; 6],   // 6 bytes (actions in envido chain, 255 if empty)
    pub envido_chain_len: u8,    // 1 byte
    pub pending_envido_from: u8, // 1 byte (0, 1, or 255)
    pub truco_level: u8,         // 1 byte (1..4)
    pub truco_caller: u8,        // 1 byte (0, 1, or 255)
    pub pending_truco_from: u8,  // 1 byte (0, 1, or 255)
    pub flags: u8,               // 1 byte (packed booleans)
    pub winner: u8,              // 1 byte (0, 1, or 255)
    pub points_won: u8,          // 1 byte
}

// Compile-time assertion for exactly 32-byte size
const _: () = assert!(std::mem::size_of::<BitboardState>() == 32);

impl BitboardState {
    pub fn new(hands: [[u8; 3]; 2], mano: u8) -> Self {
        Self {
            hands,
            trick_cards: [255, 255],
            trick_results: [127, 127, 127],
            trick_leader: mano,
            current_trick: 0,
            active_player: mano,
            mano,
            score_p0: 0,
            score_p1: 0,
            max_score: 30,
            envido_chain: [255; 6],
            envido_chain_len: 0,
            pending_envido_from: 255,
            truco_level: 1,
            truco_caller: 255,
            pending_truco_from: 255,
            flags: 0,
            winner: 255,
            points_won: 0,
        }
    }

    #[inline(always)]
    pub fn active_player(&self) -> u8 {
        self.active_player
    }

    #[inline(always)]
    pub fn mano(&self) -> u8 {
        self.mano
    }

    #[inline(always)]
    pub fn trick_leader(&self) -> u8 {
        self.trick_leader
    }

    #[inline(always)]
    pub fn current_trick(&self) -> u8 {
        self.current_trick
    }

    #[inline(always)]
    pub fn truco_level(&self) -> u8 {
        self.truco_level
    }

    #[inline(always)]
    pub fn envido_resolved(&self) -> bool {
        (self.flags & FLAG_ENVIDO_RESOLVED) != 0
    }

    #[inline(always)]
    pub fn is_done(&self) -> bool {
        (self.flags & FLAG_IS_HAND_DONE) != 0
    }

    #[inline(always)]
    pub fn winner(&self) -> Option<u8> {
        if self.winner != 255 {
            Some(self.winner)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn points_won(&self) -> u8 {
        self.points_won
    }

    #[inline(always)]
    pub fn score(&self) -> (u8, u8) {
        (self.score_p0, self.score_p1)
    }

    pub fn legal_actions_mask(&self) -> u32 {
        if self.is_done() {
            return 0;
        }

        let player = self.active_player;

        if (self.flags & FLAG_PENDING_ENVIDO) != 0 {
            let last_call = self.envido_chain[(self.envido_chain_len - 1) as usize];
            if last_call == ACTION_ENVIDO {
                let envido_count = self.envido_chain[..self.envido_chain_len as usize]
                    .iter()
                    .filter(|&&a| a == ACTION_ENVIDO)
                    .count();
                if envido_count >= 2 {
                    return (1 << ACTION_QUIERO_ENVIDO)
                        | (1 << ACTION_NO_QUIERO_ENVIDO)
                        | (1 << ACTION_REAL_ENVIDO)
                        | (1 << ACTION_FALTA_ENVIDO);
                }
                return (1 << ACTION_QUIERO_ENVIDO)
                    | (1 << ACTION_NO_QUIERO_ENVIDO)
                    | (1 << ACTION_ENVIDO)
                    | (1 << ACTION_REAL_ENVIDO)
                    | (1 << ACTION_FALTA_ENVIDO);
            } else if last_call == ACTION_REAL_ENVIDO {
                return (1 << ACTION_QUIERO_ENVIDO)
                    | (1 << ACTION_NO_QUIERO_ENVIDO)
                    | (1 << ACTION_FALTA_ENVIDO);
            }
            return (1 << ACTION_QUIERO_ENVIDO) | (1 << ACTION_NO_QUIERO_ENVIDO);
        }

        if (self.flags & FLAG_PENDING_TRUCO) != 0 {
            let mut mask = (1 << ACTION_QUIERO_TRUCO)
                | (1 << ACTION_NO_QUIERO_TRUCO)
                | (1 << ACTION_IR_AL_MAZO);
            if self.truco_level == 1 {
                mask |= 1 << ACTION_RETRUCO;
            } else if self.truco_level == 2 {
                mask |= 1 << ACTION_VALE_CUATRO;
            }
            return mask;
        }

        let mut mask = 0u32;

        // Envido calls
        if self.current_trick == 0
            && !self.envido_resolved()
            && self.trick_cards[1] == 255
            && self.envido_chain_len == 0
        {
            mask |= (1 << ACTION_ENVIDO) | (1 << ACTION_REAL_ENVIDO) | (1 << ACTION_FALTA_ENVIDO);
        }

        // Truco raises
        if self.truco_level == 1 {
            mask |= 1 << ACTION_TRUCO;
        } else if self.truco_level == 2 && self.truco_caller != player {
            mask |= 1 << ACTION_RETRUCO;
        } else if self.truco_level == 3 && self.truco_caller != player {
            mask |= 1 << ACTION_VALE_CUATRO;
        }

        // Card plays
        let hand = &self.hands[player as usize];
        let remaining = hand.iter().filter(|&&c| c < 40).count();
        if remaining > 0 {
            mask |= (1 << ACTION_PLAY_CARD_0) | (1 << ACTION_PLAY_CARD_DOWN_0);
        }
        if remaining > 1 {
            mask |= (1 << ACTION_PLAY_CARD_1) | (1 << ACTION_PLAY_CARD_DOWN_1);
        }
        if remaining > 2 {
            mask |= (1 << ACTION_PLAY_CARD_2) | (1 << ACTION_PLAY_CARD_DOWN_2);
        }

        mask |= 1 << ACTION_IR_AL_MAZO;
        mask
    }

    pub fn step(&self, action: u8) -> Self {
        let mut next = *self;
        let player = next.active_player;
        let other = 1 - player;

        if action == ACTION_IR_AL_MAZO {
            next.flags |= FLAG_IS_HAND_DONE;
            next.winner = other;
            next.points_won = next.truco_level;
            return next;
        }

        if action == ACTION_ENVIDO || action == ACTION_REAL_ENVIDO || action == ACTION_FALTA_ENVIDO {
            if (next.envido_chain_len as usize) < next.envido_chain.len() {
                next.envido_chain[next.envido_chain_len as usize] = action;
                next.envido_chain_len += 1;
            }
            next.flags |= FLAG_PENDING_ENVIDO;
            next.pending_envido_from = other;
            next.active_player = other;
            return next;
        }

        if action == ACTION_QUIERO_ENVIDO {
            if (next.envido_chain_len as usize) < next.envido_chain.len() {
                next.envido_chain[next.envido_chain_len as usize] = action;
                next.envido_chain_len += 1;
            }
            next.flags &= !FLAG_PENDING_ENVIDO;
            next.flags |= FLAG_ENVIDO_RESOLVED;
            next.active_player = if next.trick_cards[0] != 255 && next.trick_cards[1] == 255 {
                other
            } else {
                next.trick_leader
            };

            let e0 = calculate_envido(&next.hands[0]);
            let e1 = calculate_envido(&next.hands[1]);
            let env_winner = if e0 > e1 {
                0
            } else if e1 > e0 {
                1
            } else {
                next.mano
            };

            let pts = if next.envido_chain[..next.envido_chain_len as usize].contains(&ACTION_FALTA_ENVIDO) {
                calculate_falta_envido_points(next.score_p0, next.score_p1, next.max_score)
            } else {
                let mut p = 0u8;
                for &a in &next.envido_chain[..next.envido_chain_len as usize] {
                    if a == ACTION_ENVIDO {
                        p += 2;
                    } else if a == ACTION_REAL_ENVIDO {
                        p += 3;
                    }
                }
                p
            };

            if env_winner == 0 {
                next.score_p0 = (next.score_p0 + pts).min(next.max_score);
            } else {
                next.score_p1 = (next.score_p1 + pts).min(next.max_score);
            }
            return next;
        }

        if action == ACTION_NO_QUIERO_ENVIDO {
            if (next.envido_chain_len as usize) < next.envido_chain.len() {
                next.envido_chain[next.envido_chain_len as usize] = action;
                next.envido_chain_len += 1;
            }
            next.flags &= !FLAG_PENDING_ENVIDO;
            next.flags |= FLAG_ENVIDO_RESOLVED;
            next.active_player = if next.trick_cards[0] != 255 && next.trick_cards[1] == 255 {
                other
            } else {
                next.trick_leader
            };

            let caller = other;
            let chain = &next.envido_chain[..next.envido_chain_len as usize];
            let pts = if chain.len() == 2 {
                1
            } else {
                let mut p = 0u8;
                for &a in &chain[..chain.len() - 2] {
                    if a == ACTION_ENVIDO {
                        p += 2;
                    } else if a == ACTION_REAL_ENVIDO {
                        p += 3;
                    }
                }
                if p == 0 {
                    1
                } else {
                    p
                }
            };

            if caller == 0 {
                next.score_p0 = (next.score_p0 + pts).min(next.max_score);
            } else {
                next.score_p1 = (next.score_p1 + pts).min(next.max_score);
            }
            return next;
        }

        if action == ACTION_TRUCO || action == ACTION_RETRUCO || action == ACTION_VALE_CUATRO {
            next.flags |= FLAG_PENDING_TRUCO;
            next.pending_truco_from = other;
            next.truco_caller = player;
            next.active_player = other;
            if action == ACTION_RETRUCO {
                next.truco_level = 2;
            } else if action == ACTION_VALE_CUATRO {
                next.truco_level = 3;
            }
            next.flags |= FLAG_ENVIDO_RESOLVED;
            return next;
        }

        if action == ACTION_QUIERO_TRUCO {
            next.flags &= !FLAG_PENDING_TRUCO;
            next.truco_level += 1;
            next.active_player = if next.trick_cards[0] != 255 && next.trick_cards[1] == 255 {
                other
            } else {
                next.trick_leader
            };
            return next;
        }

        if action == ACTION_NO_QUIERO_TRUCO {
            next.flags |= FLAG_IS_HAND_DONE;
            next.winner = other;
            next.points_won = next.truco_level;
            return next;
        }

        // Card plays (0..5)
        if action <= 5 {
            next.flags |= FLAG_ENVIDO_RESOLVED;
            let card_idx = (action % 3) as usize;
            let played_card = next.hands[player as usize][card_idx];

            // Shift remaining cards to keep valid cards contiguous at start of hands[player]
            let mut new_hand = [255u8; 3];
            let mut new_idx = 0;
            for i in 0..3 {
                if i != card_idx && next.hands[player as usize][i] < 40 {
                    new_hand[new_idx] = next.hands[player as usize][i];
                    new_idx += 1;
                }
            }
            next.hands[player as usize] = new_hand;

            if next.trick_cards[0] == 255 {
                next.trick_cards[0] = played_card;
                next.active_player = other;
            } else {
                next.trick_cards[1] = played_card;
                let leader = next.trick_leader;
                let p0_card = if leader == 0 {
                    next.trick_cards[0]
                } else {
                    played_card
                };
                let p1_card = if leader == 1 {
                    next.trick_cards[0]
                } else {
                    played_card
                };

                let res = compare_cards(p0_card, p1_card);
                let trick_idx = (next.current_trick as usize).min(2);
                next.trick_results[trick_idx] = res;

                let hr = resolve_hand(&next.trick_results[..=trick_idx], next.mano);
                if let Some(winner) = hr {
                    next.flags |= FLAG_IS_HAND_DONE;
                    next.winner = winner;
                    next.points_won = next.truco_level;
                } else if next.current_trick >= 2 {
                    next.flags |= FLAG_IS_HAND_DONE;
                    next.winner = next.mano;
                    next.points_won = next.truco_level;
                } else {
                    next.current_trick += 1;
                    if res != 0 {
                        let trick_winner = if res == 1 { 0 } else { 1 };
                        next.trick_leader = trick_winner;
                        next.active_player = trick_winner;
                    } else {
                        next.active_player = next.trick_leader;
                    }
                    next.trick_cards = [255, 255];
                }
            }
        }

        next
    }

    pub fn canonical_infoset_key(&self, player: u8) -> u64 {
        let hand_slice = &self.hands[player as usize];
        let canon_hand = canonicalize_hand(hand_slice);

        // ponytail: fast FNV-1a hash over canonical hand, trick rank history, and state variables
        let mut h: u64 = 0xcbf29ce484222325;
        macro_rules! feed {
            ($b:expr) => {
                h = (h ^ ($b as u64)).wrapping_mul(0x100000001b3);
            };
        }

        feed!(canon_hand[0]);
        feed!(canon_hand[1]);
        feed!(canon_hand[2]);

        // Trick cards on table normalized to rank
        for &tc in &self.trick_cards {
            if tc < 40 {
                feed!(CARD_RANKS[tc as usize]);
            } else {
                feed!(255u8);
            }
        }

        feed!(self.trick_results[0] as u8);
        feed!(self.trick_results[1] as u8);
        feed!(self.trick_results[2] as u8);

        feed!(self.trick_leader);
        feed!(self.current_trick);
        feed!(self.mano);

        feed!(self.flags);
        feed!(self.score_p0);
        feed!(self.score_p1);
        feed!(self.pending_envido_from);
        feed!(self.envido_chain_len);
        for i in 0..(self.envido_chain_len as usize).min(6) {
            feed!(self.envido_chain[i]);
        }

        feed!(self.truco_level);
        feed!(self.truco_caller);
        feed!(self.pending_truco_from);

        if h == 0 {
            1
        } else {
            h
        }
    }
}
