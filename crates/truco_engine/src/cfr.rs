//! Chance-Sampled CFR (both players traverse full combinatorial tree, no action sampling).

use rayon::prelude::*;
use std::sync::Arc;

use crate::state::BitboardState;
use crate::table::SharedPolicyTable;

pub fn train_cfr_parallel(
    table: Arc<SharedPolicyTable>,
    iterations: usize,
    threads: usize,
    cfr_plus: bool,
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .expect("Failed to build rayon thread pool");

    pool.install(|| {
        (0..iterations).into_par_iter().for_each(|iter| {
            // Partial Fisher-Yates for 6 cards using fast deterministic PRNG from iter
            let mut seed = (iter as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
            let mut deck: [u8; 40] = std::array::from_fn(|i| i as u8);
            for i in 0..6 {
                // Xorshift64star
                seed ^= seed >> 12;
                seed ^= seed << 25;
                seed ^= seed >> 27;
                let r = (seed.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as usize;
                let j = i + (r % (40 - i));
                deck.swap(i, j);
            }

            let hands = [
                [deck[0], deck[1], deck[2]],
                [deck[3], deck[4], deck[5]],
            ];
            let mano = (iter % 2) as u8;
            let initial_state = BitboardState::new(hands, mano);

            // CFR: update both players with exact counterfactual reach weights
            for update_player in 0..=1 {
                cfr_traverse(&table, initial_state, update_player, 1.0, 1.0, cfr_plus);
            }
        });
    });
}

fn cfr_traverse(
    table: &SharedPolicyTable,
    state: BitboardState,
    update_player: u8,
    pi_0: f32,
    pi_1: f32,
    cfr_plus: bool,
) -> f32 {
    if state.is_done() {
        let pts = state.points_won() as f32;
        let winner = state.winner().unwrap_or(state.mano());
        return if winner == update_player { pts } else { -pts };
    }

    let mask = state.legal_actions_mask();
    if mask == 0 {
        return 0.0;
    }

    let mut actions = [0u8; 8];
    let mut num_actions = 0;
    for a in 0..26u8 {
        if (mask & (1 << a)) != 0 && num_actions < 8 {
            actions[num_actions] = a;
            num_actions += 1;
        }
    }

    let active = state.active_player();
    let key = state.canonical_infoset_key(active);
    let legal_submask = (1u32 << num_actions) - 1;
    let slot = table.get_or_create(key, legal_submask);

    let mut strategy = [0.0f32; 8];
    slot.compute_regret_matching(legal_submask, &mut strategy);

    let mut action_values = [0.0f32; 8];
    let mut node_value = 0.0f32;

    for i in 0..num_actions {
        let next_state = state.step(actions[i]);
        let next_pi_0 = if active == 0 { pi_0 * strategy[i] } else { pi_0 };
        let next_pi_1 = if active == 1 { pi_1 * strategy[i] } else { pi_1 };

        let v = cfr_traverse(
            table,
            next_state,
            update_player,
            next_pi_0,
            next_pi_1,
            cfr_plus,
        );
        action_values[i] = v;
        node_value += strategy[i] * v;
    }

    if active == update_player {
        // Counterfactual regret weighted by opponent reach probability
        let opp_pi = if update_player == 0 { pi_1 } else { pi_0 };
        for i in 0..num_actions {
            let regret = opp_pi * (action_values[i] - node_value);
            slot.update_regret(i, regret, cfr_plus);
        }
    } else {
        // Accumulate average strategy weighted by active player's reach probability
        let active_pi = if active == 0 { pi_0 } else { pi_1 };
        for i in 0..num_actions {
            slot.accumulate_strategy(i, strategy[i], active_pi);
        }
    }

    node_value
}
