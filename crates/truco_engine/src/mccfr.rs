//! External Sampling MCCFR+ with Rayon parallel worker loop.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

use crate::state::BitboardState;
use crate::table::SharedPolicyTable;

pub fn train_parallel(
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
            // ponytail: thread-local SmallRng seeded from iteration index for speed
            let mut rng = SmallRng::seed_from_u64(0x9e3779b97f4a7c15 ^ (iter as u64));

            // Deal 6 random cards from 40
            let mut deck = [0u8; 40];
            for i in 0..40 {
                deck[i] = i as u8;
            }
            // Partial Fisher-Yates for 6 cards
            for i in 0..6 {
                let j = rng.gen_range(i..40);
                deck.swap(i, j);
            }

            let hands = [
                [deck[0], deck[1], deck[2]],
                [deck[3], deck[4], deck[5]],
            ];
            let mano = (iter % 2) as u8;
            let initial_state = BitboardState::new(hands, mano);

            // External sampling: update both players with opposite perspectives
            for update_player in 0..=1 {
                mccfr_traverse(&table, initial_state, update_player, cfr_plus, &mut rng);
            }
        });
    });
}

fn mccfr_traverse(
    table: &SharedPolicyTable,
    state: BitboardState,
    update_player: u8,
    cfr_plus: bool,
    rng: &mut SmallRng,
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

    let key = state.canonical_infoset_key(state.active_player());
    let legal_submask = (1u32 << num_actions) - 1;
    let slot = table.get_or_create(key, legal_submask);

    let mut strategy = [0.0f32; 8];
    slot.compute_regret_matching(legal_submask, &mut strategy);

    if state.active_player() == update_player {
        let mut action_values = [0.0f32; 8];
        let mut node_value = 0.0f32;

        for i in 0..num_actions {
            let next_state = state.step(actions[i]);
            let v = mccfr_traverse(table, next_state, update_player, cfr_plus, rng);
            action_values[i] = v;
            node_value += strategy[i] * v;
        }

        for i in 0..num_actions {
            let regret = action_values[i] - node_value;
            slot.update_regret(i, regret, cfr_plus);
        }

        node_value
    } else {
        let mut r: f32 = rng.gen();
        let mut sampled_idx = 0;
        for i in 0..num_actions {
            if r <= strategy[i] || i == num_actions - 1 {
                sampled_idx = i;
                break;
            }
            r -= strategy[i];
        }

        for i in 0..num_actions {
            slot.accumulate_strategy(i, strategy[i], 1.0);
        }

        let next_state = state.step(actions[sampled_idx]);
        mccfr_traverse(table, next_state, update_player, cfr_plus, rng)
    }
}
