//! Concurrency & Hogwild! stress test for `SharedPolicyTable`.
//! Hammers the shared policy table with 16 to 32 concurrent threads
//! performing lock-free atomic CAS insertions, regret accumulations,
//! and strategy weighting without deadlocks, data races, or memory leaks.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use truco_engine::table::SharedPolicyTable;

#[test]
fn test_table_concurrent_cas_insertions_no_deadlock() {
    let capacity = 65536; // 64K slots
    let table = Arc::new(SharedPolicyTable::new(capacity));

    let num_threads = 32;
    let keys_per_thread = 1000;
    let shared_keys_count = 50;

    let start = Instant::now();
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            // 1. Thread-private keys
            for i in 1..=keys_per_thread {
                let key = (t as u64 + 1) * 1_000_000 + (i as u64);
                let slot = table_clone.get_or_create(key, 0b111);
                assert_eq!(
                    slot.key.load(Ordering::Relaxed),
                    key,
                    "Slot key must match queried key"
                );
            }

            // 2. Shared keys hammered by all threads simultaneously
            for i in 1..=shared_keys_count {
                let shared_key = 999_000_000 + (i as u64);
                let slot = table_clone.get_or_create(shared_key, 0b1111);
                assert_eq!(slot.key.load(Ordering::Relaxed), shared_key);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked during CAS stress test");
    }

    let elapsed = start.elapsed();
    println!(
        "32-thread concurrent CAS insertion completed in {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );
    assert!(
        elapsed.as_secs() < 10,
        "Concurrent CAS insertion must complete promptly without deadlocks"
    );
}

#[test]
fn test_hogwild_atomic_regret_accumulation_accuracy() {
    let capacity = 1024;
    let table = Arc::new(SharedPolicyTable::new(capacity));

    let target_key = 42_424_242u64;
    let mask = 0b111;
    let slot = table.get_or_create(target_key, mask);

    let num_threads = 32;
    let increments_per_thread = 5_000;

    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            let slot = table_clone.get_or_create(target_key, mask);
            for _ in 0..increments_per_thread {
                slot.update_regret(0, 1.0, false);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked during regret update");
    }

    let final_regret = slot.regrets[0].load(Ordering::Relaxed);
    let expected = (num_threads * increments_per_thread) as f32;
    let diff = (final_regret - expected).abs();

    println!("Final accumulated regret: {final_regret}, Expected: {expected}");
    assert!(
        diff < 1.0,
        "Accumulation must be accurate across 32 threads: diff={diff}"
    );
}

#[test]
fn test_hogwild_cfr_plus_clipping_invariant() {
    // Under CFR+, cumulative regrets must never fall below 0.0
    let capacity = 1024;
    let table = Arc::new(SharedPolicyTable::new(capacity));

    let target_key = 88_888_888u64;
    let mask = 0b111;
    let slot = table.get_or_create(target_key, mask);

    let num_positive_threads = 16;
    let num_negative_threads = 16;
    let ops = 5_000;

    let mut handles = Vec::new();

    // Positive delta threads
    for _ in 0..num_positive_threads {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            let slot = table_clone.get_or_create(target_key, mask);
            for _ in 0..ops {
                slot.update_regret(1, 2.0, true);
            }
        }));
    }

    // Negative delta threads
    for _ in 0..num_negative_threads {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            let slot = table_clone.get_or_create(target_key, mask);
            for _ in 0..ops {
                slot.update_regret(1, -5.0, true);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked in CFR+ stress test");
    }

    let final_val = slot.regrets[1].load(Ordering::Relaxed);
    assert!(
        final_val >= 0.0,
        "CFR+ invariant violated: regret value {final_val} is negative"
    );
}

#[test]
fn test_strategy_sum_accumulation_accuracy() {
    let capacity = 1024;
    let table = Arc::new(SharedPolicyTable::new(capacity));
    let key = 123_456_789u64;
    let slot = table.get_or_create(key, 0b11);

    let num_threads = 32;
    let updates_per_thread = 5_000;
    let prob = 0.5f32;
    let weight = 2.0f32; // prob * weight = 1.0 per update

    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            let slot = table_clone.get_or_create(key, 0b11);
            for _ in 0..updates_per_thread {
                slot.accumulate_strategy(0, prob, weight);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked in strategy accumulation");
    }

    let final_sum = slot.strategy_sum[0].load(Ordering::Relaxed);
    let expected = (num_threads * updates_per_thread) as f32 * (prob * weight);
    let diff = (final_sum - expected).abs();

    println!("Final strategy sum: {final_sum}, Expected: {expected}");
    assert!(
        diff < 1.0,
        "Strategy sum accumulation difference too large: diff={diff}"
    );
}

#[test]
fn test_multi_threaded_throughput_benchmark() {
    let capacity = 262_144; // 256K slots
    let table = Arc::new(SharedPolicyTable::new(capacity));

    let num_threads = 16;
    let iterations_per_thread = 100_000;

    let start = Instant::now();
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let table_clone = Arc::clone(&table);
        handles.push(thread::spawn(move || {
            let base_key = (t as u64 + 1) * 10_000;
            for i in 0..iterations_per_thread {
                let key = base_key + (i % 1000) as u64;
                let slot = table_clone.get_or_create(key, 0b111);
                slot.update_regret((i % 3) as usize, 0.5, true);
                slot.accumulate_strategy((i % 3) as usize, 0.33, 1.0);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Throughput thread panicked");
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_operations = (num_threads * iterations_per_thread) as f64;
    let throughput = total_operations / elapsed;

    println!(
        "Total operations: {total_operations}, Time: {:.3}s, Throughput: {:.2} ops/sec",
        elapsed, throughput
    );
    assert!(
        throughput > 500_000.0,
        "Throughput must exceed 500k ops/sec on multi-threaded stress test"
    );
}

#[test]
fn test_get_strategy_average_strategy_and_regret_matching_fallback() {
    let capacity = 1024;
    let table = SharedPolicyTable::new(capacity);
    let key = 0xbeef_cafe_u64;
    let legal_mask = 0b101; // actions 0 and 2 are legal

    let slot = table.get_or_create(key, legal_mask);

    // 1. Initial state: both strategy_sum and regrets are empty (0.0).
    // get_strategy falls back from compute_average_strategy (sum <= 1e-6)
    // to compute_regret_matching, which yields uniform over legal actions.
    let initial_strat = table.get_strategy(key, legal_mask);
    assert!((initial_strat[0] - 0.5).abs() < 1e-6);
    assert_eq!(initial_strat[1], 0.0);
    assert!((initial_strat[2] - 0.5).abs() < 1e-6);

    // 2. Regrets are accumulated, but strategy_sum is still empty.
    // slot.update_regret for action 0: 3.0, action 2: 1.0.
    slot.update_regret(0, 3.0, false);
    slot.update_regret(2, 1.0, false);

    // compute_average_strategy must return false because strategy_sum is empty.
    let mut out_avg = [0.0f32; 8];
    assert!(!slot.compute_average_strategy(legal_mask, &mut out_avg));

    // get_strategy must fall back to regret matching:
    // action 0: 3.0 / 4.0 = 0.75, action 2: 1.0 / 4.0 = 0.25.
    let regret_strat = table.get_strategy(key, legal_mask);
    assert!((regret_strat[0] - 0.75).abs() < 1e-6);
    assert_eq!(regret_strat[1], 0.0);
    assert!((regret_strat[2] - 0.25).abs() < 1e-6);

    // 3. Strategy is accumulated (average strategy).
    // We accumulate strategy: action 0 with prob 0.2, weight 10.0 (sum = 2.0);
    // action 2 with prob 0.8, weight 10.0 (sum = 8.0).
    // Total legal sum = 10.0 > 1e-6.
    slot.accumulate_strategy(0, 0.2, 10.0);
    slot.accumulate_strategy(2, 0.8, 10.0);

    // Direct check of compute_average_strategy
    assert!(slot.compute_average_strategy(legal_mask, &mut out_avg));
    assert!((out_avg[0] - 0.2).abs() < 1e-6);
    assert_eq!(out_avg[1], 0.0);
    assert!((out_avg[2] - 0.8).abs() < 1e-6);
    assert!((out_avg[0] + out_avg[2] - 1.0).abs() < 1e-6);

    // get_strategy must return the normalized average strategy (0.2, 0.8),
    // NOT the regret matching (0.75, 0.25).
    let avg_strat = table.get_strategy(key, legal_mask);
    assert!((avg_strat[0] - 0.2).abs() < 1e-6);
    assert_eq!(avg_strat[1], 0.0);
    assert!((avg_strat[2] - 0.8).abs() < 1e-6);
    assert!((avg_strat[0] + avg_strat[2] - 1.0).abs() < 1e-6);

    // 4. Test with a legal mask where legal actions have no strategy_sum.
    // Suppose legal_mask is 0b010 (only action 1 is legal), but action 1 has no strategy_sum.
    let legal_mask_1 = 0b010;
    let mut out_mask1 = [0.0f32; 8];
    assert!(!slot.compute_average_strategy(legal_mask_1, &mut out_mask1));
    let strat_mask1 = table.get_strategy(key, legal_mask_1);
    assert!((strat_mask1[1] - 1.0).abs() < 1e-6);
}

