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
