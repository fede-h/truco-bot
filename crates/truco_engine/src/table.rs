//! 64-byte cache-line aligned lock-free SharedPolicyTable with AtomicF32 and open addressing.

use atomic_float::AtomicF32;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const EMPTY_KEY: u64 = 0;

#[repr(C, align(64))]
pub struct InfosetSlot {
    pub key: AtomicU64,
    pub action_mask: AtomicU32,
    pub _pad: [u8; 4],
    pub regrets: [AtomicF32; 8],
    pub strategy_sum: [AtomicF32; 8],
}

impl Default for InfosetSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl InfosetSlot {
    pub fn new() -> Self {
        Self {
            key: AtomicU64::new(EMPTY_KEY),
            action_mask: AtomicU32::new(0),
            _pad: [0; 4],
            regrets: [
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
            ],
            strategy_sum: [
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
                AtomicF32::new(0.0),
            ],
        }
    }

    #[inline(always)]
    pub fn update_regret(&self, action: usize, delta: f32, cfr_plus: bool) {
        if action >= 8 {
            return;
        }
        if cfr_plus {
            // ponytail: lock-free CAS loop enforcing CFR+ non-negative regret invariant
            let mut prev = self.regrets[action].load(Ordering::Relaxed);
            loop {
                let new_val = (prev + delta).max(0.0);
                match self.regrets[action].compare_exchange_weak(
                    prev,
                    new_val,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => prev = actual,
                }
            }
        } else {
            self.regrets[action].fetch_add(delta, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn accumulate_strategy(&self, action: usize, prob: f32, weight: f32) {
        if action >= 8 {
            return;
        }
        self.strategy_sum[action].fetch_add(prob * weight, Ordering::Relaxed);
    }

    pub fn compute_regret_matching(&self, legal_mask: u32, out: &mut [f32]) {
        let mut sum_positive = 0.0f32;
        let mut count = 0;
        for a in 0..8 {
            if (legal_mask & (1 << a)) != 0 {
                let r = self.regrets[a].load(Ordering::Relaxed).max(0.0);
                out[a] = r;
                sum_positive += r;
                count += 1;
            } else {
                out[a] = 0.0;
            }
        }

        if sum_positive > 1e-6 {
            let inv = 1.0 / sum_positive;
            for a in 0..8 {
                if (legal_mask & (1 << a)) != 0 {
                    out[a] *= inv;
                }
            }
        } else if count > 0 {
            let uniform = 1.0 / (count as f32);
            for a in 0..8 {
                if (legal_mask & (1 << a)) != 0 {
                    out[a] = uniform;
                }
            }
        }
    }
}

pub struct SharedPolicyTable {
    pub slots: Box<[InfosetSlot]>,
    mask: usize,
}

impl SharedPolicyTable {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(16).next_power_of_two();
        let mut slots_vec = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots_vec.push(InfosetSlot::new());
        }
        Self {
            slots: slots_vec.into_boxed_slice(),
            mask: cap - 1,
        }
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn get_or_create(&self, key: u64, legal_mask: u32) -> &InfosetSlot {
        // ponytail: fast 64-bit integer mix for linear probing
        let mut idx = (key.wrapping_mul(0x517cc1b727220a95) >> 16) as usize & self.mask;
        for _ in 0..self.slots.len() {
            let slot = &self.slots[idx];
            let cur = slot.key.load(Ordering::Acquire);
            if cur == key {
                return slot;
            }
            if cur == EMPTY_KEY {
                match slot.key.compare_exchange_weak(
                    EMPTY_KEY,
                    key,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        slot.action_mask.store(legal_mask, Ordering::Release);
                        return slot;
                    }
                    Err(actual) => {
                        if actual == key {
                            return slot;
                        }
                    }
                }
            }
            idx = (idx + 1) & self.mask;
        }
        panic!("SharedPolicyTable capacity exhausted! Capacity: {}", self.slots.len());
    }

    pub fn count_occupied(&self) -> usize {
        self.slots.iter().filter(|s| s.key.load(Ordering::Relaxed) != EMPTY_KEY).count()
    }

    pub fn get_strategy(&self, key: u64, legal_mask: u32) -> [f32; 8] {
        let mut idx = (key.wrapping_mul(0x517cc1b727220a95) >> 16) as usize & self.mask;
        for _ in 0..self.slots.len() {
            let slot = &self.slots[idx];
            let cur = slot.key.load(Ordering::Acquire);
            if cur == key {
                let mut out = [0.0f32; 8];
                slot.compute_regret_matching(legal_mask, &mut out);
                return out;
            }
            if cur == EMPTY_KEY {
                break;
            }
            idx = (idx + 1) & self.mask;
        }
        let mut out = [0.0f32; 8];
        let count = legal_mask.count_ones();
        if count > 0 {
            let uniform = 1.0 / (count as f32);
            for a in 0..8 {
                if (legal_mask & (1 << a)) != 0 {
                    out[a] = uniform;
                }
            }
        }
        out
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<usize> {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut count = 0usize;
        for slot in self.slots.iter() {
            let key = slot.key.load(Ordering::Relaxed);
            if key != EMPTY_KEY {
                file.write_all(&key.to_le_bytes())?;
                let mask = slot.action_mask.load(Ordering::Relaxed);
                file.write_all(&mask.to_le_bytes())?;
                for r in &slot.regrets {
                    file.write_all(&r.load(Ordering::Relaxed).to_le_bytes())?;
                }
                for s in &slot.strategy_sum {
                    file.write_all(&s.load(Ordering::Relaxed).to_le_bytes())?;
                }
                count += 1;
            }
        }
        file.flush()?;
        Ok(count)
    }

    pub fn load_from_file(&self, path: &str) -> std::io::Result<usize> {
        use std::io::Read;
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut count = 0usize;
        let mut buf = [0u8; 76];
        while file.read_exact(&mut buf).is_ok() {
            let key = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let mask = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            let slot = self.get_or_create(key, mask);
            for i in 0..8 {
                let r = f32::from_le_bytes(buf[12 + i * 4..16 + i * 4].try_into().unwrap());
                slot.regrets[i].store(r, Ordering::Relaxed);
            }
            for i in 0..8 {
                let s = f32::from_le_bytes(buf[44 + i * 4..48 + i * 4].try_into().unwrap());
                slot.strategy_sum[i].store(s, Ordering::Relaxed);
            }
            count += 1;
        }
        Ok(count)
    }
}

