//! PyO3 FFI bindings for Python training and zero-copy strategy lookup.

use pyo3::prelude::*;
use std::sync::Arc;

use crate::card::compare_cards as rust_compare_cards;
use crate::rules::{
    calculate_envido as rust_calculate_envido,
    calculate_falta_envido_points as rust_calculate_falta,
    resolve_hand as rust_resolve_hand,
};
use crate::state::BitboardState;
use crate::table::SharedPolicyTable;

#[pyclass(name = "BitboardState")]
#[derive(Clone, Copy)]
pub struct PyBitboardState(pub BitboardState);

#[pymethods]
impl PyBitboardState {
    #[new]
    #[pyo3(signature = (hands, mano=0))]
    pub fn new(hands: Vec<Vec<u8>>, mano: u8) -> PyResult<Self> {
        if hands.len() != 2 || hands[0].len() != 3 || hands[1].len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "hands must be a 2x3 nested list of card IDs",
            ));
        }
        let h = [
            [hands[0][0], hands[0][1], hands[0][2]],
            [hands[1][0], hands[1][1], hands[1][2]],
        ];
        Ok(PyBitboardState(BitboardState::new(h, mano)))
    }

    #[staticmethod]
    #[pyo3(signature = (
        hands,
        trick_cards,
        trick_results,
        trick_leader,
        current_trick,
        active_player,
        mano,
        score_p0,
        score_p1,
        max_score,
        envido_chain,
        pending_envido_from,
        truco_level,
        truco_caller,
        pending_truco_from,
        flags,
        winner,
        points_won
    ))]
    pub fn from_components(
        hands: Vec<Vec<u8>>,
        trick_cards: Vec<u8>,
        trick_results: Vec<i8>,
        trick_leader: u8,
        current_trick: u8,
        active_player: u8,
        mano: u8,
        score_p0: u8,
        score_p1: u8,
        max_score: u8,
        envido_chain: Vec<u8>,
        pending_envido_from: u8,
        truco_level: u8,
        truco_caller: u8,
        pending_truco_from: u8,
        flags: u8,
        winner: u8,
        points_won: u8,
    ) -> Self {
        let mut h = [[255u8; 3]; 2];
        for p in 0..2 {
            if p < hands.len() {
                for i in 0..3 {
                    if i < hands[p].len() {
                        h[p][i] = hands[p][i];
                    }
                }
            }
        }
        let mut tc = [255u8; 2];
        for i in 0..2 {
            if i < trick_cards.len() {
                tc[i] = trick_cards[i];
            }
        }
        let mut tr = [127i8; 3];
        for i in 0..3 {
            if i < trick_results.len() {
                tr[i] = trick_results[i];
            }
        }
        let mut ec = [255u8; 6];
        let elen = envido_chain.len().min(6) as u8;
        for i in 0..(elen as usize) {
            ec[i] = envido_chain[i];
        }

        PyBitboardState(BitboardState {
            hands: h,
            trick_cards: tc,
            trick_results: tr,
            trick_leader,
            current_trick,
            active_player,
            mano,
            score_p0,
            score_p1,
            max_score,
            envido_chain: ec,
            envido_chain_len: elen,
            pending_envido_from,
            truco_level,
            truco_caller,
            pending_truco_from,
            flags,
            winner,
            points_won,
        })
    }

    #[getter]
    pub fn active_player(&self) -> u8 {
        self.0.active_player()
    }

    #[getter]
    pub fn mano(&self) -> u8 {
        self.0.mano()
    }

    #[getter]
    pub fn trick_leader(&self) -> u8 {
        self.0.trick_leader()
    }

    #[getter]
    pub fn current_trick(&self) -> u8 {
        self.0.current_trick()
    }

    #[getter]
    pub fn truco_level(&self) -> u8 {
        self.0.truco_level()
    }

    #[getter]
    pub fn envido_resolved(&self) -> bool {
        self.0.envido_resolved()
    }

    #[getter]
    pub fn is_done(&self) -> bool {
        self.0.is_done()
    }

    #[getter]
    pub fn winner(&self) -> Option<u8> {
        self.0.winner()
    }

    #[getter]
    pub fn points_won(&self) -> u8 {
        self.0.points_won()
    }

    #[getter]
    pub fn score(&self) -> (u8, u8) {
        self.0.score()
    }

    pub fn step(&self, action: u8) -> Self {
        PyBitboardState(self.0.step(action))
    }

    pub fn legal_actions_mask(&self) -> u32 {
        self.0.legal_actions_mask()
    }

    pub fn canonical_infoset_key(&self, player: u8) -> u64 {
        self.0.canonical_infoset_key(player)
    }
}

#[pyclass(name = "SharedPolicyTable")]
pub struct PySharedPolicyTable {
    pub inner: Arc<SharedPolicyTable>,
}

#[pymethods]
impl PySharedPolicyTable {
    #[new]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(SharedPolicyTable::new(capacity)),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn count_occupied(&self) -> usize {
        self.inner.count_occupied()
    }

    pub fn get_strategy(&self, key: u64, legal_mask: u32) -> [f32; 8] {
        self.inner.get_strategy(key, legal_mask)
    }

    pub fn save_to_file(&self, path: &str) -> PyResult<usize> {
        self.inner.save_to_file(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to save policy: {e}"))
        })
    }

    pub fn load_from_file(&self, path: &str) -> PyResult<usize> {
        self.inner.load_from_file(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load policy: {e}"))
        })
    }
}

#[pyfunction]
pub fn compare_cards(c1: u8, c2: u8) -> i8 {
    rust_compare_cards(c1, c2)
}

#[pyfunction]
pub fn calculate_envido(cards: Vec<u8>) -> u8 {
    rust_calculate_envido(&cards)
}

#[pyfunction]
#[pyo3(signature = (trick_results, mano=0))]
pub fn resolve_hand(trick_results: Vec<i8>, mano: u8) -> Option<u8> {
    rust_resolve_hand(&trick_results, mano)
}

#[pyfunction]
#[pyo3(signature = (score_p0, score_p1, max_score=30))]
pub fn calculate_falta_envido_points(score_p0: u8, score_p1: u8, max_score: u8) -> u8 {
    rust_calculate_falta(score_p0, score_p1, max_score)
}

#[pyfunction]
#[pyo3(signature = (table, iterations, threads=4, cfr_plus=true))]
pub fn train_parallel(
    py: Python<'_>,
    table: &PySharedPolicyTable,
    iterations: usize,
    threads: usize,
    cfr_plus: bool,
) {
    let inner = Arc::clone(&table.inner);
    py.allow_threads(move || {
        crate::mccfr::train_parallel(inner, iterations, threads, cfr_plus);
    });
}

#[pyfunction]
pub fn get_policy_distribution(
    bstate: &PyBitboardState,
    table: &PySharedPolicyTable,
) -> Vec<(u8, f32)> {
    let mask = bstate.0.legal_actions_mask();
    if mask == 0 {
        return vec![];
    }
    let mut actions = [0u8; 8];
    let mut num_actions = 0;
    for a in 0..26u8 {
        if (mask & (1 << a)) != 0 && num_actions < 8 {
            actions[num_actions] = a;
            num_actions += 1;
        }
    }
    let key = bstate.0.canonical_infoset_key(bstate.0.active_player());
    let legal_submask = (1u32 << num_actions) - 1;
    let strat = table.inner.get_strategy(key, legal_submask);
    let mut out = Vec::with_capacity(num_actions);
    for i in 0..num_actions {
        out.push((actions[i], strat[i]));
    }
    out
}

#[pyfunction]
#[pyo3(signature = (table, iterations, threads=4, cfr_plus=true))]
pub fn train_cfr_parallel(
    py: Python<'_>,
    table: &PySharedPolicyTable,
    iterations: usize,
    threads: usize,
    cfr_plus: bool,
) {
    let inner = Arc::clone(&table.inner);
    py.allow_threads(move || {
        crate::cfr::train_cfr_parallel(inner, iterations, threads, cfr_plus);
    });
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBitboardState>()?;
    m.add_class::<PySharedPolicyTable>()?;
    m.add_function(wrap_pyfunction!(compare_cards, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_envido, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_hand, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_falta_envido_points, m)?)?;
    m.add_function(wrap_pyfunction!(train_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(train_cfr_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(get_policy_distribution, m)?)?;
    Ok(())
}
