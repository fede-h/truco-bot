pub mod actions;
pub mod card;
pub mod isomorphism;
pub mod mccfr;
pub mod rules;
pub mod state;
pub mod table;

#[cfg(feature = "pyo3")]
pub mod ffi;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
#[pymodule]
fn truco_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ffi::register_module(m)
}
