//! Shared inference contract for the market-structure model.
//!
//! Both the training crate (`bot-training`) and the external serving service
//! depend on this crate, so the feature/vocab/model definitions live here once.

pub mod config;
pub mod model;
pub mod normalizer;
pub mod report;
pub mod sequence;
pub mod serve;
pub mod tensors;
pub mod vocab;
