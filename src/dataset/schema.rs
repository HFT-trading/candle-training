use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct FeatureSchema {
    pub version: u32,
    pub categorical_vocab: HashMap<String, HashMap<String, u32>>,
    pub model_input: ModelInputSchema,
    pub target_groups: TargetGroups,
}

#[derive(Debug, Deserialize)]
pub struct ModelInputSchema {
    pub categorical_features: Vec<String>,
    pub categorical_unknown_id: u32,
    pub numeric_features: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct TargetGroups {
    #[serde(default)]
    pub categorical: Vec<String>,
    pub boolean: Vec<String>,
    pub numeric: Vec<String>,
}
