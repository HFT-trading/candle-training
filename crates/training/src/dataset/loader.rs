use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::context::Context;

/// Load every JSONL row into memory. The file is small (~3.9k rows), so a plain
/// `Vec` is fine.
pub fn load_contexts(path: impl AsRef<Path>) -> Result<Vec<Context>, DatasetError> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| DatasetError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut contexts = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| DatasetError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let context = serde_json::from_str::<Context>(&line).map_err(|source| DatasetError::Json {
            line: index + 1,
            source,
        })?;
        contexts.push(context);
    }
    if contexts.is_empty() {
        return Err(DatasetError::Empty);
    }
    Ok(contexts)
}

#[derive(Debug)]
pub enum DatasetError {
    Io {
        path: String,
        source: std::io::Error,
    },
    Json {
        line: usize,
        source: serde_json::Error,
    },
    Empty,
}

impl Display for DatasetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(formatter, "cannot read {path}: {source}"),
            Self::Json { line, source } => write!(formatter, "invalid JSON at line {line}: {source}"),
            Self::Empty => write!(formatter, "dataset is empty"),
        }
    }
}

impl Error for DatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Empty => None,
        }
    }
}
