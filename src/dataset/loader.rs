use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::{MarketDataset, MarketSequence};

pub fn load_dataset(
    schema_path: impl AsRef<Path>,
    sequences_path: impl AsRef<Path>,
) -> Result<MarketDataset, DatasetError> {
    let schema_path = schema_path.as_ref();
    let sequences_path = sequences_path.as_ref();

    let schema_file = File::open(schema_path).map_err(|source| DatasetError::Io {
        path: schema_path.to_path_buf(),
        source,
    })?;
    let schema = serde_json::from_reader(schema_file).map_err(|source| DatasetError::Json {
        path: schema_path.to_path_buf(),
        line: None,
        source,
    })?;

    let sequences_file = File::open(sequences_path).map_err(|source| DatasetError::Io {
        path: sequences_path.to_path_buf(),
        source,
    })?;
    let mut sequences = Vec::new();
    for (line_index, line) in BufReader::new(sequences_file).lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.map_err(|source| DatasetError::Io {
            path: sequences_path.to_path_buf(),
            source,
        })?;
        let sequence =
            serde_json::from_str::<MarketSequence>(&line).map_err(|source| DatasetError::Json {
                path: sequences_path.to_path_buf(),
                line: Some(line_number),
                source,
            })?;
        validate_sequence(&sequence, line_number)?;
        sequences.push(sequence);
    }

    if sequences.is_empty() {
        return Err(DatasetError::EmptySequences);
    }

    Ok(MarketDataset { schema, sequences })
}

fn validate_sequence(sequence: &MarketSequence, line: usize) -> Result<(), DatasetError> {
    let group_lengths = [
        sequence.state_features.len(),
        sequence.range_telemetry.len(),
        sequence.cycle_context.len(),
        sequence.episode_context.len(),
    ];
    if group_lengths
        .iter()
        .any(|length| *length != sequence.seq_len)
    {
        return Err(DatasetError::InvalidSequence {
            line,
            sample_id: sequence.sample_id.clone(),
            message: format!(
                "seq_len={} but group lengths are {group_lengths:?}",
                sequence.seq_len
            ),
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum DatasetError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        line: Option<usize>,
        source: serde_json::Error,
    },
    InvalidSequence {
        line: usize,
        sample_id: String,
        message: String,
    },
    EmptySequences,
}

impl Display for DatasetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::Json { path, line, source } => match line {
                Some(line) => write!(
                    formatter,
                    "invalid JSON at {}:{line}: {source}",
                    path.display()
                ),
                None => write!(formatter, "invalid JSON in {}: {source}", path.display()),
            },
            Self::InvalidSequence {
                line,
                sample_id,
                message,
            } => write!(
                formatter,
                "invalid sequence {sample_id} at line {line}: {message}"
            ),
            Self::EmptySequences => write!(formatter, "market sequence file is empty"),
        }
    }
}

impl Error for DatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            _ => None,
        }
    }
}
