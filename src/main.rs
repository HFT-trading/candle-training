fn main() {
    bot_training::logging::init();
    tracing::info!("bot-training");
    tracing::info!(
        command = "cargo run --bin train",
        "training pipeline command"
    );
    tracing::info!(
        command = "RUST_LOG=debug cargo run --bin inspect -- 0",
        "checkpoint inspection command"
    );
}
