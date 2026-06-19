fn main() {
    bot_training::logging::init();
    tracing::info!("bot-training");
    tracing::info!(
        command = "cargo run --bin train",
        "training pipeline command"
    );
}
