from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="L",
        benchmark_id="L4",
        timesteps=35,
        learning_rate=1e-3,
        batch_size=64,
        epochs=280,
        early_stop_patience=70,
        weight_bits=4,
        signed=True,
    )
