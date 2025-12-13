from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="L",
        benchmark_id="L3",
        timesteps=30,
        learning_rate=1e-3,
        batch_size=64,
        epochs=260,
        early_stop_patience=60,
        weight_bits=4,
        signed=True,
    )
