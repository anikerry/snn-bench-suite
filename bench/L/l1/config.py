from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="L",
        benchmark_id="L1",
        timesteps=15,
        learning_rate=1e-3,
        batch_size=64,
        epochs=200,
        early_stop_patience=40,
        weight_bits=4,
        signed=True,
    )
