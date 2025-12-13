from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="L",
        benchmark_id="L2",
        timesteps=25,
        learning_rate=1e-3,
        batch_size=64,
        epochs=220,
        early_stop_patience=50,
        weight_bits=4,
        signed=True,
    )
