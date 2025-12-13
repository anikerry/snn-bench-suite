from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="P",
        benchmark_id="P3",
        timesteps=20,
        learning_rate=1e-3,
        batch_size=128,
        epochs=260,
        early_stop_patience=60,
        weight_bits=4,
        signed=True,
    )
