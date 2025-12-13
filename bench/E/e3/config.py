from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="E",
        benchmark_id="E3",
        timesteps=20,
        learning_rate=1e-3,
        batch_size=64,
        epochs=240,
        early_stop_patience=60,
        weight_bits=4,
        signed=True,
    )
