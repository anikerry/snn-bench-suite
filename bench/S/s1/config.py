from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="S",
        benchmark_id="S1",
        timesteps=20,
        learning_rate=1e-3,
        batch_size=128,
        epochs=150,
        early_stop_patience=30,
        weight_bits=4,
        signed=True,
    )
