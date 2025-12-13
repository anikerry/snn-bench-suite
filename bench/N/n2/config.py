from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    return TrainingConfig(
        task="classification",
        series="N",
        benchmark_id="N2",
        timesteps=20,
        learning_rate=1e-3,
        batch_size=128,
        epochs=250,
        early_stop_patience=50,
        weight_bits=4,
        signed=True,
    )
