from common.train_manager import TrainingConfig

def get_config() -> TrainingConfig:
    cfg = TrainingConfig(
        task="classification",   # XOR as 2-class classification
        series="XOR",
        benchmark_id="xor",
        timesteps=20,
        learning_rate=1e-3,
        batch_size=128,
        epochs=200,
        early_stop_patience=50,
        weight_bits=4,
        signed=True,
    )
    return cfg
