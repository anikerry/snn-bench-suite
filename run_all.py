import json
import importlib

with open("registry.json") as f:
    reg = json.load(f)

def run(bench):
    print(f"Running benchmark: {bench}")
    info = reg[bench]

    gen = importlib.import_module(f"{info['path']}.generator")
    model_mod = importlib.import_module(f"{info['path']}.model")
    cfg_mod = importlib.import_module(f"{info['path']}.config")

    X, y = gen.generate()
    cfg = cfg_mod.Config()
    model = model_mod.Model(cfg)

    from common.train_manager import Trainer
    t = Trainer(model, cfg)
    t.train(X, y)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
