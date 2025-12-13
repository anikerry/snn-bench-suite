# snn-bench-suite 🧠⚡
A **hardware-agnostic Spiking Neural Network (SNN) benchmark suite** designed to stress the kinds of constraints
you run into on real neuromorphic / edge accelerators: **limited neurons**, **limited synapses**, **temporal budgets (latency)**,
**depth/topology**, and **sparsity/energy**.

This repo is intentionally *not* tied to a single chip. It produces a **canonical JSON export** that downstream tools
(mapping, scheduling, placement, memory layout) can consume for *any* target with similar limitations.

## What you get
- ✅ **23 benchmarks** across 6 families (plus XOR)
- ✅ self-contained synthetic datasets (no downloads)
- ✅ per-benchmark `train.py` entrypoints
- ✅ batch runner `run_all.py` driven by `registry.json`
- ✅ canonical model export to `artifacts/<BENCH_ID>/canonical_model.json`
- ✅ docs + example notebooks

## Benchmark families (23 models)
| Family | What it stresses | Benchmarks |
|---|---|---|
| XOR | sanity check | `xor` |
| N (Neuron scaling) | neuron count / width scaling | `N1`, `N2`, `N3`, `N4` |
| S (Synapse scaling) | synapse load / fan-out | `S1`, `S2`, `S3`, `S4` |
| P (Path/Topology) | depth / chains / bottlenecks | `P1`, `P2`, `P3`, `P4` |
| L (Latency/Temporal) | sequence length / temporal coding | `L1`, `L2`, `L3`, `L4` |
| E (Energy/Sparsity) | sparse weights + low activity patterns | `E1`, `E2`, `E3` |
| A (Application-style) | toy “vision/sensor/event” workloads | `A1`, `A2`, `A3` |

## Quickstart
### 1) Create environment & install deps
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run one benchmark
```bash
python bench/S/s2/train.py
# artifacts/S2/canonical_model.json will be written
```

### 3) Run multiple benchmarks (batch runner)
```bash
python run_all.py --bench xor N1 S1 P1 L1 E1 A1
```

## Outputs
All runs write to:
- `artifacts/<BENCH_ID>/canonical_model.json`

The canonical export is **hardware-neutral**: it focuses on layers, weights, shapes, precision hints, and timing settings.
See `examples/Using_Canonical_Export.ipynb`.

## Repo layout
```
bench/                 # benchmark families + per-benchmark packages
common/                # trainer + canonical export utilities
docs/                  # philosophy + hardware profiles + contributing
examples/              # notebooks demonstrating usage
registry.json          # single source of truth for benchmark IDs
run_all.py             # batch runner driven by registry.json
```

## Notes on “hardware-agnostic”
This repo deliberately avoids chip-specific terminology. If you have a target with:
- neuron capacity per core/tile
- synapse capacity per crossbar/bank
- limited bit-width weights
- fixed timestep budgets / latencies

…then these benchmarks are meant to be a clean, reusable input to *your* mapper/simulator.

## License
MIT (see `LICENSE`).
