snn-bench-suite 🧠⚡

A hardware-agnostic Spiking Neural Network Benchmark Collection

Neuromorphic hardware — digital, mixed-signal, or analog — always shares common resource constraints:

limited neuron count

synapse budget constraints

restricted precision (often 4–8 bits)

limited fan-in / fan-out

structured parallel groups

temporal coding assumptions

strict energy efficiency

sparse connectivity

snn-bench-suite is a standardized benchmark library designed around these universal constraints — not any single device.

It is compatible with THOR, Loihi 1/2, DYNAP-CNN/SE, SpiNNaker, Tianjic, TrueNorth, simulation backends, and custom neuromorphic accelerators.

🚀 Benchmark Families
Series	Meaning	Tests Which Limitation?
XOR	Minimal sanity checks	end-to-end correctness
N-series	Neuron scaling	capacity limits
S-series	Synapse scaling	dense vs sparse connectivity
P-series	Path/topology complexity	parallelism & routing
L-series	Latency & temporal coding	inference cycles & timing
E-series	Energy/sparsity	spike rate + pruning behaviours
A-series	Real application tasks	MNIST, SHD, tiny sensor datasets

Every benchmark has:

a canonical config

a model definition

a data generator

a training script

a hardware-neutral canonical JSON export

📦 Installation
git clone https://github.com/<yourname>/snn-bench-suite.git
cd snn-bench-suite
pip install -r requirements.txt

🧪 Running a Benchmark
python run_all.py --bench xor


Or train manually:

python bench/xor/train.py

📤 Canonical JSON Export

All trained models export to a device-agnostic JSON format:

{
  "neurons": [...],
  "synapses": [...],
  "precision": { "weight_bits": 4 },
  "metadata": { "task": "xor", "series": "XOR" }
}


Any mapper can ingest this.

🔁 Use Cases

hardware mapping algorithms

quantization/low-precision studies

spike simulators

educational teaching tools

model compression experiments

hardware comparison (Loihi vs TrueNorth vs DYNAP vs THOR)

reproducible neuromorphic research

🤝 Contributing

New benchmarks welcome! See docs/contributing.md.
