# Design Philosophy

snn-bench-suite is built around the idea that neuromorphic benchmarks should
reflect *resource dimensions* rather than specific devices. Each benchmark
family isolates one or more axes such as neuron count, synapse budget,
connectivity pattern, temporal depth, or sparsity.

This allows researchers and engineers to:

- compare different hardware platforms on equal footing,
- probe failure modes of compilers and mappers,
- study the impact of quantization and pruning,
- teach SNN concepts with concrete, runnable examples.
