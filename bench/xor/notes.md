# XOR Benchmark

The XOR benchmark is a minimal, end-to-end sanity test for SNN toolchains.

- Task: binary XOR on two static inputs
- Encoding: rate coding over a fixed temporal window
- Architecture: 2-4-1 fully connected SNN with Leaky-Integrate-and-Fire neurons
- Use-cases:
  - pipeline smoke test
  - regression or classification depending on downstream interpretation
  - debugging canonical export and hardware mapping
