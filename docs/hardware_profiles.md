# Hardware Profiles

While snn-bench-suite is hardware-agnostic, users are encouraged to define
their own hardware profiles describing neuron limits, synapse limits, precision
constraints, and architectural features.

Example profile fields:

- max_neurons
- max_synapses
- weight_bits
- neuron_state_bits
- max_fan_in
- max_fan_out
- parallel_groups
- max_timesteps

Toolchains can then map canonical benchmark exports to concrete hardware
configurations.
