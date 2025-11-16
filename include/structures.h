#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <cstdint>
#include <vector>

#define NEURON_FLAG_EXCITATORY 0b00000001
#define NEURON_MODEL_LIF 0b00000010
#define NEURON_MODEL_IZ_RS 0b00000100
#define NEURON_MODEL_IZ_IB 0b00000110
#define NEURON_FLAG_FIRED 0b00010000

struct Neuron {
    uint64_t id;
    uint32_t flags;
    float v;
    float u;
    float input_current;
    uint64_t last_spike_time;
    float position[3];
    uint64_t axon_start_index;
    uint32_t axon_count;
};

struct Synapse {
    uint64_t source_neuron_id;
    uint64_t target_neuron_id;
    float weight;
    uint16_t delay;
    float eligibility_trace;
    uint16_t prune_timer;
};

struct SpikeEvent {
    uint64_t target_neuron_id;
    float weight;
};

#endif
