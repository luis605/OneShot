#ifndef GLOBALS_H
#define GLOBALS_H

#include "structures.h"
#include <vector>
#include <list>

extern std::vector<Neuron> AllNeurons;
extern std::vector<Synapse> AllSynapses;

constexpr int SPIKE_DELIVERY_QUEUE_SIZE = 256;
extern std::vector<std::list<SpikeEvent>> SpikeDeliveryQueue;

#endif
