#include "cuda/cuda_memory.hpp"
#include "structures/Neuron.hpp"
#include "structures/Synapse.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// ============================================================================
// Allocation Functions
// ============================================================================

void allocateNeuronDataGPU(NeuronDataGPU& data, int numNeurons) {
    data.numNeurons = numNeurons;
    size_t floatBytes = numNeurons * sizeof(float);
    size_t uint8Bytes = numNeurons * sizeof(uint8_t);

    // State variables
    CUDA_CHECK(cudaMalloc(&data.v, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.u, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.postTrace, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.firingRateAvg, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.sensitivity, floatBytes));

    // Izhikevich parameters
    CUDA_CHECK(cudaMalloc(&data.a, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.b, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.c, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.d, floatBytes));

    // Flags
    CUDA_CHECK(cudaMalloc(&data.didSpike, uint8Bytes));
    CUDA_CHECK(cudaMalloc(&data.isExcitatory, uint8Bytes));
    CUDA_CHECK(cudaMalloc(&data.layer, uint8Bytes));
}

void allocateSynapseDataGPU(SynapseDataGPU& data, int numSynapses, int maxDelayLength) {
    data.numSynapses = numSynapses;
    data.maxDelayLength = maxDelayLength;

    size_t floatBytes = numSynapses * sizeof(float);
    size_t intBytes = numSynapses * sizeof(int);
    size_t uint8Bytes = numSynapses * sizeof(uint8_t);
    size_t delayBufferBytes = numSynapses * maxDelayLength * sizeof(uint8_t);

    // Connectivity
    CUDA_CHECK(cudaMalloc(&data.targetNeuronIndex, intBytes));
    CUDA_CHECK(cudaMalloc(&data.sourceNeuronIndex, intBytes));

    // Weights
    CUDA_CHECK(cudaMalloc(&data.weight, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.maxWeight, floatBytes));

    // Traces
    CUDA_CHECK(cudaMalloc(&data.preTrace, floatBytes));
    CUDA_CHECK(cudaMalloc(&data.visualConductance, floatBytes));

    // Type
    CUDA_CHECK(cudaMalloc(&data.type, uint8Bytes));

    // Delay buffers
    CUDA_CHECK(cudaMalloc(&data.delayBuffer, delayBufferBytes));
    CUDA_CHECK(cudaMalloc(&data.delayLength, intBytes));
    CUDA_CHECK(cudaMalloc(&data.bufferCursor, intBytes));

    // Flags
    CUDA_CHECK(cudaMalloc(&data.incomingSpike, uint8Bytes));
    CUDA_CHECK(cudaMalloc(&data.justArrived, uint8Bytes));
}

void allocateSynapseConnectivityGPU(SynapseConnectivityGPU& connectivity, int numNeurons, int totalSynapses) {
    connectivity.totalSynapses = totalSynapses;

    CUDA_CHECK(cudaMalloc(&connectivity.synapseOffsets, (numNeurons + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&connectivity.synapseCount, numNeurons * sizeof(int)));
}

void allocateConductanceBuffersGPU(ConductanceBuffersGPU& buffers, int numNeurons) {
    buffers.numNeurons = numNeurons;
    size_t floatBytes = numNeurons * sizeof(float);

    CUDA_CHECK(cudaMalloc(&buffers.g_exc, floatBytes));
    CUDA_CHECK(cudaMalloc(&buffers.g_inh, floatBytes));
}

// ============================================================================
// Host -> Device Transfer Functions
// ============================================================================

void copyNeuronDataToGPU(NeuronDataGPU& gpuData, const std::vector<Neuron>& cpuNeurons) {
    int n = cpuNeurons.size();

    // Temporary host buffers in SoA layout
    std::vector<float> v(n), u(n), postTrace(n), firingRateAvg(n), sensitivity(n);
    std::vector<float> a(n), b(n), c(n), d(n);
    std::vector<uint8_t> didSpike(n), isExcitatory(n), layer(n);

    // Convert AoS to SoA
    for (int i = 0; i < n; i++) {
        const Neuron& neuron = cpuNeurons[i];
        v[i] = (float)neuron.v;
        u[i] = (float)neuron.u;
        postTrace[i] = (float)neuron.postTrace;
        firingRateAvg[i] = (float)neuron.firingRateAvg;
        sensitivity[i] = (float)neuron.sensitivity;

        a[i] = (float)neuron.a;
        b[i] = (float)neuron.b;
        c[i] = (float)neuron.c;
        d[i] = (float)neuron.d;

        didSpike[i] = neuron.didSpikeStep ? 1 : 0;
        isExcitatory[i] = neuron.isExcitatory ? 1 : 0;
        layer[i] = (uint8_t)neuron.layer;
    }

    // Transfer to GPU
    size_t floatBytes = n * sizeof(float);
    size_t uint8Bytes = n * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpy(gpuData.v, v.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.u, u.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.postTrace, postTrace.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.firingRateAvg, firingRateAvg.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.sensitivity, sensitivity.data(), floatBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpuData.a, a.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.b, b.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.c, c.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.d, d.data(), floatBytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpuData.didSpike, didSpike.data(), uint8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.isExcitatory, isExcitatory.data(), uint8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.layer, layer.data(), uint8Bytes, cudaMemcpyHostToDevice));
}

void copySynapseDataToGPU(SynapseDataGPU& gpuData, const std::vector<Neuron>& cpuNeurons) {
    // Count total synapses
    int totalSynapses = 0;
    for (const auto& neuron : cpuNeurons) {
        totalSynapses += neuron.axon.size();
    }

    if (totalSynapses == 0) return;

    // Temporary host buffers in SoA layout
    std::vector<int> targetNeuronIndex(totalSynapses);
    std::vector<int> sourceNeuronIndex(totalSynapses);
    std::vector<float> weight(totalSynapses), maxWeight(totalSynapses);
    std::vector<float> preTrace(totalSynapses), visualConductance(totalSynapses);
    std::vector<uint8_t> type(totalSynapses);
    std::vector<int> delayLength(totalSynapses), bufferCursor(totalSynapses);
    std::vector<uint8_t> incomingSpike(totalSynapses), justArrived(totalSynapses);
    std::vector<uint8_t> delayBuffer(totalSynapses * gpuData.maxDelayLength, 0);

    // Flatten synapse data
    int synapseIdx = 0;
    int neuronIdx = 0;
    for (const auto& neuron : cpuNeurons) {
        for (const auto& synapse : neuron.axon) {
            targetNeuronIndex[synapseIdx] = synapse.targetNeuronIndex;
            sourceNeuronIndex[synapseIdx] = neuronIdx; // Store source index
            weight[synapseIdx] = (float)synapse.weight;
            maxWeight[synapseIdx] = (float)synapse.maxWeight;
            preTrace[synapseIdx] = (float)synapse.preTrace;
            visualConductance[synapseIdx] = (float)synapse.visualConductance;
            type[synapseIdx] = (synapse.type == GLUTAMATE) ? 0 : 1;
            delayLength[synapseIdx] = synapse.delayBuffer.size();
            bufferCursor[synapseIdx] = synapse.bufferCursor;
            incomingSpike[synapseIdx] = synapse.incomingSpikePending ? 1 : 0;
            justArrived[synapseIdx] = synapse.justArrived ? 1 : 0;

            // Copy delay buffer
            int offset = synapseIdx * gpuData.maxDelayLength;
            for (size_t j = 0; j < synapse.delayBuffer.size(); j++) {
                delayBuffer[offset + j] = synapse.delayBuffer[j];
            }

            synapseIdx++;
        }
        neuronIdx++;
    }

    // Transfer to GPU
    size_t floatBytes = totalSynapses * sizeof(float);
    size_t intBytes = totalSynapses * sizeof(int);
    size_t uint8Bytes = totalSynapses * sizeof(uint8_t);
    size_t delayBufferBytes = totalSynapses * gpuData.maxDelayLength * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpy(gpuData.targetNeuronIndex, targetNeuronIndex.data(), intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.sourceNeuronIndex, sourceNeuronIndex.data(), intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.weight, weight.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.weight, weight.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.maxWeight, maxWeight.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.preTrace, preTrace.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.visualConductance, visualConductance.data(), floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.type, type.data(), uint8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.delayLength, delayLength.data(), intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.bufferCursor, bufferCursor.data(), intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.incomingSpike, incomingSpike.data(), uint8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.justArrived, justArrived.data(), uint8Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuData.delayBuffer, delayBuffer.data(), delayBufferBytes, cudaMemcpyHostToDevice));
}

void buildSynapseConnectivityGPU(SynapseConnectivityGPU& connectivity, const std::vector<Neuron>& cpuNeurons) {
    int numNeurons = cpuNeurons.size();

    std::vector<int> synapseOffsets(numNeurons + 1);
    std::vector<int> synapseCount(numNeurons);

    // Build CSR offsets
    int currentOffset = 0;
    for (int i = 0; i < numNeurons; i++) {
        synapseOffsets[i] = currentOffset;
        synapseCount[i] = cpuNeurons[i].axon.size();
        currentOffset += cpuNeurons[i].axon.size();
    }
    synapseOffsets[numNeurons] = currentOffset;

    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(connectivity.synapseOffsets, synapseOffsets.data(),
                         (numNeurons + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(connectivity.synapseCount, synapseCount.data(),
                         numNeurons * sizeof(int), cudaMemcpyHostToDevice));
}

// ============================================================================
// Device -> Host Transfer Functions
// ============================================================================

void copyNeuronDataFromGPU(std::vector<Neuron>& cpuNeurons, const NeuronDataGPU& gpuData) {
    int n = gpuData.numNeurons;

    // Temporary host buffers
    std::vector<float> v(n), u(n), postTrace(n), firingRateAvg(n), sensitivity(n);
    std::vector<uint8_t> didSpike(n);

    // Transfer from GPU
    size_t floatBytes = n * sizeof(float);
    size_t uint8Bytes = n * sizeof(uint8_t);

    CUDA_CHECK(cudaMemcpy(v.data(), gpuData.v, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u.data(), gpuData.u, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(postTrace.data(), gpuData.postTrace, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(firingRateAvg.data(), gpuData.firingRateAvg, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sensitivity.data(), gpuData.sensitivity, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(didSpike.data(), gpuData.didSpike, uint8Bytes, cudaMemcpyDeviceToHost));

    // Convert SoA back to AoS
    for (int i = 0; i < n; i++) {
        cpuNeurons[i].v = v[i];
        cpuNeurons[i].u = u[i];
        cpuNeurons[i].postTrace = postTrace[i];
        cpuNeurons[i].firingRateAvg = firingRateAvg[i];
        cpuNeurons[i].sensitivity = sensitivity[i];
        cpuNeurons[i].didSpikeStep = (didSpike[i] != 0);
    }
}

void copySynapseDataFromGPU(std::vector<float>& visualConductance, const SynapseDataGPU& gpuData) {
    int totalSynapses = gpuData.numSynapses;
    if (totalSynapses == 0) return;

    // Resize if necessary (should be pre-sized but safe to check)
    if (visualConductance.size() != totalSynapses) {
        visualConductance.resize(totalSynapses);
    }

    // Direct copy from GPU to flat host buffer - NO SCATTER LOOP
    CUDA_CHECK(cudaMemcpy(visualConductance.data(), gpuData.visualConductance,
                         totalSynapses * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Cleanup Functions
// ============================================================================

void freeNeuronDataGPU(NeuronDataGPU& data) {
    CUDA_CHECK(cudaFree(data.v));
    CUDA_CHECK(cudaFree(data.u));
    CUDA_CHECK(cudaFree(data.postTrace));
    CUDA_CHECK(cudaFree(data.firingRateAvg));
    CUDA_CHECK(cudaFree(data.sensitivity));
    CUDA_CHECK(cudaFree(data.a));
    CUDA_CHECK(cudaFree(data.b));
    CUDA_CHECK(cudaFree(data.c));
    CUDA_CHECK(cudaFree(data.d));
    CUDA_CHECK(cudaFree(data.didSpike));
    CUDA_CHECK(cudaFree(data.isExcitatory));
    CUDA_CHECK(cudaFree(data.layer));
}

void freeSynapseDataGPU(SynapseDataGPU& data) {
    CUDA_CHECK(cudaFree(data.targetNeuronIndex));
    CUDA_CHECK(cudaFree(data.sourceNeuronIndex));
    CUDA_CHECK(cudaFree(data.weight));
    CUDA_CHECK(cudaFree(data.maxWeight));
    CUDA_CHECK(cudaFree(data.preTrace));
    CUDA_CHECK(cudaFree(data.visualConductance));
    CUDA_CHECK(cudaFree(data.type));
    CUDA_CHECK(cudaFree(data.delayBuffer));
    CUDA_CHECK(cudaFree(data.delayLength));
    CUDA_CHECK(cudaFree(data.bufferCursor));
    CUDA_CHECK(cudaFree(data.incomingSpike));
    CUDA_CHECK(cudaFree(data.justArrived));
}

void freeSynapseConnectivityGPU(SynapseConnectivityGPU& connectivity) {
    CUDA_CHECK(cudaFree(connectivity.synapseOffsets));
    CUDA_CHECK(cudaFree(connectivity.synapseCount));
}

void freeConductanceBuffersGPU(ConductanceBuffersGPU& buffers) {
    CUDA_CHECK(cudaFree(buffers.g_exc));
    CUDA_CHECK(cudaFree(buffers.g_inh));
}
