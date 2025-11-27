#include "cuda/cuda_types.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

// ============================================================================
// Constant Memory for Simulation Parameters
// ============================================================================

__constant__ SimulationConstants d_constants;

// ============================================================================
// Kernel 1: Update Synapses (Delay Buffers & Spike Arrival Detection)
// ============================================================================

__global__ void updateSynapsesKernel(SynapseDataGPU synapses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.numSynapses) return;

    // Reset arrival flag
    synapses.justArrived[idx] = 0;

    // Get delay buffer info
    int delayLen = synapses.delayLength[idx];
    int cursor = synapses.bufferCursor[idx];

    // Check for spike arrival
    uint8_t signal = synapses.delayBuffer[idx * synapses.maxDelayLength + cursor];

    if (signal == 1) {
        synapses.justArrived[idx] = 1;
    }

    // Decay pre-synaptic trace
    float decay_factor = 1.0f - (d_constants.dt / d_constants.tau_trace_pre);
    synapses.preTrace[idx] *= decay_factor;

    // Advance circular buffer cursor
    synapses.bufferCursor[idx] = (cursor + 1) % delayLen;
}

// ============================================================================
// Kernel 2: Clear Conductance Buffers
// ============================================================================

__global__ void clearConductancesKernel(ConductanceBuffersGPU buffers) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx >= buffers.numNeurons) return;

    buffers.g_exc[neuronIdx] = 0.0f;
    buffers.g_inh[neuronIdx] = 0.0f;
}

// ============================================================================
// Kernel 3: Accumulate Conductances (Synapse -> Neuron)
// OPTIMIZED: One thread per synapse, atomic add to target neuron
// ============================================================================

__global__ void accumulateConductancesKernel(
    SynapseDataGPU synapses,
    ConductanceBuffersGPU buffers
) {
    int synIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (synIdx >= synapses.numSynapses) return;

    // Only process synapses with arriving spikes
    if (!synapses.justArrived[synIdx]) return;

    int targetNeuronIdx = synapses.targetNeuronIndex[synIdx];

    // Convert uint16_t to __half, then to FP32 for computation
    __half* weightPtr = reinterpret_cast<__half*>(&synapses.weight[synIdx]);
    float weight = __half2float(*weightPtr);

    // Atomic add to target neuron's conductance buffer
    if (synapses.type[synIdx] == 0) {  // GLUTAMATE
        atomicAdd(&buffers.g_exc[targetNeuronIdx], weight);
    } else {  // GABA
        atomicAdd(&buffers.g_inh[targetNeuronIdx], weight);
    }
}

// ============================================================================
// Kernel: Initialize RNG
// ============================================================================

__global__ void initRNGKernel(curandState* states, int numNeurons, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNeurons) return;

    curand_init(seed, idx, 0, &states[idx]);
}

// ============================================================================
// Kernel 4: Update Neurons (RK4 Integration & Spike Detection)
// ============================================================================

__global__ void updateNeuronsKernel(
    NeuronDataGPU neurons,
    ConductanceBuffersGPU buffers,
    float panicInhibition,
    curandState* rngStates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.numNeurons) return;

    // Load neuron state
    float v = neurons.v[idx];
    float u = neurons.u[idx];
    float sensitivity = neurons.sensitivity[idx];

    // Reset spike flag
    neurons.didSpike[idx] = 0;

    // Check for NaN/Inf
    if (isnan(v) || isinf(v)) {
        neurons.v[idx] = -65.0f;
        neurons.u[idx] = 0.0f;
        return;
    }

    // Generate noise
    float noise = 0.0f;
    curandState localState = rngStates[idx];
    if (curand_uniform(&localState) < 0.001f) {
        noise = 15.0f;
    }
    rngStates[idx] = localState; // Save state back

    // Compute synaptic currents
    float g_exc = fminf(buffers.g_exc[idx], 100.0f);  // Clamp to prevent instability
    float g_inh = buffers.g_inh[idx] + panicInhibition;

    float exc_current = g_exc * (d_constants.E_exc - v) * sensitivity;
    float inh_current = g_inh * (d_constants.E_inh - v);
    float I_total = exc_current + inh_current + noise;

    // RK4 Integration
    float dt = d_constants.dt;
    float a = neurons.a[idx];
    float b = neurons.b[idx];

    // Define differential equations as lambdas (manually inlined)
    auto compute_dv = [&](float v_val, float u_val) {
        return 0.04f * v_val * v_val + 5.0f * v_val + 140.0f - u_val + I_total;
    };

    auto compute_du = [&](float v_val, float u_val) {
        return a * (b * v_val - u_val);
    };

    // RK4 stages
    float kv1 = compute_dv(v, u);
    float ku1 = compute_du(v, u);

    float kv2 = compute_dv(v + kv1 * dt * 0.5f, u + ku1 * dt * 0.5f);
    float ku2 = compute_du(v + kv1 * dt * 0.5f, u + ku1 * dt * 0.5f);

    float kv3 = compute_dv(v + kv2 * dt * 0.5f, u + ku2 * dt * 0.5f);
    float ku3 = compute_du(v + kv2 * dt * 0.5f, u + ku2 * dt * 0.5f);

    float kv4 = compute_dv(v + kv3 * dt, u + ku3 * dt);
    float ku4 = compute_du(v + kv3 * dt, u + ku3 * dt);

    v += (dt / 6.0f) * (kv1 + 2.0f * kv2 + 2.0f * kv3 + kv4);
    u += (dt / 6.0f) * (ku1 + 2.0f * ku2 + 2.0f * ku3 + ku4);

    // Clamp voltage to valid range
    v = fminf(fmaxf(v, -90.0f), 50.0f);

    // Decay post-synaptic trace
    float decay_factor = 1.0f - (dt / d_constants.tau_trace_post);
    float postTrace = neurons.postTrace[idx] * decay_factor;

    // Check for spike
    float firingRateAvg = neurons.firingRateAvg[idx];
    if (v > 30.0f) {
        v = neurons.c[idx];
        u += neurons.d[idx];
        postTrace += 1.0f;
        neurons.didSpike[idx] = 1;
        firingRateAvg = 0.95f * firingRateAvg + 0.05f * 1.0f;
    } else {
        firingRateAvg *= 0.9995f;
    }

    // Write back updated state
    neurons.v[idx] = v;
    neurons.u[idx] = u;
    neurons.postTrace[idx] = postTrace;
    neurons.firingRateAvg[idx] = firingRateAvg;
}

// ============================================================================
// Kernel 5: STDP Depression (LTD)
// ============================================================================

__global__ void stdpDepressionKernel(
    SynapseDataGPU synapses,
    NeuronDataGPU neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.numSynapses) return;

    // Only apply to excitatory synapses that just arrived
    if (!synapses.justArrived[idx]) return;

    int targetIdx = synapses.targetNeuronIndex[idx];
    float postTrace = neurons.postTrace[targetIdx];

    // LTD: if post-synaptic neuron has recent activity, weaken synapse
    if (postTrace > 0.1f) {
        // Convert uint16_t to __half, then to FP32 for computation
        __half* weightPtr = reinterpret_cast<__half*>(&synapses.weight[idx]);
        float weight = __half2float(*weightPtr);
        weight -= d_constants.learning_rate * postTrace;
        weight = fmaxf(weight, 0.0f);  // Clamp to non-negative

        // Convert back to FP16 and store as uint16_t
        __half hw = __float2half(weight);
        synapses.weight[idx] = *reinterpret_cast<uint16_t*>(&hw);
    }
}

// ============================================================================
// Kernel 6: Process Spikes (Propagate to Delay Buffers)
// ============================================================================

__global__ void processSpikeKernel(
    NeuronDataGPU neurons,
    SynapseDataGPU synapses,
    SynapseConnectivityGPU connectivity
) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx >= neurons.numNeurons) return;

    // Only process if neuron spiked this timestep
    if (!neurons.didSpike[neuronIdx]) return;

    // Get synapse range for this neuron
    int synStart = connectivity.synapseOffsets[neuronIdx];
    int synEnd = connectivity.synapseOffsets[neuronIdx + 1];

    // Propagate spike to all outgoing synapses
    for (int synIdx = synStart; synIdx < synEnd; synIdx++) {
        // Increment pre-synaptic trace
        synapses.preTrace[synIdx] += 1.0f;

        // Write spike into delay buffer at current cursor
        int cursor = synapses.bufferCursor[synIdx];
        int bufferOffset = synIdx * synapses.maxDelayLength;
        synapses.delayBuffer[bufferOffset + cursor] = 1;
    }
}

// ============================================================================
// Kernel 7: STDP Potentiation (LTP)
// ============================================================================

__global__ void stdpPotentiationKernel(
    SynapseDataGPU synapses,
    NeuronDataGPU neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.numSynapses) return;

    int targetIdx = synapses.targetNeuronIndex[idx];

    // LTP: if target neuron just spiked and synapse has pre-trace, strengthen
    if (neurons.didSpike[targetIdx] && synapses.preTrace[idx] > 0.1f) {
        // Convert uint16_t to __half, then to FP32 for computation
        __half* weightPtr = reinterpret_cast<__half*>(&synapses.weight[idx]);
        __half* maxWeightPtr = reinterpret_cast<__half*>(&synapses.maxWeight[idx]);
        float weight = __half2float(*weightPtr);
        float maxWeight = __half2float(*maxWeightPtr);

        weight += d_constants.learning_rate * synapses.preTrace[idx];
        weight = fminf(weight, maxWeight);  // Clamp to max

        // Convert back to FP16 and store as uint16_t
        __half hw = __float2half(weight);
        synapses.weight[idx] = *reinterpret_cast<uint16_t*>(&hw);
    }
}

// ============================================================================
// Kernel 8: Homeostasis (Synaptic Scaling)
// ============================================================================

__global__ void homeostasisKernel(NeuronDataGPU neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neurons.numNeurons) return;

    float firingRate = neurons.firingRateAvg[idx];
    float sensitivity = neurons.sensitivity[idx];

    // Compute error from target rate
    float error = d_constants.target_firing_rate - firingRate;

    // Adjust sensitivity
    sensitivity += error * d_constants.homeostasis_rate;
    sensitivity = fminf(fmaxf(sensitivity, 0.1f), 2.5f);  // Clamp to [0.1, 2.5]

    neurons.sensitivity[idx] = sensitivity;
}

// ============================================================================
// Kernel 9: Compute System Averages (Parallel Reduction)
// ============================================================================

__global__ void computeSystemAveragesKernel(
    NeuronDataGPU neurons,
    float* d_voltageSum,
    float* d_sensitivitySum
) {
    __shared__ float sharedVoltage[256];
    __shared__ float sharedSensitivity[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory (or zero if out of bounds)
    sharedVoltage[tid] = (idx < neurons.numNeurons) ? neurons.v[idx] : 0.0f;
    sharedSensitivity[tid] = (idx < neurons.numNeurons) ? neurons.sensitivity[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedVoltage[tid] += sharedVoltage[tid + s];
            sharedSensitivity[tid] += sharedSensitivity[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        atomicAdd(d_voltageSum, sharedVoltage[0]);
        atomicAdd(d_sensitivitySum, sharedSensitivity[0]);
    }
}

// ============================================================================
// Host Function: Upload Simulation Constants
// ============================================================================

extern "C" void uploadSimulationConstants(const SimulationConstants& constants) {
    cudaMemcpyToSymbol(d_constants, &constants, sizeof(SimulationConstants));
}

// ============================================================================
// Host Function: Kernel Launch Wrappers
// ============================================================================

extern "C" void launchUpdateSynapses(SynapseDataGPU synapses) {
    int blockSize = 256;
    int numBlocks = (synapses.numSynapses + blockSize - 1) / blockSize;
    updateSynapsesKernel<<<numBlocks, blockSize>>>(synapses);
}

extern "C" void launchClearConductances(ConductanceBuffersGPU buffers) {
    int blockSize = 256;
    int numBlocks = (buffers.numNeurons + blockSize - 1) / blockSize;
    clearConductancesKernel<<<numBlocks, blockSize>>>(buffers);
}

extern "C" void launchAccumulateConductances(
    SynapseDataGPU synapses,
    ConductanceBuffersGPU buffers
) {
    int blockSize = 256;
    int numBlocks = (synapses.numSynapses + blockSize - 1) / blockSize;
    accumulateConductancesKernel<<<numBlocks, blockSize>>>(synapses, buffers);
}

extern "C" void launchUpdateNeurons(
    NeuronDataGPU neurons,
    ConductanceBuffersGPU buffers,
    float panicInhibition,
    void* rngStates
) {
    int blockSize = 256;
    int numBlocks = (neurons.numNeurons + blockSize - 1) / blockSize;
    updateNeuronsKernel<<<numBlocks, blockSize>>>(neurons, buffers, panicInhibition, (curandState*)rngStates);
}

extern "C" void launchInitRNG(void* states, int numNeurons, unsigned long long seed) {
    int blockSize = 256;
    int numBlocks = (numNeurons + blockSize - 1) / blockSize;
    initRNGKernel<<<numBlocks, blockSize>>>((curandState*)states, numNeurons, seed);
}

extern "C" void launchStdpDepression(SynapseDataGPU synapses, NeuronDataGPU neurons) {
    int blockSize = 256;
    int numBlocks = (synapses.numSynapses + blockSize - 1) / blockSize;
    stdpDepressionKernel<<<numBlocks, blockSize>>>(synapses, neurons);
}

extern "C" void launchProcessSpikes(
    NeuronDataGPU neurons,
    SynapseDataGPU synapses,
    SynapseConnectivityGPU connectivity
) {
    int blockSize = 256;
    int numBlocks = (neurons.numNeurons + blockSize - 1) / blockSize;
    processSpikeKernel<<<numBlocks, blockSize>>>(neurons, synapses, connectivity);
}

extern "C" void launchStdpPotentiation(SynapseDataGPU synapses, NeuronDataGPU neurons) {
    int blockSize = 256;
    int numBlocks = (synapses.numSynapses + blockSize - 1) / blockSize;
    stdpPotentiationKernel<<<numBlocks, blockSize>>>(synapses, neurons);
}

extern "C" void launchHomeostasis(NeuronDataGPU neurons) {
    int blockSize = 256;
    int numBlocks = (neurons.numNeurons + blockSize - 1) / blockSize;
    homeostasisKernel<<<numBlocks, blockSize>>>(neurons);
}

extern "C" void launchComputeSystemAverages(
    NeuronDataGPU neurons,
    float* d_voltageSum,
    float* d_sensitivitySum
) {
    int blockSize = 256;
    int numBlocks = (neurons.numNeurons + blockSize - 1) / blockSize;

    // Initialize sums to zero
    float zero = 0.0f;
    cudaMemcpy(d_voltageSum, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sensitivitySum, &zero, sizeof(float), cudaMemcpyHostToDevice);

    computeSystemAveragesKernel<<<numBlocks, blockSize>>>(neurons, d_voltageSum, d_sensitivitySum);
}

// ============================================================================
// Kernel 10: Collect Active Synapses (GPU-side Compaction)
// ============================================================================

__global__ void collectActiveSynapsesKernel(
    SynapseDataGPU synapses,
    int* d_count,
    int* d_sourceIndices,
    int* d_targetIndices,
    float* d_conductances,
    int maxSynapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= synapses.numSynapses) return;

    // Compute visual conductance on-the-fly based on recent activity
    // Use justArrived flag as proxy for visualization
    float visualCond = synapses.justArrived[idx] ? 1.0f : 0.0f;

    if (visualCond > 0.2f) { // Threshold
        int pos = atomicAdd(d_count, 1);
        if (pos < maxSynapses) {
            d_sourceIndices[pos] = synapses.sourceNeuronIndex[idx];
            d_targetIndices[pos] = synapses.targetNeuronIndex[idx];
            d_conductances[pos] = visualCond;
        }
    }
}

extern "C" void launchCollectActiveSynapses(
    SynapseDataGPU synapses,
    int* d_count,
    int* d_sourceIndices,
    int* d_targetIndices,
    float* d_conductances,
    int maxSynapses
) {
    // Reset count
    cudaMemset(d_count, 0, sizeof(int));

    int blockSize = 256;
    int numBlocks = (synapses.numSynapses + blockSize - 1) / blockSize;
    collectActiveSynapsesKernel<<<numBlocks, blockSize>>>(
        synapses, d_count, d_sourceIndices, d_targetIndices, d_conductances, maxSynapses
    );
}
