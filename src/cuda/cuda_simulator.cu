#include "cuda/cuda_simulator.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <algorithm>

CUDASimulator::CUDASimulator()
    : m_ticks(0)
    , m_numNeurons(0)
    , m_maxDelayLength(0)
    , m_systemAvgVoltage(-65.0f)
    , m_systemAvgSensitivity(1.0f)
    , m_initialized(false)
    , d_voltageSum(nullptr)
    , d_sensitivitySum(nullptr)
    , d_rngStates(nullptr)
{
}

CUDASimulator::~CUDASimulator() {
    cleanup();
}

void CUDASimulator::initialize(const std::vector<Neuron>& cpuNeurons, int numNeurons) {
    if (m_initialized) {
        fprintf(stderr, "CUDASimulator: Already initialized!\n");
        return;
    }

    m_numNeurons = numNeurons;
    printf("CUDASimulator: Initializing with %d neurons...\n", numNeurons);

    // Find maximum delay length across all synapses
    m_maxDelayLength = 1;
    int totalSynapses = 0;
    for (const auto& neuron : cpuNeurons) {
        totalSynapses += neuron.axon.size();
        for (const auto& syn : neuron.axon) {
            m_maxDelayLength = std::max(m_maxDelayLength, (int)syn.delayBuffer.size());
        }
    }

    printf("CUDASimulator: Total synapses = %d, max delay = %d\n", totalSynapses, m_maxDelayLength);

    // Allocate GPU memory
    allocateNeuronDataGPU(m_neurons, numNeurons);
    allocateSynapseDataGPU(m_synapses, totalSynapses, m_maxDelayLength);
    allocateSynapseConnectivityGPU(m_connectivity, numNeurons, totalSynapses);
    allocateConductanceBuffersGPU(m_conductances, numNeurons);

    // Allocate reduction result buffers
    CUDA_CHECK(cudaMalloc(&d_voltageSum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sensitivitySum, sizeof(float)));

    // Allocate RNG states
    CUDA_CHECK(cudaMalloc(&d_rngStates, numNeurons * sizeof(curandState)));

    // Initialize RNG
    launchInitRNG(d_rngStates, numNeurons, 1234ULL);

    // Transfer data to GPU
    copyNeuronDataToGPU(m_neurons, cpuNeurons);
    copySynapseDataToGPU(m_synapses, cpuNeurons);
    buildSynapseConnectivityGPU(m_connectivity, cpuNeurons);

    // Initialize conductance buffers to zero
    CUDA_CHECK(cudaMemset(m_conductances.g_exc, 0, numNeurons * sizeof(float)));
    CUDA_CHECK(cudaMemset(m_conductances.g_inh, 0, numNeurons * sizeof(float)));

    // Upload simulation constants to GPU
    SimulationConstants constants;
    constants.dt = (float)DT;
    constants.tau_trace_pre = (float)TAU_TRACE_PRE;
    constants.tau_trace_post = (float)TAU_TRACE_POST;
    constants.E_exc = (float)E_EXC;
    constants.E_inh = (float)E_INH;
    constants.learning_rate = 0.01f;
    constants.target_firing_rate = 0.01f;
    constants.homeostasis_rate = 0.01f;

    uploadSimulationConstants(constants);

    // Allocate compact visualization buffers
    CUDA_CHECK(cudaMalloc(&d_activeSynapseCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_activeSourceIndices, MAX_ACTIVE_SYNAPSES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_activeTargetIndices, MAX_ACTIVE_SYNAPSES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_activeConductances, MAX_ACTIVE_SYNAPSES * sizeof(float)));

    // Synchronize to ensure all initialization is complete
    CUDA_CHECK(cudaDeviceSynchronize());

    m_initialized = true;
    printf("CUDASimulator: Initialization complete!\n");
}

void CUDASimulator::step() {
    if (!m_initialized) {
        fprintf(stderr, "CUDASimulator: Not initialized!\n");
        return;
    }

    // Step 1: Update synapses (delay buffers, spike arrival detection)
    launchUpdateSynapses(m_synapses);

    // Step 2: Clear conductance buffers
    launchClearConductances(m_conductances);

    // Step 3: Accumulate conductances from arriving spikes (OPTIMIZED: atomic adds)
    launchAccumulateConductances(m_synapses, m_conductances);

    // Step 4: Compute panic inhibition based on average voltage
    float panicInhibition = 0.0f;
    if (m_systemAvgVoltage > -60.0f) {
        panicInhibition = (m_systemAvgVoltage + 60.0f) * 5.0f;
    }

    // Step 5: Update neurons (RK4 integration, spike detection)
    launchUpdateNeurons(m_neurons, m_conductances, panicInhibition, d_rngStates);

    // Step 6: Apply STDP depression (LTD)
    launchStdpDepression(m_synapses, m_neurons);

    // Step 7: Propagate new spikes to delay buffers
    launchProcessSpikes(m_neurons, m_synapses, m_connectivity);

    // Step 8: Apply STDP potentiation (LTP)
    launchStdpPotentiation(m_synapses, m_neurons);

    // Step 9: Homeostasis (every 1000 steps)
    if (m_ticks % 1000 == 0) {
        launchHomeostasis(m_neurons);
    }

    // Step 10: Compute system averages
    launchComputeSystemAverages(m_neurons, d_voltageSum, d_sensitivitySum);

    // Synchronize to ensure all kernels complete
    CUDA_CHECK(cudaDeviceSynchronize());

    float voltageSum, sensitivitySum;
    CUDA_CHECK(cudaMemcpy(&voltageSum, d_voltageSum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sensitivitySum, d_sensitivitySum, sizeof(float), cudaMemcpyDeviceToHost));

    m_systemAvgVoltage = voltageSum / m_numNeurons;
    m_systemAvgSensitivity = sensitivitySum / m_numNeurons;

    m_ticks++;
}

void CUDASimulator::injectNoise(int neuronIdx, float current) {
    // TODO: Implement external current injection
    // This would require a separate kernel or buffer for external currents
    // For now, this is a placeholder
}

void CUDASimulator::syncToHost(std::vector<Neuron>& cpuNeurons, std::vector<CorticalColumn::ActiveSynapse>& activeSynapses) {
    if (!m_initialized) {
        fprintf(stderr, "CUDASimulator: Not initialized!\n");
        return;
    }

    // Copy GPU data back to CPU
    copyNeuronDataFromGPU(cpuNeurons, m_neurons);

    // Compact active synapses on GPU
    launchCollectActiveSynapses(
        m_synapses,
        d_activeSynapseCount,
        d_activeSourceIndices,
        d_activeTargetIndices,
        d_activeConductances,
        MAX_ACTIVE_SYNAPSES
    );

    // Get count
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_activeSynapseCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > MAX_ACTIVE_SYNAPSES) count = MAX_ACTIVE_SYNAPSES;

    // Resize host vector
    activeSynapses.resize(count);

    if (count > 0) {
        // Temporary buffers
        std::vector<int> source(count);
        std::vector<int> target(count);
        std::vector<float> cond(count);

        CUDA_CHECK(cudaMemcpy(source.data(), d_activeSourceIndices, count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(target.data(), d_activeTargetIndices, count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cond.data(), d_activeConductances, count * sizeof(float), cudaMemcpyDeviceToHost));

        // Fill struct vector
        for (int i = 0; i < count; i++) {
            activeSynapses[i].sourceIndex = source[i];
            activeSynapses[i].targetIndex = target[i];
            activeSynapses[i].conductance = cond[i];
        }
    }
}

void CUDASimulator::getSystemAverages(double& avgVoltage, double& avgSensitivity) {
    avgVoltage = (double)m_systemAvgVoltage;
    avgSensitivity = (double)m_systemAvgSensitivity;
}

void CUDASimulator::cleanup() {
    if (!m_initialized) return;

    printf("CUDASimulator: Cleaning up GPU resources...\n");

    freeNeuronDataGPU(m_neurons);
    freeSynapseDataGPU(m_synapses);
    freeSynapseConnectivityGPU(m_connectivity);
    freeConductanceBuffersGPU(m_conductances);

    CUDA_CHECK(cudaFree(d_voltageSum));
    CUDA_CHECK(cudaFree(d_sensitivitySum));
    CUDA_CHECK(cudaFree(d_rngStates));

    CUDA_CHECK(cudaFree(d_activeSynapseCount));
    CUDA_CHECK(cudaFree(d_activeSourceIndices));
    CUDA_CHECK(cudaFree(d_activeTargetIndices));
    CUDA_CHECK(cudaFree(d_activeConductances));

    m_initialized = false;
    printf("CUDASimulator: Cleanup complete!\n");
}

// Assuming CUDASimulator class definition is in cuda_simulator.hpp
// and looks something like this (adding d_rngStates):
/*
#ifndef CUDA_SIMULATOR_HPP
#define CUDA_SIMULATOR_HPP

#include <vector>
#include "neuron.hpp" // Assuming Neuron struct is defined here
#include "synapse.hpp" // Assuming Synapse struct is defined here
#include <curand_kernel.h> // For curandState

// Forward declarations for GPU data structures
struct NeuronDataGPU;
struct SynapseDataGPU;
struct SynapseConnectivityGPU;
struct ConductanceBuffersGPU;

// Forward declarations for kernel launch functions
void allocateNeuronDataGPU(NeuronDataGPU& data, int numNeurons);
void freeNeuronDataGPU(NeuronDataGPU& data);
void copyNeuronDataToGPU(NeuronDataGPU& gpuData, const std::vector<Neuron>& cpuData);
void copyNeuronDataFromGPU(std::vector<Neuron>& cpuData, const NeuronDataGPU& gpuData);

void allocateSynapseDataGPU(SynapseDataGPU& data, int totalSynapses, int maxDelayLength);
void freeSynapseDataGPU(SynapseDataGPU& data);
void copySynapseDataToGPU(SynapseDataGPU& gpuData, const std::vector<Neuron>& cpuNeurons);

void allocateSynapseConnectivityGPU(SynapseConnectivityGPU& data, int numNeurons, int totalSynapses);
void freeSynapseConnectivityGPU(SynapseConnectivityGPU& data);
void buildSynapseConnectivityGPU(SynapseConnectivityGPU& gpuData, const std::vector<Neuron>& cpuNeurons);

void allocateConductanceBuffersGPU(ConductanceBuffersGPU& data, int numNeurons);
void freeConductanceBuffersGPU(ConductanceBuffersGPU& data);

struct SimulationConstants;
void uploadSimulationConstants(const SimulationConstants& constants);

// Kernel launch functions
void launchUpdateSynapses(SynapseDataGPU& synapses);
void launchClearConductances(ConductanceBuffersGPU& conductances);
void launchAccumulateConductances(SynapseDataGPU& synapses, ConductanceBuffersGPU& conductances);
void launchUpdateNeurons(NeuronDataGPU& neurons, ConductanceBuffersGPU& conductances, float panicInhibition, curandState* d_rngStates); // Updated signature
void launchStdpDepression(SynapseDataGPU& synapses, NeuronDataGPU& neurons);
void launchProcessSpikes(NeuronDataGPU& neurons, SynapseDataGPU& synapses, SynapseConnectivityGPU& connectivity);
void launchStdpPotentiation(SynapseDataGPU& synapses, NeuronDataGPU& neurons);
void launchHomeostasis(NeuronDataGPU& neurons);
void launchComputeSystemAverages(NeuronDataGPU& neurons, float* d_voltageSum, float* d_sensitivitySum);
void launchInitRNG(curandState* d_rngStates, int numNeurons, unsigned long long seed); // New function

class CUDASimulator {
public:
    CUDASimulator();
    ~CUDASimulator();

    void initialize(const std::vector<Neuron>& cpuNeurons, int numNeurons);
    void step();
    void injectNoise(int neuronIdx, float current);
    void syncToHost(std::vector<Neuron>& cpuNeurons);
    void getSystemAverages(double& avgVoltage, double& avgSensitivity);
    void cleanup();

private:
    int m_ticks;
    int m_numNeurons;
    int m_maxDelayLength;
    float m_systemAvgVoltage;
    float m_systemAvgSensitivity;
    bool m_initialized;

    NeuronDataGPU m_neurons;
    SynapseDataGPU m_synapses;
    SynapseConnectivityGPU m_connectivity;
    ConductanceBuffersGPU m_conductances;

    float* d_voltageSum;
    float* d_sensitivitySum;
    curandState* d_rngStates; // New member variable
};

#endif // CUDA_SIMULATOR_HPP
*/
