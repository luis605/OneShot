#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <cstdint>

class Simulator {
public:
    Simulator();
    void initialize();
    void run();

private:
    uint64_t time;
};

#endif // SIMULATOR_H
