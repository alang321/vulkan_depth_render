#ifndef PARSE_FLIGHTPLAN_H
#define PARSE_FLIGHTPLAN_H

#include <string> 
#include <array> 
#include <vector>

struct GateType {
    float size_outer;
    float size_inner;
    float thickness;
    std::string name;
    std::string sim_model;

    // flags
    bool is_double_gate = false;
};

struct GatePosition {
	float x;
	float y;
	float z;
	float psi;
	
	GateType type;
};

inline std::array<GateType, 4> gate_types = {
    GateType{2.7, 1.5, 0.145, "A2RL", "a2rl_gate", false},
    GateType{2.7, 1.5, 0.145, "A2RL_DOUBLE", "a2rl_gate", true},
    GateType{2.1, 1.5, 0.05, "MAVLAB", "mavlab_gate", false},
    GateType{2.1, 1.5, 0.05, "MAVLAB_DOUBLE", "mavlab_gate", true}
};

std::vector<GatePosition> parseFlightplan(const std::string& fp_file);

#endif // PARSE_FLIGHTPLAN_H

