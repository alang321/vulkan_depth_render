#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

#include <parse_flightplan.h>

std::vector<GatePosition> parseFlightplan(const std::string &fp_file)
{
    // Expand realtive path
    std::string absolute_path = std::filesystem::absolute(fp_file).string();

    std::ifstream ifs(absolute_path);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Could not open flightplans file: " + absolute_path);
    }

    nlohmann::json flightplan_json;
    ifs >> flightplan_json;
    ifs.close();

    // reset flightplan
    std::vector<GatePosition> gates;

    if (flightplan_json.contains("waypoints"))
    {
        const auto &waypoints_json = flightplan_json["waypoints"];
        for (const auto &wp : waypoints_json)
        {
            if (wp["type"].get<std::string>() == "GATE")
            {
                GatePosition gate;
                if (wp.contains("gate_type"))
                {
                    // loop through gate types to find the correct one
                    bool found = false;
                    for (auto &gate_type : gate_types)
                    {
                        if (gate_type.name == wp["gate_type"].get<std::string>())
                        {
                            gate.type = gate_type;
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        throw std::runtime_error("Gate type not found: " + wp["gate_type"].get<std::string>());
                    }

                    gate.x = wp["x"].get<float>();
                    gate.y = wp["y"].get<float>();
                    gate.z = wp["z"].get<float>();
                    gate.psi = wp["yaw"].get<float>() * M_PI / 180.0f;
                    gates.push_back(gate);
                }
                else
                {
                    throw std::runtime_error("Gate type not specified");
                }
            }
            else if (wp["type"].get<std::string>() == "START")
            {
            }
            else
            {
                throw std::runtime_error("Unknown waypoint type: " + wp["type"].get<std::string>());
            }
        }
    }
    else
    {
        throw std::runtime_error("Flightplan does not specify any waypoints.");
    }

    return gates;
}
