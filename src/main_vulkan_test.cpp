#include <vk_engine.h>
#include <vk_types.h>
#include <parse_flightplan.h>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>
#include <iostream>
#include <IterativeEngine.hpp>
#include <cuda_runtime.h>

int main(int argc, char* argv[])
{
    VulkanEngine engine;

    int width = 360;
    int height = 360;
    glm::mat3 camera_intrinsics = glm::mat3(1.0f);
    camera_intrinsics[0][0] = 165.0f;
    camera_intrinsics[1][1] = 165.0f;
    camera_intrinsics[0][2] = 190.0f;
    camera_intrinsics[1][2] = 170.0f;

    float z_near = 0.1f;
    float z_far = 30.0f;

    std::vector<GatePosition> fp = parseFlightplan("flightplans/dhl_3dtrack_full.json");

    engine.init(width, height, camera_intrinsics, z_near, z_far, fp);

    // Create a CUDA stream with error checking
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize view transformation
    glm::mat4 view = glm::lookAt(glm::vec3(-15.0f, 0.0f, -2.5f),
                                 glm::vec3(7.0f, 0.0f, -1.0f),
                                 glm::vec3(0.0f, 0.0f, -1.0f));

    // Main loop: update view, prepare CUDA access, render, and dump image
    for (int i = 0; i < 270; i++) {
        view = glm::translate(view, glm::vec3(-0.1f, 0.0f, 0.0f));

        // Prepare engine for CUDA interop using the stream
        engine.cudaInterop.prepareForCudaAccess(stream);

        // Render the image with the updated view
        engine.render_image(view);

        // Synchronize the stream to ensure rendering is complete
        cudaStreamSynchronize(stream);

        // Dump the depth image to file
        std::string filename = "image_output/depth_image_" + std::to_string(i) + ".png";
        engine.cudaInterop.dumpCudaDepthImageToFile(filename);
    }
    

    engine.cleanup();

    // Destroy the CUDA stream to free resources
    cudaStreamDestroy(stream);

    return 0;
}
