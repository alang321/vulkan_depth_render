#ifndef VULKAN_CUDA_INTEROP_H
#define VULKAN_CUDA_INTEROP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <cstring>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

// Helper macro for CUDA error checking (you might want a more robust solution)
#define checkCudaErrors(val)                                         \
    do                                                               \
    {                                                                \
        cudaError_t err = (val);                                     \
        if (err != cudaSuccess)                                      \
        {                                                            \
            SPDLOG_ERROR("CUDA error: {}", cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));       \
        }                                                            \
    } while (0)

class VulkanCudaInterop
{
public:
    VulkanCudaInterop() = default;
    VulkanCudaInterop(VkInstance instance, VkDevice vulkanDevice, VkPhysicalDevice physicalDevice, uint32_t imageWidth, uint32_t imageHeight, 
                      VkDeviceMemory image_memory, VkSemaphore renderFinishedSemaphores, VkImage depthImage);
    ~VulkanCudaInterop();

    void init(VkInstance instance, VkDevice vulkanDevice, VkPhysicalDevice physicalDevice, uint32_t imageWidth, uint32_t imageHeight, 
              VkDeviceMemory image_memory, VkSemaphore renderFinishedSemaphores, VkImage depthImage);

    // Call before launching CUDA operations that need Vulkan synchronization.
    void prepareForCudaAccess(cudaStream_t stream = nullptr);

    // Debug function to dump the CUDA depth image array to a file.
    // This function copies the data from the CUDA array to host memory, wraps it in an OpenCV cv::Mat,
    // converts it for visualization, and saves it using imwrite.
    void dumpCudaDepthImageToFile(const std::string &filename);

    // get a pointer to the cuda array
    cudaArray_t getCudaDepthImageArray() { return cudaDepthImageArray; }

private:
    bool initialized = false;

    // Vulkan members
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkSemaphore renderFinishedSemaphores;
    uint8_t vkDeviceUUID[VK_UUID_SIZE];
    VkDeviceMemory vkImageMemory;
    VkImage vkDepthImage;
    VkDeviceSize totalImageMemSize;

    // pfn
    PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = nullptr;
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;

    // depth image members
    uint32_t imageWidth;
    uint32_t imageHeight;

    // CUDA members
    cudaExternalSemaphore_t cudaExtRenderFinishedSemaphore;
    cudaExternalMemory_t cudaExtMemDepthImageBuffer;
    cudaArray_t cudaDepthImageArray; // For Vulkan depth image
    cudaSurfaceObject_t surfaceObjDepthInput; // For accessing the depth image in CUDA, not suere if we will actually ever use this

    // Internal helper functionsge
    void getVulkanDeviceUUID();
    void setCudaVkDevice();
    void cudaVkImportSemaphore();
    void cudaVkImportDepthImageMem();
    void cleanup();

    // Platform-specific implementations: You must implement these using the appropriate Vulkan extension calls.
    int getVkImageMemHandle(VkDeviceMemory memory);
    int getVkSemaphoreHandle(VkSemaphore semaphore);
    VkDeviceSize getDepthImageMemoryRequirements(VkImage depthImage);
};

#endif // VULKAN_CUDA_INTEROP_H

