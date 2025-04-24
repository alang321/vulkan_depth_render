#include "vk_cuda_interop.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuda.h>
#include <vulkan/vulkan.h>
#include "spdlog/spdlog.h"
#include "vk_cuda_interop.h"
#include <vulkan/vk_enum_string_helper.h>

// Constructor: initialize members and zero-out UUID.

VulkanCudaInterop::VulkanCudaInterop(VkInstance instance, VkDevice vulkanDevice, VkPhysicalDevice physicalDevice, uint32_t imageWidth, uint32_t imageHeight, 
    VkDeviceMemory image_memory, VkSemaphore renderFinishedSemaphores, VkImage depthImage)
{
    init(instance, vulkanDevice, physicalDevice, imageWidth, imageHeight, 
         image_memory, renderFinishedSemaphores, depthImage);
}

void VulkanCudaInterop::init(VkInstance instance, VkDevice vulkanDevice, VkPhysicalDevice physicalDevice, uint32_t imageWidth, uint32_t imageHeight, 
    VkDeviceMemory image_memory, VkSemaphore renderFinishedSemaphores, VkImage depthImage)
{
    this->instance = instance;
    this->device = vulkanDevice;
    this->physicalDevice = physicalDevice;
    this->imageWidth = imageWidth;
    this->imageHeight = imageHeight;
    this->vkImageMemory = image_memory;
    this->renderFinishedSemaphores = renderFinishedSemaphores;
    this->vkDepthImage = depthImage;
    this->totalImageMemSize = getDepthImageMemoryRequirements(depthImage);
    
    this->cudaExtRenderFinishedSemaphore = nullptr;
    this->cudaExtMemDepthImageBuffer = nullptr;
    this->cudaDepthImageArray = nullptr;
    this->surfaceObjDepthInput = 0;

    vkGetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetInstanceProcAddr(instance, "vkGetSemaphoreFdKHR"));
    vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetInstanceProcAddr(instance, "vkGetMemoryFdKHR"));

    memset(vkDeviceUUID, 0, sizeof(vkDeviceUUID));
    getVulkanDeviceUUID();
    setCudaVkDevice();
    cudaVkImportSemaphore();
    cudaVkImportDepthImageMem();
}

// Destructor: clean up Vulkan and CUDA resources.
VulkanCudaInterop::~VulkanCudaInterop()
{
    cleanup();
}

void VulkanCudaInterop::prepareForCudaAccess(cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = 0; // Set as required

    // Schedule the semaphore wait in the stream without blocking the CPU.
    cudaError_t err = cudaWaitExternalSemaphoresAsync(&cudaExtRenderFinishedSemaphore, &waitParams, 1, stream);
    if (err != cudaSuccess)
    {
        // Handle the error immediately if needed, but avoid blocking.
        SPDLOG_ERROR("Error scheduling cudaWaitExternalSemaphoresAsync: {}", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }

    // Defer error checking for the stream if necessary.
    // cudaStreamSynchronize(stream);
}

// Helper: retrieve the Vulkan device UUID.
void VulkanCudaInterop::getVulkanDeviceUUID()
{
    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");

    if (fpGetPhysicalDeviceProperties2 == NULL)
    {
        throw std::runtime_error("Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not found.\n");
    }

    VkPhysicalDeviceIDProperties deviceIDProps{};
    deviceIDProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &deviceIDProps;

    fpGetPhysicalDeviceProperties2(physicalDevice, &props2);

    memcpy(vkDeviceUUID, deviceIDProps.deviceUUID, VK_UUID_SIZE);
}

void VulkanCudaInterop::setCudaVkDevice()
{
    int current_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        SPDLOG_ERROR("CUDA error: no devices supporting CUDA.\n");
        throw std::runtime_error("CUDA error: no devices supporting CUDA.\n");
    }

    // Find the GPU which is selected by Vulkan
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        if ((deviceProp.computeMode != cudaComputeModeProhibited))
        {
            // Compare the cuda device UUID with vulkan UUID
            int ret = memcmp(&deviceProp.uuid, &vkDeviceUUID, VK_UUID_SIZE);
            if (ret == 0)
            {
                checkCudaErrors(cudaSetDevice(current_device));
                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
                SPDLOG_INFO("GPU Device {}: \"{}\" with compute capability {}.{}",
                            current_device, deviceProp.name, deviceProp.major,
                            deviceProp.minor);
            }
        }
        else
        {
            devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count)
    {
        SPDLOG_ERROR("CUDA error: No Vulkan-CUDA Interop capable GPU found.");
        throw std::runtime_error("CUDA error: No Vulkan-CUDA Interop capable GPU found.");
    }
}

void VulkanCudaInterop::cudaVkImportSemaphore()
{
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));

    externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(renderFinishedSemaphores);
    externalSemaphoreHandleDesc.flags = 0;

    checkCudaErrors(cudaImportExternalSemaphore(&cudaExtRenderFinishedSemaphore, &externalSemaphoreHandleDesc));

    SPDLOG_INFO("CUDA Imported Vulkan semaphore\n");
}

void VulkanCudaInterop::cudaVkImportDepthImageMem()
{
    // Import the Vulkan depth image external memory
    cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
    memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));

    cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    // Replace getVkDepthImageMemHandle() with your function for the depth image handle.
    cudaExtMemHandleDesc.handle.fd = getVkImageMemHandle(vkImageMemory);

    cudaExtMemHandleDesc.size = totalImageMemSize; // set to the size of your depth image memory

    // Import the external memory.
    checkCudaErrors(cudaImportExternalMemory(&cudaExtMemDepthImageBuffer, &cudaExtMemHandleDesc));

    // Set up the mipmapped array descriptor with a single level.
    cudaExternalMemoryMipmappedArrayDesc mipmappedArrayDesc;
    memset(&mipmappedArrayDesc, 0, sizeof(mipmappedArrayDesc));
    cudaExtent extent = make_cudaExtent(imageWidth, imageHeight, 0);

    // Define a channel format for a 32-bit float depth image.
    cudaChannelFormatDesc formatDesc;
    formatDesc.x = 32;    // 32 bits for the float channel
    formatDesc.y = 0;
    formatDesc.z = 0;
    formatDesc.w = 0;
    formatDesc.f = cudaChannelFormatKindFloat;

    mipmappedArrayDesc.offset = 0;
    mipmappedArrayDesc.formatDesc = formatDesc;
    mipmappedArrayDesc.extent = extent;
    mipmappedArrayDesc.flags = 0;
    mipmappedArrayDesc.numLevels = 1; // single level for a non-mipmapped image

    // Get the mapped mipmapped array from the external memory.
    cudaMipmappedArray_t mipmappedArray;
    checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(
        &mipmappedArray, cudaExtMemDepthImageBuffer, &mipmappedArrayDesc));

    // Extract the base level (level 0) array.
    checkCudaErrors(cudaGetMipmappedArrayLevel(&cudaDepthImageArray, mipmappedArray, 0));

    // (Optional) Create a surface object to allow CUDA kernels to access the depth image.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaDepthImageArray;

    checkCudaErrors(cudaCreateSurfaceObject(&surfaceObjDepthInput, &resDesc));

    SPDLOG_INFO("CUDA Vulkan depth image buffer imported and mapped.");
}

// Clean up all Vulkan and CUDA resources.
void VulkanCudaInterop::cleanup()
{
    // Destroy the imported CUDA semaphore used for Vulkan synchronization.
    if (cudaExtRenderFinishedSemaphore) {
        checkCudaErrors(cudaDestroyExternalSemaphore(cudaExtRenderFinishedSemaphore));
        cudaExtRenderFinishedSemaphore = nullptr;
    }

    // Destroy the CUDA surface object for the depth image (if it was created).
    if (surfaceObjDepthInput) {
        checkCudaErrors(cudaDestroySurfaceObject(surfaceObjDepthInput));
        surfaceObjDepthInput = 0;
    }

    // Destroy the imported external memory for the Vulkan depth image.
    if (cudaExtMemDepthImageBuffer) {
        checkCudaErrors(cudaDestroyExternalMemory(cudaExtMemDepthImageBuffer));
        cudaExtMemDepthImageBuffer = nullptr;
    }

    // Note: The cudaDepthImageArray is not explicitly destroyed.
    // It is managed by the lifetime of cudaExtMemDepthImageBuffer.
    SPDLOG_INFO("CUDA cleanup completed.");
}


// Platform-specific: get file descriptor for a Vulkan semaphore using vkGetSemaphoreFdKHR.
// You must ensure that the vkGetSemaphoreFdKHR function pointer is loaded (typically via vkGetDeviceProcAddr).
int VulkanCudaInterop::getVkSemaphoreHandle(VkSemaphore semaphore)
{
    VkSemaphoreGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    getFdInfo.semaphore = semaphore;
    getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    // Assuming vkGetSemaphoreFdKHR has been loaded appropriately.
    if (vkGetSemaphoreFdKHR(device, &getFdInfo, &fd) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to get semaphore file descriptor");
    }
    return fd;
}

// Platform-specific: get file descriptor for Vulkan device memory using vkGetMemoryFdKHR.
int VulkanCudaInterop::getVkImageMemHandle(VkDeviceMemory memory)
{
    VkMemoryGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.pNext = NULL;
    getFdInfo.memory = memory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    VkResult result = vkGetMemoryFdKHR(device, &getFdInfo, &fd);
    if (result != VK_SUCCESS)
    {
        SPDLOG_ERROR("Failed to get memory file descriptor: {}", string_VkResult(result));
        throw std::runtime_error("Failed to get memory file descriptor");
    }
    return fd;
}

VkDeviceSize VulkanCudaInterop::getDepthImageMemoryRequirements(VkImage depthImage)
{
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, depthImage, &memRequirements);
    return memRequirements.size;
}

void VulkanCudaInterop::dumpCudaDepthImageToFile(const std::string &filename)
{
    if (!cudaDepthImageArray)
    {
        SPDLOG_ERROR("CUDA depth image array is not initialized.");
        return;
    }

    // Assume the depth image has a single channel of float values.
    size_t pitch = imageWidth * sizeof(float);
    std::vector<float> hostData(imageWidth * imageHeight);

    // Copy data from CUDA array to host memory using the provided stream.
    checkCudaErrors(cudaMemcpy2DFromArray(
        hostData.data(),       // Destination pointer
        pitch,                 // Destination pitch in bytes
        cudaDepthImageArray,   // Source CUDA array
        0, 0,                  // Source x and y offset
        pitch,                 // Width of the data to copy (in bytes)
        imageHeight,           // Height of the data to copy (in elements)
        cudaMemcpyDeviceToHost // Direction of copy
        ));
    checkCudaErrors(cudaDeviceSynchronize());

    // Wrap host data as an OpenCV matrix.
    cv::Mat depthMat(imageHeight, imageWidth, CV_32FC1, hostData.data());

    // Convert to 8-bit for easier visualization.
    cv::Mat display;
    depthMat.convertTo(display, CV_8UC1, 255.0f);

    // Write the image to the provided filename.
    if (!cv::imwrite(filename, display))
    {
        SPDLOG_ERROR("Failed to write image to file: {}", filename);
    }
    else
    {
        SPDLOG_INFO("Successfully dumped CUDA depth image to: {}", filename);
        checkCudaErrors(cudaDeviceSynchronize());
    }
}
