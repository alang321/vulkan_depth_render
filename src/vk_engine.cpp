
#include "vk_engine.h"

#include "vk_images.h"
#include "vk_descriptors.h"

#include <vk_initializers.h>
#include <span>
#include <vk_types.h>

#include "VkBootstrap.h"

#include <glm/gtx/transform.hpp>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#ifdef DEBUG_BUILD
constexpr bool bUseValidationLayers = true;
#else
constexpr bool bUseValidationLayers = false;
#endif

// we want to immediately abort when there is an error. In normal engines this
// would give an error message to the user, or perform a dump of state.
using namespace std;

VulkanEngine *loadedEngine = nullptr;

VulkanEngine &VulkanEngine::Get()
{
    return *loadedEngine;
}

void VulkanEngine::init(int width, int height, const glm::mat3 &camera_intrinsics, float z_near, float z_far, const std::vector<GatePosition> &flightplan)
{
    _z_far = z_far;
    _z_near = z_near;
    _vk_projection_mat = intrinsics_to_vk_proj(camera_intrinsics, width, height, z_near, z_far);

    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    _imgExtent.width = width;
    _imgExtent.height = height;
    _drawExtent = _imgExtent;

    init_vulkan();

    init_depth_img();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_mesh(flightplan);

    _push_constants.vertexBuffer = _main_mesh.vertexBufferAddress;
    _push_constants.far_plane = z_far;

    init_cuda_interop();

    // everything went fine
    _isInitialized = true;

    SPDLOG_INFO("Vulkan Mask Rendering Engine initialized");
}

void VulkanEngine::init_cuda_interop()
{
    cudaInterop.init(_instance, _device, _chosenGPU, _imgExtent.width, _imgExtent.height, _depthImage.allocation, _renderFinishedSemaphore, _depthImage.image);
}

void VulkanEngine::cleanup()
{
    if (_isInitialized)
    {

        // make sure the gpu has stopped doing its things
        vkDeviceWaitIdle(_device);

        _mainDeletionQueue.flush();

        destroy_buffer(_main_mesh.indexBuffer);
        destroy_buffer(_main_mesh.vertexBuffer);

        vmaDestroyAllocator(_allocator);

        vkDestroyDevice(_device, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);
    }
}

void VulkanEngine::draw_main(VkCommandBuffer cmd)
{
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_imgExtent, &depthAttachment);

    vkCmdBeginRendering(cmd, &renderInfo);

    draw_geometry(cmd);

    vkCmdEndRendering(cmd);
}

void VulkanEngine::draw()
{
    VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, UINT64_MAX));
    VK_CHECK(vkResetFences(_device, 1, &_renderFence));

    // now that we are sure that the commands finished executing, we can safely reset the command buffer to begin recording again.
    VK_CHECK(vkResetCommandBuffer(_mainCommandBuffer, 0));

    // naming it cmd for shorter writing
    VkCommandBuffer cmd = _mainCommandBuffer;

    // begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    //> draw_first
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    // transition our main draw image into general layout so we can write into it
    // we will overwrite it all so we dont care about what was the older layout
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    draw_main(cmd);

#ifdef SAVE_IMAGES
    save_to_file("depth" + std::to_string(_frameNumber) + ".png", cmd);
#else
    // Finalize the command buffer
    VK_CHECK(vkEndCommandBuffer(cmd));
    // Submit the command buffer and wait for execution to complete
    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo signalSemaphoreInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, _renderFinishedSemaphore);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalSemaphoreInfo, nullptr);
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _renderFence));
#endif

    _frameNumber++;
    return;
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0;
    viewport.width = static_cast<float>(_drawExtent.width);
    viewport.height = static_cast<float>(_drawExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = _drawExtent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdPushConstants(cmd, _depthPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &_push_constants);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _depthPipeline);

    vkCmdBindIndexBuffer(cmd, _main_mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, _main_mesh.indexCount, 1, 0, 0, 0);
}

void VulkanEngine::render_image(const glm::mat4 &view)
{
    bool bQuit = false;

    _push_constants.view_proj_matrix = _vk_projection_mat * view;

    draw();
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    // allocate buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = nullptr;
    bufferInfo.size = allocSize;

    bufferInfo.usage = usage;

    VmaAllocationCreateInfo vmaallocInfo = {};
    vmaallocInfo.usage = memoryUsage;
    vmaallocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    AllocatedBuffer newBuffer;

    // allocate the buffer
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation,
                             &newBuffer.info));

    return newBuffer;
}

GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexBuffer = create_buffer(vertexBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                            VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferDeviceAddressInfo deviceAdressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = newSurface.vertexBuffer.buffer};
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAdressInfo);

    newSurface.indexBuffer = create_buffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VMA_MEMORY_USAGE_GPU_ONLY);

    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void *data = staging.allocation->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char *)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd)
                     {
        VkBufferCopy vertexCopy { 0 };
        vertexCopy.dstOffset = 0;
        vertexCopy.srcOffset = 0;
        vertexCopy.size = vertexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        VkBufferCopy indexCopy { 0 };
        indexCopy.dstOffset = 0;
        indexCopy.srcOffset = vertexBufferSize;
        indexCopy.size = indexBufferSize;

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy); });

    destroy_buffer(staging);

    newSurface.indexCount = indices.size();

    return newSurface;
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function)
{
    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
    
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;
    // begin the command buffer recording. We will use this command buffer exactly
    // once, so we want to let vulkan know that
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

    // submit command buffer to the queue and execute it.
    //  _renderFence will now block until the graphic commands finish execution
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer &buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

void VulkanEngine::init_vulkan()
{
    vkb::InstanceBuilder builder;

    // make the vulkan instance, with basic debug features
    auto inst_ret = builder.set_app_name("Example Vulkan Application")
                        .request_validation_layers(bUseValidationLayers)
                        .use_default_debug_messenger()
                        .require_api_version(1, 3, 0)
                        .build();

    vkb::Instance vkb_inst = inst_ret.value();

    // grab the instance
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    // use vkbootstrap to select a gpu.
    vkb::PhysicalDeviceSelector selector{vkb_inst};

    // VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    // VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME

    std::vector<const char *> deviceExtensions = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                                  VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                                                  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                                                  VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME};

    // print all available gpus
    auto gpus = selector.defer_surface_initialization().add_required_extensions(deviceExtensions).require_present(false).set_minimum_version(1, 3).set_required_features_13(features13).set_required_features_12(features12).select_devices();
    for (auto &gpu : gpus.value())
    {
        SPDLOG_INFO("Available GPU: {}", gpu.properties.deviceName);
        // log if the gpu is a discrete gpu
        if (gpu.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            SPDLOG_INFO("GPU is discrete");
        }
    }

    vkb::PreferredDeviceType preferredType = vkb::PreferredDeviceType::discrete;

    auto physicalDevices = selector.defer_surface_initialization().require_present(false).set_minimum_version(1, 3).prefer_gpu_device_type(preferredType).set_required_features_13(features13).set_required_features_12(features12).select_devices().value();

    vkb::PhysicalDevice physicalDevice = physicalDevices[0];
    // select nvidia gpu if available
    for (auto &physicalDeviceTmp : physicalDevices)
    {
        if (physicalDeviceTmp.properties.vendorID == 0x10DE)
        {
            physicalDevice = physicalDeviceTmp;
            break;
        }
    }

    SPDLOG_INFO("Selected GPU: {}", physicalDevice.properties.deviceName);

    physicalDevice.enable_extensions_if_present(deviceExtensions);

    // create the logical vulkan device
    vkb::DeviceBuilder deviceBuilder{physicalDevice};

    vkb::Device vkbDevice = deviceBuilder.build().value();

    // Get the VkDevice handle used in the rest of a vulkan application
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // use vkbootstrap to get a Graphics queue
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();

    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // initialize the memory allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);
}

uint32_t VulkanEngine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(_chosenGPU, &memProperties); // _physicalDevice should be set at initialization
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanEngine::init_depth_img()
{
    // The depth image will match the window size.
    VkExtent3D depthImageExtent = {
        _imgExtent.width,
        _imgExtent.height,
        1};

    // Set the format to 32-bit float.
    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = depthImageExtent;

    // Define the usage flags.
    VkImageUsageFlags depthImageUsages = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | // for copying if needed
                                         VK_IMAGE_USAGE_SAMPLED_BIT;

    // Set up the image creation info.
    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, depthImageExtent);

    // Chain in external memory create info so the image can be shared with CUDA.
    VkExternalMemoryImageCreateInfo externalImageCreateInfo = {};
    externalImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;

    externalImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    externalImageCreateInfo.pNext = dimg_info.pNext;
    dimg_info.pNext = &externalImageCreateInfo;

    // Create the depth image.
    if (vkCreateImage(_device, &dimg_info, nullptr, &_depthImage.image) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create depth image!");
    }

    // Query the image memory requirements.
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(_device, _depthImage.image, &memRequirements);

    // Prepare the allocation info with exportable memory.
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    // Chain the export memory allocation info.
    VkExportMemoryAllocateInfoKHR exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;

    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    allocInfo.pNext = &exportAllocInfo;

    // Find a memory type index that is device local.
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Allocate memory for the image.
    VkDeviceMemory imageMemory; // Make sure _depthImage.allocation is of type VkDeviceMemory.
    if (vkAllocateMemory(_device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate depth image memory!");
    }

    // Bind the allocated memory to the depth image.
    if (vkBindImageMemory(_device, _depthImage.image, imageMemory, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to bind depth image memory!");
    }
    // Store the memory handle for cleanup.
    _depthImage.allocation = imageMemory;

    // Create an image view for the depth image.
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    if (vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create depth image view!");
    }

    // Add cleanup functions to your deletion queue.
    _mainDeletionQueue.push_function([this]()
                                     {
    
        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vkDestroyImage(_device, _depthImage.image, nullptr);
        vkFreeMemory(_device, _depthImage.allocation, nullptr); });
}

void VulkanEngine::init_commands()
{
    // create a command pool for commands submitted to the graphics queue.
    // we also want the pool to allow for resetting of individual command buffers
    VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool));

    // allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_mainCommandBuffer));

    _mainDeletionQueue.push_function([this]()
                                     { vkDestroyCommandPool(_device, _commandPool, nullptr); });

    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    // allocate the default command buffer that we will use for rendering
    VkCommandBufferAllocateInfo cmdAllocInfo2 = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo2, &_immCommandBuffer));

    _mainDeletionQueue.push_function([this]()
                                     { vkDestroyCommandPool(_device, _immCommandPool, nullptr); });
}

void VulkanEngine::init_sync_structures()
{
    // create synchronization structures
    // we only need one fence since we can't overlap frames
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([this]()
                                     { vkDestroyFence(_device, _immFence, nullptr); });

    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_renderFence));
    _mainDeletionQueue.push_function([this]()
                                     { vkDestroyFence(_device, _renderFence, nullptr); });

    // can be used for vk cuda interop
    createSyncObjectsExt();
}

void VulkanEngine::createSyncObjectsExt()
{
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    memset(&semaphoreInfo, 0, sizeof(semaphoreInfo));
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
    vulkanExportSemaphoreCreateInfo.sType =
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

    vulkanExportSemaphoreCreateInfo.pNext = NULL;
    vulkanExportSemaphoreCreateInfo.handleTypes =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

    VK_CHECK(vkCreateSemaphore(_device, &semaphoreInfo, nullptr, &_renderFinishedSemaphore));

    _mainDeletionQueue.push_function([this]()
                                     { vkDestroySemaphore(_device, _renderFinishedSemaphore, nullptr); });
}

void VulkanEngine::init_pipelines()
{
    // Define push constant range for the camera pose (a 4x4 matrix)
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(GPUDrawPushConstants);

    // Create a pipeline layout with no descriptor sets but with push constants for the camera pose
    VkPipelineLayoutCreateInfo depthLayout{};
    depthLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    depthLayout.pNext = nullptr;
    depthLayout.setLayoutCount = 0;
    depthLayout.pSetLayouts = nullptr;
    depthLayout.pushConstantRangeCount = 1;
    depthLayout.pPushConstantRanges = &pushConstantRange;

    VK_CHECK(vkCreatePipelineLayout(_device, &depthLayout, nullptr, &_depthPipelineLayout));

    // Load our depth-only shaders (vertex and fragment)
    // The vertex shader should output clip-space positions and the fragment shader can be empty.
    VkShaderModule depthVertShader;
    if (!vkutil::load_shader_module("shaders/depth.vert.spv", _device, &depthVertShader))
    {
        SPDLOG_ERROR("Failed to create depth vertex shader module");
        return;
    }

    // Build a graphics pipeline that renders only to a depth attachment
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(depthVertShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();

    // Disable color blending since no color outputs will be used.
    pipelineBuilder.disable_blending();

    // Enable depth test and write. Choose a compare op appropriate for your usage.
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_LESS_OR_EQUAL);

    // IMPORTANT: Do not set any color attachment formats. We only provide a depth format.
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    // Use created pipeline layout with push constants.
    pipelineBuilder._pipelineLayout = _depthPipelineLayout;

    // Build the pipeline (using dynamic rendering or a render pass that has only a depth attachment).
    _depthPipeline = pipelineBuilder.build_pipeline(_device);

    // Clean up shader modules now that pipeline creation is done.
    vkDestroyShaderModule(_device, depthVertShader, nullptr);

    // Register deletion to destroy our new pipeline and layout.
    _mainDeletionQueue.push_function([&]()
                                     {
        vkDestroyPipelineLayout(_device, _depthPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _depthPipeline, nullptr); });
}

void VulkanEngine::init_descriptors()
{
    // create a descriptor pool for offscreen depth image usage (only one frame, no overlap)
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
    };

    globalDescriptorAllocator.init_pool(_device, 1, sizes);
    _mainDeletionQueue.push_function(
        [&]()
        { vkDestroyDescriptorPool(_device, globalDescriptorAllocator.pool, nullptr); });

    // create a descriptor set layout for the depth image
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    _depthImageDescriptorLayout = layoutBuilder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);

    _mainDeletionQueue.push_function([&]()
                                     { vkDestroyDescriptorSetLayout(_device, _depthImageDescriptorLayout, nullptr); });

    // allocate a descriptor set that binds our offscreen depth image
    _depthImageDescriptors = globalDescriptorAllocator.allocate(_device, _depthImageDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, _depthImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        writer.update_set(_device, _depthImageDescriptors);
    }
}

glm::mat4 VulkanEngine::intrinsics_to_vk_proj(const glm::mat3 &intrinsics,
                                float width, float height,
                                float znear, float zfar)
{
    // 1. Extract intrinsics (in pixels).
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];

    SPDLOG_INFO("fx: {}, fy: {}, cx: {}, cy: {}", fx, fy, cx, cy);

    glm::mat4 P(0.0f);
    P[0][0] = 2.0f * fx / width;
    P[1][1] = 2.0f * fy / height;
    P[0][2] = 1.0f - 2.0f * (cx / width);
    P[1][2] = 2.0f * (cy / height) - 1.0f;
    P[3][2] = -1.0f;

    float n = znear;
    float f = zfar;
    if (f <= 0.0f) {
        P[2][2] = -1.0f;
        P[2][3] = -2.0f * n;
    } else {
        P[2][2] = (f + n) / (n - f);
        P[2][3] = (2.0f * f * n) / (n - f);
    }

    // Construct the Vulkan correction matrix:
    //    - Flip Y
    //    - Remap depth from [-1,1] to [0,1]
    glm::mat4 vulkanClip(1.0f);
    vulkanClip[1][1] = -1.0f;  // Flip Y
    vulkanClip[2][2] = 0.5f;   // Depth remap factor
    vulkanClip[2][3] = 0.5f;   // Depth remap offset

    glm::mat4 M_vk = glm::transpose(vulkanClip) * glm::transpose(P); // to be honest, i thought i wouldn't need to transpose it, but it seems I messed up column/row major order somewhere

    return M_vk;
}

void VulkanEngine::init_mesh(const std::vector<GatePosition> &flightplan)
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    struct gate_pos_new
    {
        float x;
        float y;
        float z;
        float psi;
        float size_outer;
        float size_inner;
        float thickness;
    };

    // loop through the flightplan and deal with double gates
    std::vector<gate_pos_new> flightplan_new;
    for (const auto &gatePos : flightplan)
    {
        if (gatePos.type.is_double_gate)
        {
            // create two gates
            gate_pos_new gate1;
            gate1.x = gatePos.x;
            gate1.y = gatePos.y;
            gate1.z = gatePos.z;
            gate1.psi = gatePos.psi;
            gate1.size_outer = gatePos.type.size_outer;
            gate1.size_inner = gatePos.type.size_inner;
            gate1.thickness = gatePos.type.thickness;

            gate_pos_new gate2;
            gate2.x = gatePos.x;
            gate2.y = gatePos.y;
            gate2.z = gatePos.z - gatePos.type.size_outer;
            gate2.psi = gatePos.psi;
            gate2.size_outer = gatePos.type.size_outer;
            gate2.size_inner = gatePos.type.size_inner;
            gate2.thickness = gatePos.type.thickness;

            flightplan_new.push_back(gate1);
            flightplan_new.push_back(gate2);
        }
        else
        {
            gate_pos_new gate;
            gate.x = gatePos.x;
            gate.y = gatePos.y;
            gate.z = gatePos.z;
            gate.psi = gatePos.psi;
            gate.size_outer = gatePos.type.size_outer;
            gate.size_inner = gatePos.type.size_inner;
            gate.thickness = gatePos.type.thickness;

            flightplan_new.push_back(gate);
        }
    }

    // For each gate in the flightplan, generate its faces and append to the overall mesh.
    for (const auto &gatePos : flightplan_new)
    {
        // Get parameters from the gate type.
        float outer_size = gatePos.size_outer;
        float inner_size = gatePos.size_inner;
        float thickness = gatePos.thickness;
        float psi = gatePos.psi + M_PI / 2.0f;

        // Precompute rotation components (rotation about Y)
        float cos_psi = std::cos(psi);
        float sin_psi = std::sin(psi);

        // Define two face “levels”: one at +thickness/2 and one at -thickness/2.
        // (In the Python example these correspond to the top and bottom faces.)
        float t_values[2] = {thickness / 2.0f, -thickness / 2.0f};

        // Loop over the two faces.
        for (int face = 0; face < 2; ++face)
        {
            float t = t_values[face];
            // Define 8 local vertices for this face.
            // Outer square (indices 0-3) and inner square (indices 4-7)
            std::array<glm::vec3, 8> localVerts = {
                glm::vec3(-outer_size / 2.0f, t, outer_size / 2.0f),  // 0: outer top-left
                glm::vec3(outer_size / 2.0f, t, outer_size / 2.0f),   // 1: outer top-right
                glm::vec3(outer_size / 2.0f, t, -outer_size / 2.0f),  // 2: outer bottom-right
                glm::vec3(-outer_size / 2.0f, t, -outer_size / 2.0f), // 3: outer bottom-left
                glm::vec3(-inner_size / 2.0f, t, inner_size / 2.0f),  // 4: inner top-left
                glm::vec3(inner_size / 2.0f, t, inner_size / 2.0f),   // 5: inner top-right
                glm::vec3(inner_size / 2.0f, t, -inner_size / 2.0f),  // 6: inner bottom-right
                glm::vec3(-inner_size / 2.0f, t, -inner_size / 2.0f)  // 7: inner bottom-left
            };

            // Record the starting index for these new vertices.
            uint32_t startIndex = static_cast<uint32_t>(vertices.size());
            // Transform each vertex by applying the rotation (around Z) and then translation.
            for (auto &v : localVerts)
            {
                // Rotate around Z axis.
                float localX = v.x;
                float localY = v.y;
                float rotatedX = cos_psi * localX - sin_psi * localY;
                float rotatedY = sin_psi * localX + cos_psi * localY;
                // Update vertex position with translation.
                v.x = rotatedX + gatePos.x;
                v.y = rotatedY + gatePos.y;
                v.z = v.z + gatePos.z; // v.z remains unchanged by Z rotation

                // Add the transformed vertex.
                vertices.push_back(Vertex{{v.x, v.y, v.z}});
            }

            // Define the face’s triangles.
            // The triangles are defined in terms of the local vertex indices.
            std::vector<std::array<uint32_t, 3>> faceTriangles = {
                {0, 1, 5}, {0, 5, 4}, // Top portion of the ring
                {1, 2, 6},
                {1, 6, 5}, // Right portion
                {2, 3, 7},
                {2, 7, 6}, // Bottom portion
                {3, 0, 4},
                {3, 4, 7} // Left portion
            };

            // For the bottom face, reverse the winding order so that the face normals remain correct.
            if (face == 1)
            {
                for (auto &tri : faceTriangles)
                {
                    std::swap(tri[0], tri[2]);
                }
            }

            // Append these triangles to the global index list, offset by the starting index.
            for (const auto &tri : faceTriangles)
            {
                indices.push_back(startIndex + tri[0]);
                indices.push_back(startIndex + tri[1]);
                indices.push_back(startIndex + tri[2]);
            }
        }
    }

    // Once all gate faces have been processed and added to 'vertices' and 'indices',
    // create (upload) the mesh.
    _main_mesh = uploadMesh(indices, vertices);
}

#ifdef SAVE_IMAGES
void VulkanEngine::save_to_file(const std::string &filename, VkCommandBuffer cmd)
{
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Create a buffer to store the depth image data
    VkDeviceSize imageSize = _imgExtent.width * _imgExtent.height * sizeof(float);
    AllocatedBuffer depthBuffer = create_buffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    // Copy the depth image to the buffer
    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = {_imgExtent.width, _imgExtent.height, 1};

    vkCmdCopyImageToBuffer(cmd, _depthImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, depthBuffer.buffer, 1, &copyRegion);

    // Transition the depth image back to depth attachment layout
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // Finalize the command buffer
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit the command buffer and wait for execution to complete
    VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _renderFence));

    VK_CHECK(vkWaitForFences(_device, 1, &_renderFence, true, UINT64_MAX));
    VK_CHECK(vkResetFences(_device, 1, &_renderFence)); // Add fence reset

    // Map the buffer memory and save the depth data to a PNG file using OpenCV
    void *data;
    vmaMapMemory(_allocator, depthBuffer.allocation, &data);

    // convert to opencv image
    cv::Mat depthMat(_imgExtent.height, _imgExtent.width, CV_32FC1, data);
    cv::Mat depthMat8;
    depthMat.convertTo(depthMat8, CV_8UC1, 255.0f, 0.0f);

    // write to disk
    std::string output = "image_output/" + filename;
    cv::imwrite(output, depthMat8);

    vmaUnmapMemory(_allocator, depthBuffer.allocation);

    // Clean up the buffer
    destroy_buffer(depthBuffer);
}
#endif
