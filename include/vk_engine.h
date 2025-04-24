// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

#include <stack>
#include <functional>
#include <span>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include <parse_flightplan.h>

#include <vk_mem_alloc.h>

#include <vk_descriptors.h>
#include <vk_pipelines.h>
#include <vk_cuda_interop.h>

struct DeletionQueue 
{
	std::stack<std::function<void()>> deletors; 

	void push_function(std::function<void()>&& function) 
	{
		deletors.push(function); 
	}

	void flush() 
	{
		while (!deletors.empty())
		{
			auto topFunc = deletors.top();
			topFunc(); // call function 
			deletors.pop(); 
		}
	}
};

struct RenderObject {
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;
    
    glm::mat4 transform;
    VkDeviceAddress vertexBufferAddress;
};

struct EngineStats {
    float frametime;
    int triangle_count;
};

class VulkanEngine {
public:
    bool _isInitialized { false };
    int _frameNumber { 0 };

    EngineStats stats;

    // singleton style getter.multiple engines is not supported
    static VulkanEngine& Get();

    // initializes everything in the engine
    void init(int width, int height, const glm::mat3& camera_intrinsics, float z_near, float z_far, const std::vector<GatePosition>& flightplan);

    // shuts down the engine
    void cleanup();

    // render a depth image
    void render_image(const glm::mat4& view);

    AllocatedImage _depthImage;

    VulkanCudaInterop cudaInterop;

private:
    // draw loop
    void draw();
    void draw_main(VkCommandBuffer cmd);
    void draw_geometry(VkCommandBuffer cmd);

    // upload a mesh into a pair of gpu buffers. If descriptor allocator is not
    // null, it will also create a descriptor that points to the vertex buffer
	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

    // image helpers
    void destroy_buffer(const AllocatedBuffer& buffer);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

    // init helpers
    void init_vulkan();
    void init_depth_img();
    void init_commands();
    void init_pipelines();
    void init_descriptors();
    void init_sync_structures();
    void init_mesh(const std::vector<GatePosition>& flightplan);
    void init_cuda_interop();

    void createSyncObjectsExt();

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    glm::mat4 intrinsics_to_vk_proj(const glm::mat3& pinhole_intrinsics, float w, float h, float znear, float zfar);

    glm::mat4 _vk_projection_mat;
    GPUDrawPushConstants _push_constants;

    VkExtent2D _imgExtent { 1700, 900 };

    VkInstance _instance;
    VkDebugUtilsMessengerEXT _debug_messenger;
    VkPhysicalDevice _chosenGPU;
    VkDevice _device;

    VkSemaphore _renderFinishedSemaphore;

    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;
	VkExtent2D _drawExtent;

    VkDescriptorPool _descriptorPool;
    DescriptorAllocator globalDescriptorAllocator;

    DeletionQueue _mainDeletionQueue;
    VmaAllocator _allocator; // vma lib allocator

    // Depth resources
    VkPipeline _depthPipeline;
    VkPipelineLayout _depthPipelineLayout;
	VkDescriptorSet _depthImageDescriptors;
	VkDescriptorSetLayout _depthImageDescriptorLayout;

    // render pass
	VkFence _renderFence;
    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    VkCommandBuffer _secondaryDynamicCommandBuffer;
    VkCommandBuffer _secondaryStaticCommandBuffer;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;
	
    GPUMeshBuffers _main_mesh;

    float _z_near;
    float _z_far;


#ifdef SAVE_IMAGES
    void save_to_file(const std::string& filename, VkCommandBuffer cmd);
#endif
};
