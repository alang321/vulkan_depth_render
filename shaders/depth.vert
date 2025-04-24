#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Vertex structure (must match your app's vertex format)
struct Vertex {
    vec3 position; // World-space position
    float padding;
};

// Buffer reference for vertex data via device address
layout(buffer_reference, scalar) buffer VertexBuffer {
    Vertex vertices[];
};

// Push constants with view-projection matrix, near/far planes, and vertex buffer address
layout(push_constant) uniform constants {
    mat4 view_proj_matrix; // Projection * View matrix
    float far_plane;
    VertexBuffer vertexBuffer; 
} pc;

void main() {
    // Fetch world-space vertex directly from the buffer
    Vertex v = pc.vertexBuffer.vertices[gl_VertexIndex];

    // Transform world-space position to clip space
    vec4 pos = pc.view_proj_matrix * vec4(v.position, 1.0);

    // Adjust depth for linearization (Vulkan-style)
    float linear_depth = pos.z / pc.far_plane;  // Equivalent to (d - near) / (far - near)
    pos.z = linear_depth * pos.w;  // New clip-space Z

    gl_Position = pos;
}
