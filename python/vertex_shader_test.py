import math
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass

# -------------------------------------------------------------------
# Data Classes for Gate Types and Positions
# -------------------------------------------------------------------
@dataclass
class GateType:
    name: str
    is_double_gate: bool
    size_outer: float
    size_inner: float
    thickness: float

@dataclass
class GatePosition:
    x: float
    y: float
    z: float
    psi: float   # in radians
    type: GateType


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    proj: np.ndarray

# Example gate types (global list)
gate_types = [
    GateType(name="A2RL", is_double_gate=False, size_outer=2.7, size_inner=1.5, thickness=0.145),
    GateType(name="A2RL_DOUBLE", is_double_gate=True, size_outer=2.7, size_inner=1.5, thickness=0.145),
    GateType(name="MAVLAB", is_double_gate=False, size_outer=2.1, size_inner=1.5, thickness=0.05),
    GateType(name="MAVLAB_DOUBLE", is_double_gate=True, size_outer=2.1, size_inner=1.5, thickness=0.05)
]

# -------------------------------------------------------------------
# JSON Flightplan Parser (Port of the C++ JSON Parser)
# -------------------------------------------------------------------
def parse_flightplan(fp_file: str):
    """
    Reads a JSON flightplan from a file and returns a list of GatePosition.
    The expected JSON format contains a "waypoints" array with each waypoint
    specifying a type ("GATE", "START", etc.). For GATE waypoints, the field
    "gate_type" is used to determine the GateType.
    """
    # Resolve absolute path
    absolute_path = os.path.abspath(fp_file)
    try:
        with open(absolute_path, 'r') as f:
            flightplan_json = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not open flightplan file: {absolute_path}\n{e}")

    gates = []

    if "waypoints" in flightplan_json:
        for wp in flightplan_json["waypoints"]:
            wp_type = wp.get("type", "")
            if wp_type == "GATE":
                if "gate_type" not in wp:
                    raise RuntimeError("Gate type not specified in waypoint")
                gate_type_name = wp["gate_type"]
                # Find matching gate type.
                found = False
                for gt in gate_types:
                    if gt.name == gate_type_name:
                        selected_type = gt
                        found = True
                        break
                if not found:
                    raise RuntimeError(f"Gate type not found: {gate_type_name}")

                # Create GatePosition.
                # The C++ code multiplies yaw by PI/180 to convert degrees to radians.
                gate = GatePosition(
                    x = float(wp["x"]),
                    y = float(wp["y"]),
                    z = float(wp["z"]),
                    psi = float(wp["yaw"]) * math.pi / 180.0,
                    type = selected_type
                )
                gates.append(gate)
            elif wp_type == "START":
                # Could parse a start position if needed.
                pass
            else:
                raise RuntimeError(f"Unknown waypoint type: {wp_type}")
    else:
        raise RuntimeError("Flightplan does not specify any waypoints.")
    
    return gates

# -------------------------------------------------------------------
# Mesh Generation Function (Port of the C++ Code)
# -------------------------------------------------------------------
def init_mesh(flightplan):
    """
    Build vertices and triangle indices for a mesh representing gates.
    """
    vertices = []  # Each vertex will be a numpy array: [x, y, z]
    indices = []   # List of triangles as lists of three indices

    # Create a new flightplan list to handle double gates.
    flightplan_new = []
    for gatePos in flightplan:
        if gatePos.type.is_double_gate:
            # Create two gate positions: one at the original z, and one offset.
            gate1 = {
                'x': gatePos.x,
                'y': gatePos.y,
                'z': gatePos.z,
                'psi': gatePos.psi,
                'size_outer': gatePos.type.size_outer,
                'size_inner': gatePos.type.size_inner,
                'thickness': gatePos.type.thickness
            }
            gate2 = {
                'x': gatePos.x,
                'y': gatePos.y,
                'z': gatePos.z - gatePos.type.size_outer,
                'psi': gatePos.psi,
                'size_outer': gatePos.type.size_outer,
                'size_inner': gatePos.type.size_inner,
                'thickness': gatePos.type.thickness
            }
            flightplan_new.append(gate1)
            flightplan_new.append(gate2)
        else:
            gate = {
                'x': gatePos.x,
                'y': gatePos.y,
                'z': gatePos.z,
                'psi': gatePos.psi,
                'size_outer': gatePos.type.size_outer,
                'size_inner': gatePos.type.size_inner,
                'thickness': gatePos.type.thickness
            }
            flightplan_new.append(gate)

    # For each gate in the new flightplan, build the ring (two faces).
    for gatePos in flightplan_new:
        outer_size = gatePos['size_outer']
        inner_size = gatePos['size_inner']
        thickness  = gatePos['thickness']
        # Add pi/2 to psi as in your C++ code.
        psi = gatePos['psi'] + math.pi / 2.0

        # Precompute rotation (about Z axis)
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)

        # Two face levels: top and bottom.
        t_values = [thickness / 2.0, -thickness / 2.0]

        # Loop over both faces.
        for face, t in enumerate(t_values):
            # Define 8 local vertices: outer square (0-3) and inner square (4-7)
            localVerts = [
                np.array([-outer_size / 2.0, t,  outer_size / 2.0]),  # 0: outer top-left
                np.array([ outer_size / 2.0, t,  outer_size / 2.0]),  # 1: outer top-right
                np.array([ outer_size / 2.0, t, -outer_size / 2.0]),  # 2: outer bottom-right
                np.array([-outer_size / 2.0, t, -outer_size / 2.0]),  # 3: outer bottom-left
                np.array([-inner_size / 2.0, t,  inner_size / 2.0]),   # 4: inner top-left
                np.array([ inner_size / 2.0, t,  inner_size / 2.0]),   # 5: inner top-right
                np.array([ inner_size / 2.0, t, -inner_size / 2.0]),   # 6: inner bottom-right
                np.array([-inner_size / 2.0, t, -inner_size / 2.0])    # 7: inner bottom-left
            ]

            # Record starting index for these new vertices.
            startIndex = len(vertices)
            # Transform each vertex: rotate (around Z) then translate.
            for v in localVerts:
                localX, localY, localZ = v
                # Rotation about Z axis (affects x and y only)
                rotatedX = cos_psi * localX - sin_psi * localY
                rotatedY = sin_psi * localX + cos_psi * localY
                # Apply translation.
                worldX = rotatedX + gatePos['x']
                worldY = rotatedY + gatePos['y']
                worldZ = localZ + gatePos['z']  # z is simply translated
                vertices.append(np.array([worldX, worldY, worldZ]))

            # Define triangles from local vertex indices.
            faceTriangles = [
                [0, 1, 5], [0, 5, 4],  # top portion
                [1, 2, 6], [1, 6, 5],  # right portion
                [2, 3, 7], [2, 7, 6],  # bottom portion
                [3, 0, 4], [3, 4, 7]   # left portion
            ]
            # For the bottom face, reverse the winding order.
            if face == 1:
                for tri in faceTriangles:
                    tri[0], tri[2] = tri[2], tri[0]

            # Append triangles with proper index offset.
            for tri in faceTriangles:
                indices.append([startIndex + tri[0], startIndex + tri[1], startIndex + tri[2]])

    return vertices, indices

# -------------------------------------------------------------------
# Matrix and Transformation Functions
# -------------------------------------------------------------------
def perspective_matrix(fov, aspect, near, far):
    """
    Build a perspective projection matrix.
    fov: field of view in radians.
    """
    f = 1.0 / math.tan(fov / 2.0)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M

def intrinsics_to_vk_proj(intrinsics, w, h, znear, zfar):
    """
    Converts camera intrinsics to a Vulkan-style projection matrix.
    
    Parameters:
    - intrinsics: 3x3 NumPy array representing camera intrinsics.
    - w: Image width.
    - h: Image height.
    - znear: Near clipping plane.
    - zfar: Far clipping plane.
    
    Returns:
    - 4x4 NumPy array representing the Vulkan projection matrix.
    """
    # Extract intrinsic parameters
    alpha = intrinsics[0, 0]
    beta = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # Build an OpenGL-style projection matrix
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = alpha / cx
    P[1, 1] = beta / cy
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    P[3, 2] = -1.0

    # Vulkan clip correction matrix
    clip = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Multiply the OpenGL projection matrix by the Vulkan clip correction matrix
    return np.dot(clip, P)


def lookAt(eye, target, up):
    """
    Create a view matrix using the eye position, target, and up vector.
    """
    f = target - eye
    f = f / np.linalg.norm(f)
    up_norm = up / np.linalg.norm(up)
    s = np.cross(f, up_norm)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    M = np.identity(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M

def simulated_vertex_shader(v, VP, near, far, apply_linearity=False):
    """
    Simulates the vertex shader transformation with optional depth linearization.
    
    Parameters:
      v: A 3-element array representing the world-space vertex.
      VP: The view-projection matrix.
      near: The near plane distance.
      far: The far plane distance.
      apply_linearity: Boolean flag. If True, adjust gl_Position.z so that after perspective division,
                       depth becomes linear. If False, return the raw clip-space coordinate.
    
    Returns:
      A 4-element numpy array representing the clip-space position.
    """
    v_h = np.array([v[0], v[1], v[2], 1.0])
    pos = VP @ v_h  # Clip-space position
    
    if apply_linearity:
        # For Vulkan, adjust clip-space z to linearize depth after perspective divide
        linear_depth = pos[2] / far  # pos[2]/far gives (d - near)/(far - near)
        pos[2] = linear_depth * pos[3]  # New z_clip = linear_depth * w_clip
    
    return pos

# -------------------------------------------------------------------
# Mesh Plotting Utility
# -------------------------------------------------------------------
def plot_mesh(vertices, indices, camera=None, title="Mesh"):
    """
    Plot a mesh using matplotlib 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Build triangle faces from indices.
    triangles = []
    for tri in indices:
        triangle = [vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]]
        triangles.append(triangle)
    poly3d = Poly3DCollection(triangles, edgecolors='k', facecolors='cyan', alpha=0.5)
    ax.add_collection3d(poly3d)

    if camera is not None:
        plot_camera(ax, camera)

    # Auto-scale to the mesh size.
    vertices_array = np.array(vertices)
    max_range = (vertices_array.max(axis=0) - vertices_array.min(axis=0)).max() / 2.0
    mid = vertices_array.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_title(title)
    # Set labels to reflect NED coordinate system
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Invert the Z-axis to make 'Down' point downwards
    ax.invert_zaxis()
    
    # Invert the Y-axis to switch to a right-handed coordinate system
    ax.invert_yaxis()
    
    plt.show()

def plot_clip_space_mesh(vertices_list, indices, titles):
    """
    Plot multiple meshes in two views:
      - Top row: 3D clip space (as before)
      - Bottom row: 2D rendered view (after perspective division)
    """
    import matplotlib.collections as mc

    num_meshes = len(vertices_list)
    fig = plt.figure(figsize=(5 * num_meshes, 10))
    axes_top = []
    axes_bottom = []

    for i, vertices in enumerate(vertices_list):
        # Top view: 3D clip space plot.
        ax_top = fig.add_subplot(2, num_meshes, i + 1, projection='3d')
        axes_top.append(ax_top)
        # Build triangle faces from indices with perspective divide.
        triangles = []
        for tri in indices:
            triangle = []
            for idx in tri:
                v = vertices[idx]
                if v[3] != 0:
                    v_ndc = v[:3] / v[3]
                else:
                    v_ndc = v[:3]
                triangle.append(v_ndc)
            triangles.append(triangle)

        poly3d = Poly3DCollection(triangles, edgecolors='k', facecolors='cyan', alpha=0.5)
        ax_top.add_collection3d(poly3d)

        # Plot clip space bounding box in red.
        # Near plane
        ax_top.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [0, 0, 0, 0, 0], color='red')
        # Far plane
        ax_top.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], [1, 1, 1, 1, 1], color='red')
        # Vertical edges
        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            ax_top.plot([x, x], [y, y], [0, 1], color='red')

        ax_top.set_xlim(-1.0, 1.0)
        ax_top.set_ylim(-1.0, 1.0)
        ax_top.set_zlim(0, 1.0)
        ax_top.set_box_aspect([1, 1, 1])
        ax_top.set_title(titles[i])
        ax_top.set_xlabel('X')
        ax_top.set_ylabel('Y')
        ax_top.set_zlabel('Z')
        ax_top.invert_zaxis()
        ax_top.invert_yaxis()

        # Bottom view: 2D rendered image view.
        ax_bottom = fig.add_subplot(2, num_meshes, num_meshes + i + 1)
        axes_bottom.append(ax_bottom)
        # Build 2D triangles using only x and y after perspective division.
        triangles_2d = []
        for tri in indices:
            tri_2d = []
            for idx in tri:
                v = vertices[idx]
                if v[3] != 0:
                    v_ndc = v[:3] / v[3]
                else:
                    v_ndc = v[:3]
                # Use x and y; optionally flip y to get a rendered image look.
                tri_2d.append((v_ndc[0], v_ndc[1]))
            triangles_2d.append(tri_2d)

        # Create a PolyCollection for the 2D triangles.
        poly2d = mc.PolyCollection(triangles_2d, edgecolors='k', facecolors='cyan', alpha=0.5)
        ax_bottom.add_collection(poly2d)

        # Plot the clip space bounding box in 2D.
        # Near plane (z=0 in clip space becomes the image border after division)
        bbox = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        bbox.append(bbox[0])
        xs, ys = zip(*bbox)
        ax_bottom.plot(xs, ys, color='red')

        ax_bottom.set_xlim(-1.0, 1.0)
        ax_bottom.set_ylim(-1.0, 1.0)
        ax_bottom.set_aspect('equal', 'box')
        ax_bottom.set_title(f"{titles[i]} - Rendered")
        ax_bottom.set_xlabel('X')
        ax_bottom.set_ylabel('Y')
        # Invert y-axis if desired to mimic image coordinates:
        ax_bottom.invert_yaxis()

    plt.tight_layout()
    plt.show()

def plot_camera(ax, camera):
    # plot camera position and vision cone
    ax.scatter(camera.eye[0], camera.eye[1], camera.eye[2], c='r', marker='o')

    forward = camera.target - camera.eye
    forward = forward / np.linalg.norm(forward)
    up = camera.up / np.linalg.norm(camera.up)
    right = np.cross(forward, up)
    # Compute the inverse projection matrix to transform from clip to view space.
    inv_proj = np.linalg.inv(camera.proj)

    # Define the eight corners in Normalized Device Coordinates (NDC).
    # For Vulkan-style NDC: x, y in [-1, 1] and z in [0, 1]
    ndc_points = {
        'near': [
            np.array([-1, -1, 0, 1]),
            np.array([ 1, -1, 0, 1]),
            np.array([ 1,  1, 0, 1]),
            np.array([-1,  1, 0, 1])
        ],
        'far': [
            np.array([-1, -1, 1, 1]),
            np.array([ 1, -1, 1, 1]),
            np.array([ 1,  1, 1, 1]),
            np.array([-1,  1, 1, 1])
        ]
    }

    # Function to transform a point from clip space to view space.
    def clip_to_view(ndc_point):
        view_point = inv_proj @ ndc_point
        view_point /= view_point[3]
        return view_point[:3]

    # Transform the near and far plane corners to view space.
    near_view = [clip_to_view(p) for p in ndc_points['near']]
    far_view = [clip_to_view(p) for p in ndc_points['far']]

    # Construct the camera's world transformation matrix from the eye, forward, and up.
    f = forward
    u = up
    r = right

    # Build the camera-to-world (view inverse) matrix.
    cam_world = np.eye(4)
    cam_world[:3, 0] = r      # right
    cam_world[:3, 1] = u      # up
    cam_world[:3, 2] = -f     # -forward
    cam_world[:3, 3] = camera.eye  # translation

    # Function to transform a point from view space to world space.
    def view_to_world(view_point):
        view_homog = np.array([view_point[0], view_point[1], view_point[2], 1.0])
        world_homog = cam_world @ view_homog
        return world_homog[:3]

    # Transform the near and far plane corners to world space.
    near_world = [view_to_world(p) for p in near_view]
    far_world = [view_to_world(p) for p in far_view]

    # Plot the near plane (red lines).
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([near_world[i][0], near_world[j][0]],
                [near_world[i][1], near_world[j][1]],
                [near_world[i][2], near_world[j][2]], 'r-')

    # Plot the far plane (red lines).
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([far_world[i][0], far_world[j][0]],
                [far_world[i][1], far_world[j][1]],
                [far_world[i][2], far_world[j][2]], 'r-')

    # Connect the near and far plane corners.
    for i in range(4):
        ax.plot([near_world[i][0], far_world[i][0]],
                [near_world[i][1], far_world[i][1]],
                [near_world[i][2], far_world[i][2]], 'r-')

def plot_depth_nonlinearity(z_far, z_near):
    """
    Plot the depth nonlinearity function.
    """
    z = np.linspace(z_near, z_far, 100)
    nonlinearity = 1.0 / (1.0 - z_near / z)
    plt.plot(z, nonlinearity)
    plt.title("Depth Nonlinearity Function")
    plt.xlabel("Z")
    plt.ylabel("Nonlinearity")
    plt.show()

# -------------------------------------------------------------------
# Main Script
# -------------------------------------------------------------------
if __name__ == "__main__":
    flightplan_file = "flightplans/dhl_3dtrack_full.json"
    flightplan = parse_flightplan(flightplan_file)


    # Generate the mesh.
    vertices, indices = init_mesh(flightplan)

    K = np.array([[165.0, 0.0, 190.0],
                  [0.0, 165.0, 170.0],
                  [0.0, 0.0, 1.0]])
    w = 360
    h = 360
    # Setup the projection and view matrices.
    near = 0.1
    far = 30.0
    
    proj = intrinsics_to_vk_proj(K, w, h, near, far)

    eye = np.array([0, 0, -2.5])       # Camera position
    target = np.array([10.0, 0, 5.0])      # Look-at target
    up = np.array([0, 0, -1.0])          # Up vector
    view = lookAt(eye, target, up)

    camera = Camera(eye=eye, target=target, proj=proj, up=up)

    # Plot the original mesh in world space.
    plot_mesh(vertices, indices, camera, title="World Space Mesh")

    # Combine view and projection (assuming model is identity).
    VP = proj @ view

    # Transform all vertices into clip space.
    transformed_vertices_linear = [simulated_vertex_shader(v, VP, near, far, apply_linearity=True) for v in vertices]
    transformed_vertices = [simulated_vertex_shader(v, VP, near, far, apply_linearity=False) for v in vertices]

    # Plot the transformed (clip space) mesh.
    plot_clip_space_mesh([transformed_vertices, transformed_vertices_linear], indices, ["Clip Space Mesh", "Clip Space Mesh (Linear Depth)"])
