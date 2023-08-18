import taichi as ti
import math 
import sys

ti.init(arch=ti.vulkan) 
#ti.init(debug=True)


#0-6                  0              1       2        3                    4                  5              6
#offset_structure = ["neighbors","corners","cross", "vertical_sobel", "horizontal_sobel", "left_emboss", "right_emboss"]

grid_length = 100
quadrilateral_length = 1.0 / grid_length
dt = 4e-3 / grid_length

print("dt is: ")
print(dt)
substeps = int(1 / 60 // dt)
g = ti.Vector([0, -9.8, 0])
print("dt is: ")
print(dt)
stiffness = 3e4  
damping_coefficient = 1e4   
exp_damp = 1
cloth_height = 1.0
use_local_mass = True   
global_mass = 100.0 / float(grid_length*grid_length)
local_mass = 10.0
offset_index = 5

sphere_radius = 0.3
sphere_center = ti.Vector.field(3, dtype=float, shape=(1, ))
sphere_center[0] = [0.0, 0.0, 0.0]

positions = ti.Vector.field(3, dtype=float, shape=(grid_length, grid_length))
velocities = ti.Vector.field(3, dtype=float, shape=(grid_length, grid_length))
init_positions = ti.Vector.field(3, dtype=float, shape=(grid_length,grid_length))

masses = ti.Vector.field(1,dtype=float, shape = (grid_length,grid_length))

num_triangles = (grid_length - 1) * (grid_length - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=grid_length * grid_length)
colors = ti.Vector.field(3, dtype=float, shape=grid_length * grid_length)


#scenes
horizontal_drop = 0
boundary_hang = 1
four_corners = 2
two_corners = 3


active_scene = 3

#Horizontal Drop
if sys.argv[1] == "0":
    active_scene = 0
    #offset_index = 0
#Boundary Hang
elif sys.argv[1] == "1":
    active_scene = 1
    #offset_index = 5
#Four Corners  
elif sys.argv[1] == "2":    
    active_scene = 2
    #offset_index = 4
#Two Corners
elif sys.argv[1] == "3":
    active_scene = 3
    #offset_index = 3
elif sys.argv[1] == "4":
    active_scene = 3
    #offset_index = 1




#active_scene = 3

scenes = {"horizontal_drop" : horizontal_drop, "boundary_hang" : boundary_hang, "four_corners": four_corners, "two_corners": two_corners}

print("global_mass is: ")
print(global_mass)



@ti.kernel
def init_scene(scene_type: int):
    shift_offsets = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

    for i, j in positions:
        if scene_type == horizontal_drop:
            positions[i, j] = [
                i * quadrilateral_length - 0.5 + shift_offsets[0], cloth_height,
                j * quadrilateral_length - 0.5 + shift_offsets[1]
            ]

        elif scene_type == boundary_hang:
            positions[i,j] = [
                i * quadrilateral_length - 0.5 + shift_offsets[0], j * quadrilateral_length - 0.5 + shift_offsets[1],
                0.0
            ]
            init_positions[i, j] = [
                i * quadrilateral_length - 0.5 + shift_offsets[0], j * quadrilateral_length - 0.5 + shift_offsets[1],
                0.0
            ]
        elif scene_type == two_corners:
            positions[i,j] = [
                i * quadrilateral_length - 0.5 + shift_offsets[0], j * quadrilateral_length - 0.5 + shift_offsets[1],
                0.0
            ]
            init_positions[i, j] = [
                i * quadrilateral_length - 0.5 + shift_offsets[0], j * quadrilateral_length - 0.5 + shift_offsets[1],
                0.0
            ]
        elif scene_type == four_corners:
            positions[i, j] = [
                i * quadrilateral_length - 0.5 + shift_offsets[0], cloth_height,
                j * quadrilateral_length - 0.5 + shift_offsets[1]
            ]
            init_positions[i, j] = [
                i * quadrilateral_length - 0.5 + shift_offsets[0], cloth_height,
                j * quadrilateral_length - 0.5 + shift_offsets[1]
            ]


        velocities[i, j] = [0, 0, 0]
        masses[i,j] = [global_mass]


@ti.kernel
def init_cloth_indices():
    for i, j in ti.ndrange(grid_length - 1, grid_length - 1):
        absolute_index = (i * (grid_length - 1)) + j
        #first triangle 
        indices[absolute_index * 6 + 0] = i * grid_length + j
        indices[absolute_index * 6 + 1] = (i + 1) * grid_length + j
        indices[absolute_index * 6 + 2] = i * grid_length + (j + 1)
        #second triangle
        indices[absolute_index * 6 + 3] = (i + 1) * grid_length + j + 1
        indices[absolute_index * 6 + 4] = i * grid_length + (j + 1)
        indices[absolute_index * 6 + 5] = (i + 1) * grid_length + j

    for i, j in ti.ndrange(grid_length, grid_length):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * grid_length + j] = (1.0, 1.0, 1.0)
        else:
            colors[i * grid_length + j] = (1, 0.334, 0.52)

init_cloth_indices()

connections = []
local_mass_distribution = []


def GenerateNeighborsOffsets():
    global local_mass_distribution
    for i in range (-1,2):
        for j in range(-1,2):
            if (i,j) != (0,0):
                connections.append(ti.Vector([i, j]))
    local_mass_distribution = [(0.0625 + 0.03125) * local_mass , (0.125+ 0.03125) * local_mass, (0.0625+ 0.03125) * local_mass, (0.125+ 0.03125) * local_mass, (0.125+ 0.03125) * local_mass, (0.0625+ 0.03125) * local_mass, (0.125+ 0.03125) * local_mass, (0.0625+ 0.03125) * local_mass]

def GenerateCornersOffsets():
    global local_mass_distribution
    for i in range (-1,2):
        for j in range(-1,2):
            if abs(i) + abs(j) == 2 and (i,j) != (0,0):
                connections.append(ti.Vector([i, j]))
    local_mass_distribution = [0.25 * local_mass, 0.25 * local_mass, 0.25 * local_mass, 0.25 * local_mass]


def GenerateCrossOffsets():
    global local_mass_distribution
    for i in range (-1,2):
        for j in range(-1,2):
            if abs(i) + abs(j) == 1 and (i,j) != (0,0):
                connections.append(ti.Vector([i, j]))
    local_mass_distribution = [0.25 * local_mass, 0.25 * local_mass, 0.25 * local_mass, 0.25 * local_mass]


def GenerateVerticalSobelOffsets():
    global local_mass_distribution
    for i in range (-1,2):
        for j in range(-1,2):
            if i != 0 and (i,j) != (0,0):
                connections.append(ti.Vector([i, j]))
    #local_mass_distribution = [-1.0 * local_mass, -2.0 * local_mass, -1.0 * local_mass, 1.0 * local_mass, 2.0 * local_mass, 1.0 * local_mass]
    local_mass_distribution = [1.0 * local_mass, 2.0 * local_mass, 1.0 * local_mass, 1.0 * local_mass, 2.0 * local_mass, 1.0 * local_mass]


def GenerateHorizontalSobelOffsets():
    #top sobel weights
    global local_mass_distribution
    for i in range (-1,2):
        for j in range(-1,2):
            if j != 0 and (i,j) != (0,0):
                connections.append(ti.Vector([i, j]))
    #local_mass_distribution = [-1.0 * local_mass,1.0 * local_mass, -2.0 * local_mass, 2.0 * local_mass, -1.0 * local_mass, 1.0 * local_mass]
    local_mass_distribution = [1.0 * local_mass,1.0 * local_mass, 2.0 * local_mass, 2.0 * local_mass, 1.0 * local_mass, 1.0 * local_mass]

def GenerateLeftEmbossOffsets():
    global local_mass_distribution
    for i in range (-1,2):
        for j in range(-1,2):
            if (i,j) != (-1,1) and (i,j) != (0,0) and (i,j) != (1,-1):
                connections.append(ti.Vector([i, j]))
    #local_mass_distribution = [(-2.0 + 0.16) * local_mass, (-1.0 + 0.16) * local_mass, (-1.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass]
    local_mass_distribution = [(2.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (2.0 + 0.16) * local_mass]

def GenerateRightEmbossOffsets():
    global local_mass_distribution
    for i in range (-1,2):
        for j in range(-1,2):
            if (i,j) != (-1,-1) and (i,j) != (0,0) and (i,j) != (1,1):
                connections.append(ti.Vector([i, j]))
    #local_mass_distribution = [(-1.0 + 0.16) * local_mass, (-2.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (-1.0 + 0.16) * local_mass, (2.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass]
    local_mass_distribution = [(1.0 + 0.16) * local_mass, (2.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass, (2.0 + 0.16) * local_mass, (1.0 + 0.16) * local_mass]


offset_structure = ["neighbors","corners","cross", "vertical_sobel", "horizontal_sobel", "left_emboss", "right_emboss"]

def populate_connections(offset_type):
    if offset_type == "neighbors":
        GenerateNeighborsOffsets()
    elif offset_type == "corners":
        GenerateCornersOffsets()
    elif offset_type == "cross":
        GenerateCrossOffsets()
    elif offset_type == "vertical_sobel":
        GenerateVerticalSobelOffsets()
    elif offset_type == "horizontal_sobel":
        GenerateHorizontalSobelOffsets()
    elif offset_type == "left_emboss":
        GenerateLeftEmbossOffsets()
    elif offset_type == "right_emboss":
        GenerateRightEmbossOffsets()


populate_connections(offset_structure[offset_index])

#print(local_mass_distribution)

@ti.kernel
def substep(scene_type: int):
    for i in ti.grouped(positions):
        if scene_type == boundary_hang and i[1] != grid_length-1:
            velocities[i] += g * dt
        elif scene_type == horizontal_drop:
            velocities[i] += g * dt
        elif scene_type == two_corners and ((i[0]!= 0 and i[1] != grid_length-1) or (i[0] != grid_length-1 and i[1] != grid_length-1)):
            velocities[i] += g * dt
        elif scene_type == four_corners and ((i[0] != 0 and i[1] != 0) or (i[0] != 0 and i[1] != grid_length-1) or (i[0] != grid_length-1 and i[1] != 0) or (i[0] != grid_length-1 and i[1] != grid_length-1)):
             velocities[i] += g * dt

    temp_mass = 5.5

    for i in ti.grouped(positions):
        force = ti.Vector([0.0, 0.0, 0.0])
        for k in ti.static(range(len(connections))):
            spring_offset = ti.Vector(connections[k])
            local_mass = local_mass_distribution[k]
            j = i + spring_offset
            if 0 <= j[0] < grid_length and 0 <= j[1] < grid_length:
                positions_ij = positions[i] - positions[j]
                velocities_ij = velocities[i] - velocities[j]
                positions_ij_normalized = positions_ij.normalized()
                distance_ij = positions_ij.norm()
                spring_rest_length = quadrilateral_length * (i - j).norm()
            
                if use_local_mass:
                    force += (-stiffness * positions_ij_normalized * (distance_ij / spring_rest_length - 1) / local_mass)
                else: 
                    force += (-stiffness * positions_ij_normalized * (distance_ij / spring_rest_length - 1)/ masses[j][0])
                
                force += -velocities_ij.dot(positions_ij_normalized) * positions_ij_normalized * damping_coefficient * quadrilateral_length

        velocities[i] += force * dt


    for i in ti.grouped(positions):
        velocities[i] *= ti.exp(-exp_damp * dt)

        if scene_type == horizontal_drop:
            center_vector = positions[i] - sphere_center[0]
            if center_vector.norm() <= sphere_radius:
                
                normal = center_vector.normalized()
                velocities[i] -= min(velocities[i].dot(normal), 0) * normal
        
        positions[i] += dt * velocities[i]

@ti.kernel
def update_cloth_mesh():
    for i, j in ti.ndrange(grid_length, grid_length):
        vertices[i * grid_length + j] = positions[i, j]

@ti.kernel
def constraint_top_row():
    for i in ti.grouped(positions):
        if i[1] == grid_length-1:
             positions[i] = init_positions[i]
             velocities[i] = [0,0,0]

@ti.kernel 
def constraint_two_corners():
    for i in ti.grouped(positions):
        if (i[0] == 0 and i[1] == grid_length-1) or (i[0] == grid_length-1 and i[1] == grid_length-1):
            positions[i] = init_positions[i]
            velocities[i] = [0,0,0]

@ti.kernel
def constraint_four_corners():
    for i in ti.grouped(positions):
        if (i[0] == 0 and i[1] == grid_length-1) or (i[0] == grid_length-1 and i[1] == grid_length-1) or (i[0] == 0 and i[1] == 0) or (i[0] == grid_length-1 and i[1] == 0):
            positions[i] = init_positions[i]
            velocities[i] = [0,0,0]

window = ti.ui.Window("Cloth Simulation", (1024, 1024),
                      vsync=False)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
demo = ti.ui.Scene()
fixed_cam = ti.ui.Camera()

curr_time = 0.0
init_scene(active_scene)

scene_time_limit = 2.0

while window.running:
    if curr_time > scene_time_limit:
        init_scene(active_scene)
        curr_time = 0

    for i in range(substeps):
        substep(active_scene)
        if active_scene == boundary_hang:
            constraint_top_row()
        elif active_scene == two_corners:
            constraint_two_corners()
        elif active_scene == four_corners:
            constraint_four_corners()
        curr_time += dt
    update_cloth_mesh()

    fixed_cam.position(0.0, 0.0, 4)
    fixed_cam.lookat(0.0, 0.0, 0)
    demo.set_camera(fixed_cam)

    demo.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    demo.ambient_light((0.5, 0.5, 0.5))    
    demo.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    if active_scene == horizontal_drop:
        demo.particles(sphere_center, radius=sphere_radius * 0.95, color=(0.28, 0.80, 0.68))
    canvas.scene(demo)
    window.show()