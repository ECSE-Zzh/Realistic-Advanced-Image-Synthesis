import taichi as ti
ti.init()
# Create a Taichi field (global memory) for a 2x2 matrix with dtype specified
cam_to_world = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())  # Matrix field with 2x2 shape and float type
        
@ti.kernel
def hello():
    xc = ti.Vector([3,4,5])
    yc = ti.Vector([6,7,8])
    zc = ti.Vector([0,1,2])
    eye = ti.Vector([9,9,0])
    cam_coord = ti.Matrix.cols([xc, yc, zc, eye])
    cam_to_world[None] = ti.Matrix.zero(ti.f32, 4, 4)
    for i, j in ti.ndrange(3,4):
        cam_to_world[None][i,j] = cam_coord[i, j]
        print(cam_to_world[None][i,j])
    cam_to_world[None][3,3] = 1.0
    # print(cam_to_world)
    # result[None] = ti.Matrix.rows([x,y,m])
# Call the kernel
hello()
# Print the result
print(cam_to_world)