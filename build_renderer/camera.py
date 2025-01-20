import taichi as ti
import taichi.math as tm
import numpy as np

from .ray import Ray


@ti.data_oriented
class Camera:

    def __init__(self, width: int = 128, height: int = 128) -> None:

        # Camera pixel width and height are fixed
        self.width = width
        self.height = height

        # Camera parameters that can be modified are stored as fields
        self.eye = ti.Vector.field(n=3, shape=(), dtype=float)
        self.at = ti.Vector.field(n=3, shape=(), dtype=float)
        self.up = ti.Vector.field(n=3, shape=(), dtype=float)
        self.fov = ti.field(shape=(), dtype=float)

        self.x = ti.Vector.field(n=3, shape=(), dtype=float)
        self.y = ti.Vector.field(n=3, shape=(), dtype=float)
        self.z = ti.Vector.field(n=3, shape=(), dtype=float)

        self.camera_to_world = ti.Matrix.field(n=4, m=4, shape=(), dtype=float)

        # Initialize with some default params
        self.set_camera_parameters(
            eye=tm.vec3([0, 0, 5]),
            at=tm.vec3([0, 0, 0]),
            up=tm.vec3([0, 1, 0]),
            fov=60.
            )


    def set_camera_parameters(
        self, 
        eye: tm.vec3 = None, 
        at: tm.vec3 = None, 
        up: tm.vec3 = None, 
        fov: float = None
        ) -> None:

        if eye: self.eye[None] = eye
        if at: self.at[None] = at
        if up: self.up[None] = up
        if fov: self.fov[None] = fov
        self.compute_matrix()


    @ti.kernel
    def compute_matrix(self):

        '''
        TODO: Compute Camera to World Matrix

        self.camera_to_world[None] = tm.mat4(<You Matrix>)

        '''
        zc = tm.normalize((self.at[None] - self.eye[None]))
        xc = tm.normalize(tm.cross(self.up[None], zc))
        yc = tm.normalize(tm.cross(zc, xc))

        cam_coord = ti.Matrix.cols([xc, yc, zc, self.eye[None]])
        cam_to_world = ti.Matrix.zero(ti.f32, 4, 4)
        for i, j in ti.ndrange(3,4):
            cam_to_world[i,j] = cam_coord[i, j]
        cam_to_world[3,3] = 1.0

        self.z[None] = zc
        self.x[None] = xc
        self.y[None] = yc
        self.camera_to_world[None] = tm.mat4(cam_to_world)

    @ti.func
    def generate_ray(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> Ray:
        
        '''
        TODO: Generate Ray

        - generate ndc coords
        - generate camera coods from NDC coords
        - generate a ray
            - ray = Ray()
        - set the ray direction and origin
            - ray.origin = ...
            - ray.direction = ...
        - Ignore jittering for now
        - return ray
        '''
        # placeholder
        ndc_coord = self.generate_ndc_coords(pixel_x,pixel_y)
        cam_coord = self.generate_camera_coords(ndc_coord)
        ray = Ray()
        ray.origin = self.eye[None]
        self.compute_matrix()
        dir = self.camera_to_world[None]@cam_coord
        ray.direction = dir.xyz
        return ray


    @ti.func
    def generate_ndc_coords(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> tm.vec2:
        
        '''
        TODO: Genrate NDC coods

        - Ignore jittering for now
        
        return tm.vec2([ndc_x, ndc_y])

        '''

        # placeholder
        ndc_x, ndc_y = 0.0, 0.0
        ndc_x = (pixel_x+0.5)/(self.width/2)-1
        ndc_y = (pixel_y+0.5)/(self.height/2)-1
        return tm.vec2([ndc_x, ndc_y])

    @ti.func
    def generate_camera_coords(self, ndc_coords: tm.vec2) -> tm.vec4:
        
        '''
        TODO: Generate Camera coordinates
        - compute camera_x, camera_y, camera_z
        
        - return tm.vec4([camera_x, camera_y, camera_z, 0.0])
        '''

        # palceholder
        cam_x = 0.0
        cam_y = 0.0
        cam_z = 0.0

        aspect_ratio = self.width/self.height
        cam_x = aspect_ratio*tm.tan(tm.radians(self.fov[None])/2)*ndc_coords[0]
        cam_y = tm.tan(tm.radians(self.fov[None])/2)*ndc_coords[1]
        cam_z = 1

        return tm.vec4([cam_x, cam_y, cam_z, 0.0])