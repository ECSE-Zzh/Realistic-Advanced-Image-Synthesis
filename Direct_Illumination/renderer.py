from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .scene_data import SceneData
from .camera import Camera
from .ray import Ray, HitData
from .sampler import UniformSampler, BRDF, MicrofacetBRDF
from .materials import Material


@ti.data_oriented
class A1Renderer:

    # Enumerate the different shading modes
    class ShadeMode(IntEnum):
        HIT = 1
        TRIANGLE_ID = 2
        DISTANCE = 3
        BARYCENTRIC = 4
        NORMAL = 5
        MATERIAL_ID = 6

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.scene_data = scene_data
        self.iter_counter = ti.field(dtype=float, shape=())

        self.shade_mode = ti.field(shape=(), dtype=int)
        self.set_shade_hit()

        # Distance at which the distance shader saturates
        self.max_distance = 10.

        # Numbers used to generate colors for integer index values
        self.r = 3.14159265
        self.b = 2.71828182
        self.g = 6.62607015


    def set_shade_hit(self):          self.shade_mode[None] = self.ShadeMode.HIT
    def set_shade_triangle_ID(self):  self.shade_mode[None] = self.ShadeMode.TRIANGLE_ID
    def set_shade_distance(self):     self.shade_mode[None] = self.ShadeMode.DISTANCE
    def set_shade_barycentrics(self): self.shade_mode[None] = self.ShadeMode.BARYCENTRIC
    def set_shade_normal(self):       self.shade_mode[None] = self.ShadeMode.NORMAL
    def set_shade_material_ID(self):  self.shade_mode[None] = self.ShadeMode.MATERIAL_ID


    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1 
        for x, y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x, y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / (self.iter_counter[None])  # Average over iterations and samples

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)

    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        color = tm.vec3(0)
        if   self.shade_mode[None] == int(self.ShadeMode.HIT):         color = self.shade_hit(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.TRIANGLE_ID): color = self.shade_triangle_id(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.DISTANCE):    color = self.shade_distance(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.BARYCENTRIC): color = self.shade_barycentric(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.NORMAL):      color = self.shade_normal(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.MATERIAL_ID): color = self.shade_material_id(hit_data)
        return color
       

    @ti.func
    def shade_hit(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            if not hit_data.is_backfacing:
                color = tm.vec3(1)
            else: 
                color = tm.vec3([0.5,0,0])
        return color


    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1 # Add 1 so that ID 0 is not black
            r = triangle_id*self.r % 1
            g = triangle_id*self.g % 1
            b = triangle_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color


    @ti.func
    def shade_distance(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            d = tm.clamp(hit_data.distance / self.max_distance, 0,1)
            color = tm.vec3(d)
        return color


    @ti.func
    def shade_barycentric(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            u = hit_data.barycentric_coords[0]
            v = hit_data.barycentric_coords[1]
            w = 1. - u - v
            color = tm.vec3(u,v,w)
        return color


    @ti.func
    def shade_normal(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            normal = hit_data.normal
            color = (normal + 1.) / 2.  # Scale to range [0,1]
        return color


    @ti.func
    def shade_material_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            material_id = hit_data.material_id + 1 # Add 1 so that ID 0 is not black
            r = material_id*self.r % 1
            g = material_id*self.g % 1
            b = material_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color

@ti.data_oriented
class A2Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        MICROFACET = 3

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()


    def set_sample_uniform(self):    self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_brdf(self):       self.sample_mode[None] = self.SampleMode.BRDF
    def set_sample_microfacet(self): self.sample_mode[None] = self.SampleMode.MICROFACET


    @ti.kernel
    def render(self):
        #TODO: Change the naive renderer to do progressive rendering
        '''
        - call generate_ray with jitter = True
        - progressively accumulate the pixel values in each canvas [x, y] position
        '''
        color = tm.vec3(0.)
        self.iter_counter[None] += 1 
        for x, y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x, y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / (self.iter_counter[None])  # Average over iterations and samples

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        Lo = tm.vec3(0.)
        Le = tm.vec3(0.)
        visibility = 0
        '''
        You can change the structure of the shade ray function however you want as there will be computations that are the same for all 3 methods
        You can have your branching logic anywhere in the code
        '''
        # TODO: Implement Uniform Sampling
        hit_data = self.scene_data.ray_intersector.query_ray(ray) # background
        material = self.scene_data.material_library.materials[hit_data.material_id]
        n = hit_data.normal
        if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
            if hit_data.is_hit:
                w_i = UniformSampler.sample_direction() 
                x = ray.origin + (hit_data.distance*ray.direction) + (self.RAY_OFFSET*n) # shadow ray origin
                shadow_ray = Ray(x, w_i)

                # Le(x, w_i)
                Le = self.scene_data.environment.query_ray(shadow_ray)

                # V(x, w_i): visibility
                if self.scene_data.ray_intersector.query_ray(shadow_ray).is_hit:
                    visibility = 0
                else:
                    visibility = 1

                #fr
                w_o = -ray.direction
                fr = BRDF.evaluate_brdf(material, w_o, w_i, n)
                if material.Ns > 100:
                    color = tm.vec3(0.)
                else:
                #max
                    max = max(tm.dot(n, w_i), 0.)

                    # P_uniform
                    p_uni = UniformSampler.evaluate_probability()

                    Lo = (Le * visibility * fr * max)/p_uni

                    color = Lo
            else:
                color = self.scene_data.environment.query_ray(ray)
        # TODO: Implement BRDF Sampling
        elif self.sample_mode[None] == int(self.SampleMode.BRDF):
            if hit_data.is_hit:
                w_o = -ray.direction
                w_i = BRDF.sample_direction(material, w_o, n) 
                x = ray.origin + (hit_data.distance*ray.direction) + (self.RAY_OFFSET*n) # shadow ray origin
                shadow_ray = Ray(x, w_i)

                # Le(x, w_i)
                Le = self.scene_data.environment.query_ray(shadow_ray)

                # V(x, w_i): visibility
                if self.scene_data.ray_intersector.query_ray(shadow_ray).is_hit:
                    visibility = 0
                else:
                    visibility = 1

                #fr
                fr_factor = BRDF.evaluate_brdf_factor(material, w_o, w_i, n)

                Lo = Le * visibility * fr_factor

                color = Lo
            else:
                color = self.scene_data.environment.query_ray(ray)
                           
        # TODO: 546 Deliverable Only
        # Implement Microfacet BRDF Sampling
        elif self.sample_mode[None] == int(self.SampleMode.MICROFACET):
            pass       

        return color