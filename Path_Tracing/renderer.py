from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .scene_data import SceneData
from .camera import Camera
from .ray import Ray, HitData
from .sampler import UniformSampler, BRDF, MicrofacetBRDF, MeshLightSampler
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
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] = color


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
        for x,y in ti.ndrange(self.width, self.height):
            #TODO1: Change the naive renderer to do progressive rendering
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

        '''
        You can change the structure of the shade ray function however you want as there will be computations that are the same for all 3 methods
        You can have your branching logic anywhere in the code
        '''

        # TODO1: Implement Uniform Sampling
        if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
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
                        # max
                        max = max(tm.dot(n, w_i), 0.)

                        # P_uniform
                        p_uni = UniformSampler.evaluate_probability()

                        Lo = (Le * visibility * fr * max)/p_uni

                        color = Lo
                else:
                    color = self.scene_data.environment.query_ray(ray)
        
        # TODO1: Implement BRDF Sampling
        elif self.sample_mode[None] == int(self.SampleMode.BRDF):
            if hit_data.is_hit:
                if material.Ke.norm() > 0:  # if hit emmisive material
                    color = material.Ke
                else:                
                    w_o = -ray.direction
                    w_i = BRDF.sample_direction(material, w_o, n) 
                    x = ray.origin + (hit_data.distance*ray.direction) + (self.RAY_OFFSET*n) # shadow ray origin
                    shadow_ray = Ray(x, w_i)

                    # Le(x, w_i) and V(x, w_i): visibility
                    shadow_hit_data = self.scene_data.ray_intersector.query_ray(shadow_ray)
                    if shadow_hit_data.is_hit:
                        material_shadow_hit = self.scene_data.material_library.materials[shadow_hit_data.material_id]
                        if material_shadow_hit.Ke.norm() > 0: # emissive material
                            Le = material_shadow_hit.Ke
                            visibility = 1
                        else:
                            Le = tm.vec3(0.)
                            visibility = 0
                    else: # not hit
                        Le = self.scene_data.environment.query_ray(shadow_ray)
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


@ti.data_oriented
class EnvISRenderer:
    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        ENVMAP = 2
    
    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        
        self.camera = Camera(width=width, height=height)
        self.count_map = ti.field(dtype=float, shape=(width, height))
        
        self.background = ti.Vector.field(n=3, dtype=float, shape=(width, height))

        self.scene_data = scene_data
        self.sample_mode = ti.field(shape=(), dtype=int)

        self.set_sample_uniform()


    def set_sample_uniform(self): 
        self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_envmap(self):    
        self.sample_mode[None] = self.SampleMode.ENVMAP

    @ti.func
    def render_background(self, x: int, y: int) -> tm.vec3:
        uv_x, uv_y = float(x)/self.width, float(y)/self.height
        uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
        
        background = self.scene_data.environment.image[int(uv_x), int(uv_y)]
            

        return background


    @ti.kernel
    def render_background(self):
        for x,y in ti.ndrange(self.width, self.height):
            uv_x, uv_y = float(x)/float(self.width), float(y)/float(self.height)
            uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
            color = self.scene_data.environment.image[int(uv_x), int(uv_y)]

            self.background[x,y] = color

    @ti.kernel
    def sample_env(self, samples: int):
        for _ in ti.ndrange(samples):
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                x = int(ti.random() * self.width)
                y = int(ti.random() * self.height)


                self.count_map[x,y] += 1.0
                
            elif self.sample_mode[None] == int(self.SampleMode.ENVMAP):
                sampled_phi_theta = self.scene_data.environment.importance_sample_envmap()
                x = sampled_phi_theta[0] * self.width
                y = sampled_phi_theta[1] * self.height

                self.count_map[int(x), int(y)] += 1.0
    
    @ti.kernel
    def reset(self):
        self.count_map.fill(0.)


@ti.data_oriented
class A3Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        LIGHT = 3
        MIS = 4

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
        self.canvas_postprocessed = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data
        self.a2_renderer = A2Renderer(width=self.width, height=self.height, scene_data=self.scene_data)
        
        self.mis_plight = ti.field(dtype=float, shape=())
        self.mis_pbrdf = ti.field(dtype=float, shape=())

        self.mis_plight[None] = 0.5
        self.mis_pbrdf[None] = 0.5

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()

        self.meshLightSampler = MeshLightSampler(self.scene_data.geometry, self.scene_data.material_library)


    def set_sample_uniform(self): 
        self.sample_mode[None] = self.SampleMode.UNIFORM
        self.a2_renderer.set_sample_uniform()
    def set_sample_brdf(self):    
        self.sample_mode[None] = self.SampleMode.BRDF
        self.a2_renderer.set_sample_brdf()
    def set_sample_light(self):    self.sample_mode[None] = self.SampleMode.LIGHT
    def set_sample_mis(self):    self.sample_mode[None] = self.SampleMode.MIS


    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] += (color - self.canvas[x,y])/self.iter_counter[None]
    
    @ti.kernel
    def postprocess(self):
        for x,y in ti.ndrange(self.width, self.height):
            self.canvas_postprocessed[x, y] = tm.pow(self.canvas[x, y], tm.vec3(1.0 / 2.2))
            self.canvas_postprocessed[x, y] = tm.clamp(self.canvas_postprocessed[x, y], xmin=0.0, xmax=1.0)

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        Lo = tm.vec3(0.)
        Le = tm.vec3(0.)
        visibility = 0

        hit_data = self.scene_data.ray_intersector.query_ray(ray) # background
        material = self.scene_data.material_library.materials[hit_data.material_id]
        n = hit_data.normal

        if self.sample_mode[None] == int(self.SampleMode.UNIFORM) or self.sample_mode[None] == int(self.SampleMode.BRDF):
            # Uniform or BRDF just calls the A2 renderer
            # TODO1: Implement Mesh Light support for your A2 renderer
            color = self.a2_renderer.shade_ray(ray)
        else:
            if self.sample_mode[None] == int(self.SampleMode.LIGHT):
                # TODO1: Implement Light Importance Sampling
                if hit_data.is_hit:
                    if material.Ke.norm() > 0:  # if hit emmisive material
                        color = material.Ke
                    else:                
                        w_o = -ray.direction

                        # Le(x, w_i) and V(x, w_i): visibility
                        x = ray.origin + (hit_data.distance*ray.direction) + (self.RAY_OFFSET*n) # shadow ray origin
                        w_i, sampled_light_triangle = self.meshLightSampler.sample_mesh_lights(x) 

                        shadow_ray = Ray(x, w_i)
                        shadow_hit_data = self.scene_data.ray_intersector.query_ray(shadow_ray)
                        ny = shadow_hit_data.normal

                        if shadow_hit_data.is_hit and shadow_hit_data.triangle_id == sampled_light_triangle:
                           
                            material_shadow_hit = self.scene_data.material_library.materials[shadow_hit_data.material_id]
                            if material_shadow_hit.Ke.norm() > 0: # emissive material
                                Le = material_shadow_hit.Ke
                                visibility = 1
                            else:
                                Le = tm.vec3(0.)
                                visibility = 0
                        else: # not hit
                            Le = self.scene_data.environment.query_ray(shadow_ray)
                            visibility = 1

                        #fr
                        fr = BRDF.evaluate_brdf(material, w_o, w_i, n)

                        # max
                        max1 = max(tm.dot(n, w_i), 0.)
                        max2 = max(tm.dot(ny, -w_i), 0.)

                        # p_light
                        p_light = self.meshLightSampler.evaluate_probability()

                        # ||x-y||^2
                        shadow_hit_data_distance = shadow_hit_data.distance**2
                        if shadow_hit_data_distance == 0:
                            color = tm.vec3(0.)
                        else:
                            Lo = (Le*visibility*fr*max1*max2) / (p_light*shadow_hit_data_distance)
                            color = Lo
                else:
                    color = self.scene_data.environment.query_ray(ray)    
                # pass        
            if self.sample_mode[None] == int(self.SampleMode.MIS):
                # TODO1: Implement MIS
                if hit_data.is_hit:
                    if material.Ke.norm()> 0:
                        color = material.Ke
                    else: 
                        rnd = ti.random()
                        w_i = tm.vec3(0.)
                        w_o = -ray.direction
                        x = ray.origin + (hit_data.distance*ray.direction) + (self.RAY_OFFSET*n) # shadow ray origin
                        sampled_light_triangle = 0
                        if rnd < self.mis_pbrdf[None]:
                            w_i = BRDF.sample_direction(material, w_o, n)
                        else:
                            w_i, sampled_light_triangle = self.meshLightSampler.sample_mesh_lights(x)  
                        p_light = self.meshLightSampler.evaluate_probability()
                        p_brdf = BRDF.evaluate_probability(material, w_o, w_i, n)
                        p_mis = self.mis_pbrdf[None]*p_brdf + self.mis_plight[None]*p_light

                        # Le(x, w_i) and V(x, w_i): visibility
                        shadow_ray = Ray(x, w_i)
                        shadow_hit_data = self.scene_data.ray_intersector.query_ray(shadow_ray)
                        ny = shadow_hit_data.normal

                        if shadow_hit_data.is_hit:
                            material_shadow_hit = self.scene_data.material_library.materials[shadow_hit_data.material_id]
                            if material_shadow_hit.Ke.norm() > 0: # emissive material
                                Le = material_shadow_hit.Ke
                                visibility = 1
                            else:
                                Le = tm.vec3(0.)
                                visibility = 0
                        else: # not hit
                            Le = self.scene_data.environment.query_ray(shadow_ray)
                            visibility = 1
                        
                        #fr
                        fr = BRDF.evaluate_brdf(material, w_o, w_i, n)
                        
                        # max
                        max1 = max(tm.dot(n, w_i), 0.)
                        if sampled_light_triangle == 0:
                            Lo = (Le*visibility*fr*max1) / (p_mis)
                        else: 
                            max2 = max(tm.dot(ny, -w_i), 0.)
                            shadow_hit_data_distance = shadow_hit_data.distance**2
                            if shadow_hit_data_distance == 0:
                                color = tm.vec3(0.)
                            else:
                                Lo = (Le*visibility*fr*max1*max2) / (p_light*shadow_hit_data_distance)
                        color = Lo
                else:
                    color = self.scene_data.environment.query_ray(ray)
                # pass     
                     
        return color


@ti.data_oriented
class A4Renderer:

    # Enumerate the different sampling modes
    class ShadingMode(IntEnum):
        IMPLICIT = 1
        EXPLICIT = 2

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
        self.canvas_postprocessed = ti.Vector.field(n=3, dtype=float, shape=(width, height))

        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data
        
        self.max_bounces = ti.field(dtype=int, shape=())
        self.max_bounces[None] = 5

        self.rr_termination_probabilty = ti.field(dtype=float, shape=())
        self.rr_termination_probabilty[None] = 0.0

        self.shading_mode = ti.field(shape=(), dtype=int)
        self.set_shading_implicit()

        self.meshLightSampler = MeshLightSampler(self.scene_data.geometry, self.scene_data.material_library)

    def set_shading_implicit(self): self.shading_mode[None] = self.ShadingMode.IMPLICIT
    def set_shading_explicit(self): self.shading_mode[None] = self.ShadingMode.EXPLICIT

    @ti.kernel
    def postprocess(self):
        for x,y in ti.ndrange(self.width, self.height):
            self.canvas_postprocessed[x, y] = tm.pow(self.canvas[x, y], tm.vec3(1.0 / 2.2))
            self.canvas_postprocessed[x, y] = tm.clamp(self.canvas_postprocessed[x, y], xmin=0.0, xmax=1.0)

    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] += (color - self.canvas[x,y])/self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        
        if self.shading_mode[None] == int(self.ShadingMode.IMPLICIT):
            color = self.shade_implicit(ray)
        elif self.shading_mode[None] == int(self.ShadingMode.EXPLICIT):
            color = self.shade_explicit(ray)

        return color

    @ti.func
    def shade_implicit(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.,0.,0.)
        throughput = tm.vec3(1.0,1.0,1.0)
                
        # TODO1 A4: Implement Implicit Path Tracing
        for i in range(self.max_bounces[None] + 1):
            hit_data = self.scene_data.ray_intersector.query_ray(ray) # background
            if hit_data.is_hit:
                material = self.scene_data.material_library.materials[hit_data.material_id]
                if material.Ke.norm() > 0 and not hit_data.is_backfacing:
                    color = material.Ke * throughput
                    break
                else:
                    n = hit_data.normal
                    w_o = -ray.direction
                    ray2_wi = tm.vec3(0.)
                    ray2_x = ray.origin + (hit_data.distance * ray.direction) + (self.RAY_OFFSET * n)
                    ray2_wi = BRDF.sample_direction(material, w_o, n)
                    ray2 = Ray(ray2_x, ray2_wi)

                    fr = BRDF.evaluate_brdf(material, w_o, ray2_wi, n)
                    pdf = BRDF.evaluate_probability(material, w_o, ray2_wi, n)
                    brdf_factor = (fr / pdf) * (tm.dot(n, ray2_wi))

                    throughput *= brdf_factor
                    ray = ray2
            # TODO546 A4: Implement Specular Caustics Support - ECSE 546 Deliverable

        return color
    
    @ti.func
    def shade_explicit(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.,0.,0.)
        throughput = tm.vec3(1.0,1.0,1.0)
        # TODO1 A4: Implement Explicit Path Tracing
        for i in range(self.max_bounces[None]):
            hit_data = self.scene_data.ray_intersector.query_ray(ray)
            if hit_data.is_hit: # not hit light
                n = hit_data.normal
                material = self.scene_data.material_library.materials[hit_data.material_id]
                if material.Ke.norm()> 0 and not hit_data.is_backfacing:
                    color = material.Ke*throughput
                else:
                    # explicit: light sampler
                    ray_explicit_x = ray.origin + (hit_data.distance * ray.direction) + (self.RAY_OFFSET * n) # hit point
                    ray_explicit_wi, triangle_id = self.meshLightSampler.sample_mesh_lights(ray_explicit_x)

                    ray_explicit = Ray(ray_explicit_x, ray_explicit_wi) # shadow ray
                    hit_data_explicit = self.scene_data.ray_intersector.query_ray(ray_explicit)
                    material_explicit = self.meshLightSampler.material_library.materials[hit_data_explicit.material_id]

                    # max
                    max1 = max(tm.dot(n, ray_explicit_wi), 0.)
                    max2 = max(tm.dot(hit_data_explicit.normal, -ray_explicit_wi), 0.)
                    # p_light
                    pdf = self.meshLightSampler.evaluate_probability()
                    # ||x-y||^2
                    distance_explicit = hit_data_explicit.distance**2
                    # fr
                    w_o = -ray.direction
                    fr = BRDF.evaluate_brdf(material, w_o, ray_explicit_wi, n)
                    # color: only when ray is not backfacing or hit an emissive object, its countribution will be counted
                    if not hit_data_explicit.is_backfacing and material_explicit.Ke.norm() > 0:
                        color += material_explicit.Ke * throughput * ((fr * max1 * max2) / (distance_explicit * pdf))
                    
                    # TODO1 A4: Implement Russian Roulette Support
                    random = ti.random()
                    if random < self.rr_termination_probabilty[None]:
                        break

                    # implicit: BEDF
                    ray_implicit_wi = BRDF.sample_direction(material, w_o, n)
                    brdf_factor = BRDF.evaluate_brdf_factor(material, ray_implicit_wi, n)

                    ray_implicit = Ray(ray_explicit_x, ray_implicit_wi)
                    
                    hit_data_implicit = self.scene_data.ray_intersector.query_ray(ray_implicit)
                    material_implict = self.meshLightSampler.material_library.materials[hit_data_implicit.material_id]
                    
                    throughput *= brdf_factor / (1 - self.rr_termination_probabilty[None])
                    if hit_data_implicit.is_hit and material_implict.Ke.norm() == 0:
                        ray = ray_implicit
                        hit_data = hit_data_implicit
                    else:
                        break    
        return color
