from typing import List

import taichi as ti
import taichi.math as tm
import numpy as np

from .geometry import Geometry
from .materials import MaterialLibrary, Material

#TODO1: Implement Uniform Sampling Methods
@ti.data_oriented
class UniformSampler:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction() -> tm.vec3:
        w_i = tm.vec3(0.)
        # generate ray direction: wj
        s1 = ti.random()
        s2 = ti.random()
        w_z = 2*s1 - 1
        r = tm.sqrt(1 - w_z**2)
        phi = 2*tm.pi*s2
        w_x = r*tm.cos(phi)
        w_y = r*tm.sin(phi)
        w_i = tm.vec3(w_x,w_y,w_z)
        w_i = tm.normalize(tm.vec3(w_x,w_y,w_z))
        return w_i

    @staticmethod
    @ti.func
    def evaluate_probability() -> float:
        return 1. / (4. * tm.pi)

#TODO1: Implement BRDF Sampling Methods
@ti.data_oriented
class BRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        w_i_local = tm.vec3(0.)
        w_i= tm.vec3(0.)
        w_r = tm.vec3(0.)
        a = material.Ns
        s1 = ti.random()
        s2 = ti.random()
        w_z = tm.pow(s1, (1.0/(a+1)))
        r = tm.sqrt(1 - w_z**2)
        phi = 2.0*tm.pi*s2
        w_x = r * tm.cos(phi)
        w_y = r * tm.sin(phi)
        w_i_local = tm.vec3(w_x, w_y, w_z)
        w_r = (2.0 * (tm.dot(normal,w_o)) * normal) - w_o

        if a == 1.0:
            random_vec = tm.normalize(tm.vec3([ti.random(), ti.random(), ti.random()]))
            x_axis = tm.cross(normal, random_vec)
            x_axis = tm.normalize(x_axis)
            y_axis = tm.cross(x_axis, normal)
            y_axis = tm.normalize(y_axis)
            ortho_frames_diffuse = tm.mat3([x_axis, y_axis, normal]).transpose()
            w_i = ortho_frames_diffuse @ w_i_local
            w_i = tm.normalize(w_i)
        else: 
            random_vec = tm.normalize(tm.vec3([ti.random(), ti.random(), ti.random()]))
            x_axis = tm.cross(w_r, random_vec)
            x_axis = tm.normalize(x_axis)
            y_axis = tm.cross(x_axis, w_r)
            y_axis = tm.normalize(y_axis)
            ortho_frames_phong = tm.mat3([x_axis, y_axis, w_r]).transpose()
            w_i = ortho_frames_phong @ w_i_local
            w_i = tm.normalize(w_i)
        return w_i
    
    
    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float: 
        pdf = 0.0
        w_r = tm.vec3(0.)
        a = material.Ns
        w_r = (2.0 * (tm.dot(normal,w_o)) * normal) - w_o
        w_r = tm.normalize(w_r)
        if a == 1.0:
            pdf = (1/tm.pi) * max(0., tm.dot(normal, w_i))
        elif a > 1.0:
            pdf = ((a+1)/(2.*tm.pi)) * max(0., tm.pow(tm.dot(w_r,w_i), a))
        return pdf   

    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        fr = tm.vec3(0.)
        # fr(x, w0, wj)
        a = material.Ns
        if a == 1.0:
            pd = material.Kd
            fr = pd/tm.pi
        else:
            ps = material.Kd
            w_r = (2 * (tm.dot(normal, w_o)) * normal) - w_o
            w_r = tm.normalize(w_r)
            fr = (ps * (a+1)/(2.*tm.pi)) * min(1,(max(0., tm.pow(tm.dot(w_r,w_i), a))))
        return fr

    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        fr_factor = tm.vec3(0.)
        a = material.Ns
        if a == 1.0:
            pd = material.Kd
            fr_factor = pd
        else:
            ps = material.Kd
            fr_factor = ps * tm.max(tm.dot(normal, w_i), 0.)
        return fr_factor  
     
# Microfacet BRDF based on PBR 4th edition
# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#
# TODO546: Implement Microfacet BRDF Methods
# 546 only deliverable
@ti.data_oriented
class MicrofacetBRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass


    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float: 
        pass
        

    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass


@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        # Find all of the emissive triangles
        emissive_triangle_ids = self.get_emissive_triangle_indices()
        if len(emissive_triangle_ids) == 0:
            self.has_emissive_triangles = False
        else:
            self.has_emissive_triangles = True
            self.n_emissive_triangles = len(emissive_triangle_ids)
            emissive_triangle_ids = np.array(emissive_triangle_ids, dtype=int)
            self.emissive_triangle_ids = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=int)
            self.emissive_triangle_ids.from_numpy(emissive_triangle_ids)

        # Setup for importance sampling
        if self.has_emissive_triangles:
            # Data Fields
            self.emissive_triangle_areas = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.cdf = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.total_emissive_area = ti.field(shape=(), dtype=float)

            # Compute
            self.compute_emissive_triangle_areas()
            self.compute_cdf()


    def get_emissive_triangle_indices(self) -> List[int]:
        # Iterate over each triangle, and check for emissivity 
        emissive_triangle_ids = []
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            material_id = self.geometry.triangle_material_ids[triangle_id-1]
            emissivity = self.material_library.materials[material_id].Ke
            if emissivity.norm() > 0:
                emissive_triangle_ids.append(triangle_id)

        return emissive_triangle_ids


    @ti.kernel
    def compute_emissive_triangle_areas(self):
        for i in range(self.n_emissive_triangles):
            triangle_id = self.emissive_triangle_ids[i]
            vert_ids = self.geometry.triangle_vertex_ids[triangle_id-1] - 1  # Vertices are indexed from 1
            v0 = self.geometry.vertices[vert_ids[0]]
            v1 = self.geometry.vertices[vert_ids[1]]
            v2 = self.geometry.vertices[vert_ids[2]]

            triangle_area = self.compute_triangle_area(v0, v1, v2)
            self.emissive_triangle_areas[i] = triangle_area
            self.total_emissive_area[None] += triangle_area
        

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        # TODO1: Compute Area of a triangle given the 3 vertices
        # 
        # Area of a triangle ABC = 0.5 * | AB cross AC |
        # 
        # placholder
        AB = v1 - v0
        AC = v2 - v0
        ABC_area = 0.5 * tm.length(tm.cross(AB, AC))
        return ABC_area


    @ti.kernel
    def compute_cdf(self):
        # TODO1: Compute the CDF of your emissive triangles
        # self.cdf[i] = ...
        num = 0.
        ti.loop_config(serialize=True)
        for i in ti.ndrange(self.emissive_triangle_areas.shape[0]):
            num += self.emissive_triangle_areas[i]
            self.cdf[i] = num / self.total_emissive_area[None]
        pass


    @ti.func
    def sample_emissive_triangle(self) -> int:
        # TODO1: Sample an emissive triangle using the CDF
        # return the **index** of the triangle
        #
        # placeholder
        triangle_index = 0
        u = ti.random()
        right = self.emissive_triangle_areas.shape[0]-1

        while triangle_index < right:
            mid = (triangle_index + right) // 2
            if self.cdf[mid] < u:
                triangle_index = mid + 1
            else:
                right = mid
        return triangle_index

    @ti.func
    def evaluate_probability(self) -> float:
        # TODO1: return the probabilty of a sample
        #
        # placeholder
        probability = 1.0/self.total_emissive_area[None]
        return probability


    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        sampled_light_triangle_idx = self.sample_emissive_triangle()
        sampled_light_triangle = self.emissive_triangle_ids[sampled_light_triangle_idx]

        # Grab Vertices
        vert_ids = self.geometry.triangle_vertex_ids[sampled_light_triangle-1] - 1  # Vertices are indexed from 1
        
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # generate point on triangle using random barycentric coordinates
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#Sampling
        # https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes#SampleUniformTriangle

        # TODO1: Sample a direction towards your mesh light
        # given your sampled triangle vertices
        # generat random barycentric coordinates
        # calculate the light direction
        # light direction = (point on light - hit point)
        # don't forget to normalize!
        
        # placeholder
        u0 = ti.random()
        u1 = ti.random()
        b0, b1 = 0., 0.
        if (u0 < u1):
            b0 = u0/2.0
            b1 = u1-b0
        else:
            b1 = u1/2.0
            b0 = u0-b1
        b2 = 1-b0-b1
        y = b0*v0 + b1*v1 + b2*v2

        light_direction = tm.vec3(1.0) # wj
        light_direction = tm.normalize(y - hit_point)
        return light_direction, sampled_light_triangle


@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    pass


@ti.func
def reflect(ray_direction:tm.vec3, normal: tm.vec3) -> tm.vec3:
    pass