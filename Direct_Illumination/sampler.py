from typing import List

import taichi as ti
import taichi.math as tm
import numpy as np

from .geometry import Geometry
from .materials import MaterialLibrary, Material

#TODO: Implement Uniform Sampling Methods
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

#TODO: Implement BRDF Sampling Methods
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
        fr = tm.vec3(0.)
        w_r = tm.vec3(0.)
        a = material.Ns
        if a == 1.0:
            fr = (1/tm.pi) * max(0., tm.pow(tm.dot(normal, w_i), a))
        else:
            w_r = (2 * (tm.dot(normal,w_o)) * normal) - w_o
            fr = ((a+1)/(2.*tm.pi)) * max(0., tm.pow(tm.dot(w_r,w_i), a))
        return fr   

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
# TODO: Implement Microfacet BRDF Methods
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


'''
Ignore for now
'''
@ti.data_oriented
class MeshLightSampler:

    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        pass


    def get_emissive_triangle_indices(self) -> List[int]:
        pass


    @ti.kernel
    def compute_emissive_triangle_areas(self):
        pass

    @ti.func
    def compute_triangle_area_given_id(self, triangle_id: int) -> float:
        pass
        

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        pass


    @ti.kernel
    def compute_cdf(self):
        pass


    @ti.func
    def sample_emissive_triangle(self) -> int:
        pass

    @ti.func
    def evaluate_probability(self) -> float:
        pass


    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        pass



@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    pass


@ti.func
def reflect(ray_direction:tm.vec3, normal: tm.vec3) -> tm.vec3:
    pass