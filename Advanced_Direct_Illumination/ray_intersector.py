from abc import ABC, abstractmethod

import taichi as ti
import taichi.math as tm

from .geometry import Geometry
from .materials import Material
from .ray import Ray, HitData


@ti.data_oriented
class RayIntersector(ABC):

    def __init__(self, geometry: Geometry):
        self.EPSILON = 1e-7
        self.geometry = geometry


    @abstractmethod
    @ti.func
    def query_ray(ray: Ray) -> HitData:
        pass


    @ti.func
    def intersect_triangle(self, ray: Ray, triangle_id: int) -> HitData:

        hit_data = HitData()

        # Grab Vertices
        vert_ids = self.geometry.triangle_vertex_ids[triangle_id-1] - 1  # Vertices are indexed from 1
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # Normals at each vertex
        normal_indices = self.geometry.triangle_normal_ids[triangle_id-1]-1

        normal_0 = self.geometry.normals[normal_indices[0]]
        normal_1 = self.geometry.normals[normal_indices[1]]
        normal_2 = self.geometry.normals[normal_indices[2]]

        # Material of the triangle
        material_id = self.geometry.triangle_material_ids[triangle_id-1]

        '''
        TODO1: Copy your A1 solution
        '''
        e1 = v1-v0
        e2 = v2-v0
        det = tm.dot(e1, tm.cross(ray.direction, e2))
        intersect = False
        if ti.abs(det) > self.EPSILON:
            hit_data.triangle_id = triangle_id
            hit_data.material_id = material_id
            intersect = True
            if det > 0:
                hit_data.is_backfacing = False
            else:
                hit_data.is_backfacing = True
        if intersect: #if an intersect occurs, compute three barycentric coordinates u, v, w
            u = (1/det)*tm.dot((ray.origin-v0),(tm.cross(ray.direction, e2)))
            v = (1/det)*tm.dot(ray.direction, (tm.cross(ray.origin-v0, e1)))
            w = 1-u-v
            normal = tm.normalize(w*normal_0 + u*normal_1 + v*normal_2)
            hit_data.normal = normal
            if hit_data.is_backfacing: 
                hit_data.normal = -normal
            hit_data.barycentric_coords = ti.Vector([u, v])
            if u >= 0 and u <= 1 and v >= 0 and v<=1 and u+v <= 1:
                t = (1/det) * tm.dot(tm.cross(ray.origin-v0, e1), e2)
                hit_data.distance = t
                if t > self.EPSILON:
                    hit_data.is_hit = True
                else: 
                    hit_data.is_hit = False
        return hit_data

@ti.data_oriented
class BruteForceRayIntersector(RayIntersector):

    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)


    @ti.func
    def query_ray(self, ray: Ray) -> HitData:

        closest_hit = HitData()
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            hit_data = self.intersect_triangle(ray, triangle_id)

            if hit_data.is_hit:
                if (hit_data.distance < closest_hit.distance) or (not closest_hit.is_hit):
                    closest_hit = hit_data

        return closest_hit