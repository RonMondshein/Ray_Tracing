from surfaces.surface import Surface
import numpy as np
from surfaces.surface import Intersection  
from surfaces.surface import Epsilon as Epsilon


class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        super(InfinitePlane, self).__init__(material_index)
        self.offset = offset
        self.normal = np.array(normal, dtype=np.float64) / np.linalg.norm(np.array(normal, dtype=np.float64))

    def ray_intersection(self, ray) -> Intersection:
        normal_ray_dot = np.dot(self.normal, ray.direction) # dot product of the normal and the direction of the ray

        # check wethear the ray is parallel to the plane
        if abs(normal_ray_dot) < Epsilon: # small epsilon value 
            return None
        
        t_val = np.dot(self.offset * self.normal - ray.origin, self.normal) / normal_ray_dot

        # check if the intersection is behind the ray
        if t_val < 0:
            return None
        
        return Intersection(self, ray, t_val) 
    
    def find_normal(self, point):
        return self.normal
    
    def multipal_rays_intersection(self, rays) -> list:
        ray_o_vals = []
        ray_v_vals = []
        Intersections = []

        # divison by zero will cause inf values
        with np.errstate(divide='ignore'):
            for ray in rays:
                ray_o_vals.append(ray.origin)
                ray_v_vals.append(ray.direction)

            ray_o_vals = np.array(ray_o_vals)
            ray_v_vals = np.array(ray_v_vals)

            normal_ray_dot = np.einsum('ij,j->i', ray_v_vals, self.normal)
            t_vals = np.einsum('ij,j->i', self.offset * self.normal - ray_o_vals, self.normal) / normal_ray_dot

            # check if the intersection is behind the ray
            is_behind =  (t_vals < 0) | (np.abs(normal_ray_dot) < Epsilon) 

            for i in range(len(rays)):
                if not is_behind[i]:
                    Intersections.append(Intersection(self, rays[i], t_vals[i]))
                else:
                    Intersections.append(None)

            return Intersections
        

    def __type__(self):
        return 'InfinitePlane'


