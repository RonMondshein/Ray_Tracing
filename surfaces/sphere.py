import numpy as np
from surfaces.surface import Surface  
from surfaces.surface import Intersection  

class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        super(Sphere, self).__init__(material_index)
        self.position = np.asarray(position, dtype="float")
        self.radius = radius

    def ray_intersection(self, ray):
        center_ray = ray.origin - self.position # vector from the origin of the ray to the center of the sphere
        a = 1
        b = 2 * np.dot(ray.direction, center_ray) 
        c = np.dot(center_ray, center_ray) - (self.radius**2)

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None
        else:
            t_first = (-b - np.sqrt(discriminant)) / (2*a)
            t_second = (-b + np.sqrt(discriminant)) / (2*a)

            if t_first < 0 and t_second < 0:
                return None 

            if t_first < 0:
                t_first = float('inf')

            if t_second < 0:
                t_second = float('inf')
            
            t_final = min(t_first, t_second)

            return Intersection(self, ray, t_final) 
        
    def find_normal(self, pnt) -> np.ndarray:
        return ((pnt - self.position) /  np.linalg.norm(pnt - self.position))
    
    def multipal_rays_intersection(self, rays):
        
        ray_o_vals = []
        ray_v_vals = []
        t_final_vals = []
        Intersections = []

        for ray in rays:
            ray_o_vals.append(ray.origin)
            ray_v_vals.append(ray.direction)

        center_ray = ray_o_vals - self.position # vector from the origins of the rays to the center of the sphere
        a = 1
        b_vals = 2 * np.einsum('ij,ij->i', ray_v_vals, center_ray) 
        c_vals = np.einsum('ij,ij->i', center_ray, center_ray) - (self.radius**2) 

        discriminant_vals = b_vals**2 - 4*c_vals*a
        negative_discs = discriminant_vals < 0
        discriminant_vals[negative_discs] = 0
        
        t_first_vals = (-b_vals - np.sqrt(discriminant_vals)) / (2*a)
        t_second_vals = (-b_vals + np.sqrt(discriminant_vals)) / (2*a)

        # compare the two t values and choose the smallest one
        for i in range(len(t_first_vals)):
            if t_first_vals[i] < 0:
                if t_second_vals[i] < 0:
                     t_final_vals.append(None)
                t_final_vals.append(t_second_vals[i])
            if t_second_vals[i] < 0:
                 t_final_vals.append(t_first_vals[i])
            t_final_vals.append(min(t_first_vals[i], t_second_vals[i]))

        # calculate the intersection points where t is not None or disc is not negative
        for i in range(len(rays)):
            if (not negative_discs[i]) and t_final_vals[i] is not None:
                Intersections.append(Intersection(self, rays[i], t_final_vals[i]))
            else:
                Intersections.append(None)

        return Intersections
    
    def __type__(self):
        return 'Sphere'
