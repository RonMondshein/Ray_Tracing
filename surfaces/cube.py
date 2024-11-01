from surfaces.surface import Surface
from surfaces.surface import Intersection  
from surfaces.infinite_plane import InfinitePlane as Plane
import numpy as np
from surfaces.surface import Epsilon as Epsilon

class Cube(Surface):
    def __init__(self, position, scale, material_index):
        super(Cube, self).__init__(material_index)
        self.position = np.array(position, dtype=np.float64)
        self.scale = scale
        self.material_index = material_index
        # Compute the lower and upper bounds of the cube
        self.lower_bound = self.position - self.scale / 2
        self.upper_bound = self.position + self.scale / 2
        half_scale = self.scale / 2

        pos_zero = self.position[0]
        pos_one = self.position[1]
        pos_two = self.position[2]

        # Create the faces of the cube (6 planes) according to the position, scale and add the material index
        self.faces = [
            # Front face (X = 1)
            Plane(np.array([1, 0, 0], dtype=np.float64), pos_zero + half_scale, material_index),
            # Back face (X = -1)
            Plane(np.array([-1, 0, 0], dtype=np.float64), - pos_zero + half_scale, material_index),
            # Right face (Y = 1)
            Plane(np.array([0, 1, 0], dtype=np.float64), pos_one + half_scale, material_index),
            # Left face (Y = -1)
            Plane(np.array([0, -1, 0], dtype=np.float64), - pos_one + half_scale, material_index),
            # Top face (Z = 1)
            Plane(np.array([0, 0, 1], dtype=np.float64), pos_two + half_scale, material_index),
            # Bottom face (Z = -1)
            Plane(np.array([0, 0, -1], dtype=np.float64), - pos_two + half_scale, material_index)
        ]

    def is_inside(self, point) -> bool:
        return np.all(point >= self.lower_bound) and np.all(point <= self.upper_bound)
    
    def find_normal(self, point) -> np.ndarray:
        # Compute the normal of the cube at a given point
        # The normal is the vector from the point to the center of the cube
        for plane in self.faces:
            if np.isclose(np.dot(plane.normal, point), plane.offset, atol=Epsilon):
                return plane.normal

    def ray_intersection(self, ray) -> Intersection:
        direction = ray.direction #Direction of the ray
        origin = ray.origin #Origin of the ray

        with np.errstate(divide='ignore'): #Ignore division by zero errors
            #ray-box intersection test
            # Initialize the minimum and maximum distances
            low = (self.lower_bound - origin) / direction
            high = (self.upper_bound - origin) / direction

            # Find the entering and exiting points of intersection
            enter_t_val = np.max(np.minimum(low, high))
            exit_t_val = np.min(np.maximum(low, high))
            if enter_t_val > exit_t_val:
                return None # No intersection
            return Intersection(self, ray, enter_t_val) 

    def multipal_rays_intersection(self, rays) -> list:
        rays_directions = np.array([ray.direction for ray in rays]) #Directions of the rays
        rays_origins = np.array([ray.origin for ray in rays]) #Origins of the rays

        # Calculate intersection parameters
        with np.errstate(divide='ignore'):  # Ignore division by zero errors
            min = (self.lower_bound - rays_origins) / rays_directions
            max = (self.upper_bound - rays_origins) / rays_directions

        # Find intersection times
        enter_t_val = np.max(np.minimum(min, max), axis=1) 
        exit_t_val = np.min(np.maximum(min, max), axis=1) 

        # Generate intersection results
        intersections = [Intersection(self, rays[i], enter_t_val[i]) if exit_t_val[i] >= enter_t_val[i] else None for i in
                    range(len(rays))]
        return intersections
    
    def __type__(self):
        return 'Cube'
    