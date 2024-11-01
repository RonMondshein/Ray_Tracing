from ray import Ray
from material import Material
import numpy as np

Epsilon = 0.00001

class Intersection:
    def __init__(self, surface, ray, distance):
        self.surface = surface  # The surface (or object) that the ray intersects
        self.ray = ray  # The ray that intersects the surface
        self.distance = distance  # The distance from the ray's origin to the intersection point
        self.intersection_point = ray.origin + distance * ray.direction  # Calculate the intersection point

class Surface():
    def __init__(self, material_num):
        self.material = material_num - 1
  
    def get_material(self, materials) -> Material:
        return materials[self.material]
    
    def normalize(vector) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    def cos_theta(vector_a, vector_b) -> float:
        # Normalize the vectors
        vector_a = Surface.normalize(vector_a)
        vector_b = Surface.normalize(vector_b)
        # Compute the dot product
        dot_product = np.dot(vector_a, vector_b)
        return dot_product
    
    def get_reflection(self, ray: Ray, point) -> Ray:
        # Calculate the reflection direction
        n = self.find_normal(point)
        normalize_n = Surface.normalize(n)
        normalize_incident_direction = Surface.normalize(ray.direction)
        costeta = normalize_incident_direction @ normalize_n
        reflection_direction = - 2 * costeta * n + ray.direction
        return Ray(point, reflection_direction)

    def ray_intersection(self, ray: Ray):
        pass
    def multipal_rays_intersection(self, rays):
        pass
    def find_normal(self, point):
        pass

