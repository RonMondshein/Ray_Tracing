import numpy as np
from ray import Ray
from surfaces.surface import Epsilon as Epsilon
from transform_utils import plane_mapping_matrix

INFINITY = float('inf')

def batch_light_intensity_calc(light, intersection, scene_settings, surfaces, materials) -> tuple:
    """""""""""""""""
    batch_light_intensity_calc
    input - light, intersection
    output - light_intensity, light_color

    calculate the light intensity and color of the light
    """""""""""""""""    
    # Calculate the normal vector to the plane defined by the intersection point and the light position
    normal_of_pln = (intersection.intersection_point - light.position) / np.linalg.norm(intersection.intersection_point - light.position)
    
    # Get the transformation matrix from the xy-plane to the general plane defined by the light and intersection
    mtx_transformed = plane_mapping_matrix(normal_of_pln, light.position)
    
    # Calculate the length of each unit square in the grid
    num_shadow_rays = scene_settings.root_number_shadow_rays
    unitLength = light.radius / num_shadow_rays
    
    # Create a grid of starting points for shadow rays using broadcasting
    x_grid_values = (np.arange(num_shadow_rays) * unitLength - light.radius / 2).reshape(-1, 1)
    y_grid_values = (np.arange(num_shadow_rays) * unitLength - light.radius / 2).reshape(1, -1)
    
    # Create the base points in xy-plane with broadcasting
    base_x, base_y = np.meshgrid(x_grid_values, y_grid_values)
    xy_coordinates = np.vstack([base_x.flatten(), base_y.flatten(), np.zeros(base_x.size), np.zeros(base_x.size)])
    
    # Offset points randomly within each cell of the grid
    offset_x = np.random.uniform(0, unitLength, base_x.size)
    offset_y = np.random.uniform(0, unitLength, base_y.size)
    offset = np.vstack([offset_x, offset_y, np.zeros(base_x.size), np.ones(base_x.size)])
    
    # Calculate the rectangle points in xy-plane
    transformed_points = xy_coordinates + offset
    
    # Transform the points from the xy-plane to the general plane
    light_points = (np.dot(mtx_transformed, transformed_points))[:3].T
    
    # Create rays from the light points to the intersection point
    rays = [Ray(point, intersection.intersection_point - point) for point in light_points]

    return find_intensity(light, intersection, rays, scene_settings, surfaces), light.color

def find_intensity(light_ray, intersection, rays, scene_settings, surfaces) -> float: 
    """"""""""""""""""""""
    find_intensity
    input - light_ray, intersection, rays
    output - intensity
    calculate the intensity of the light at the intersection point
    """""""""""""""""""""""
    # Find the closest intersections for a batch of rays
    intersections_rays = find_intersections_group(rays, surfaces)

    # Count the number of light hits that are close enough to the intersection point
    # Initialize the count of close hits
    close_hits_count = 0

    # Iterate over each light hit to count the close ones
    for light_hit in intersections_rays:
        if light_hit is not None:
            distance = np.linalg.norm(intersection.intersection_point - light_hit.intersection_point)
            if distance < Epsilon:
                close_hits_count += 1

    # Calculate the total number of shadow rays
    total_shadow_rays = scene_settings.root_number_shadow_rays ** 2
    # Calculate the light intensity considering shadow rays and shadow intensity
    base_intensity = 1 - light_ray.shadow_intensity
    shadow_contribution = light_ray.shadow_intensity * (close_hits_count / total_shadow_rays)
    intensity = base_intensity + shadow_contribution

    return intensity


def find_intersections_group(rays, surfaces) -> list: 
    """""""""""
    find_intersections_group
    input - rays
    output - intersections
    find all the closest intersections of the rays with the surfaces
    """""""""""
    # Initialize arrays to store minimum t values and closest intersections
    min_values = np.full(len(rays), INFINITY)
    closest = [None] * len(rays)
    closest = np.array(closest)
    
    # Iterate over each surface to find intersections
    for surface in surfaces:
        # Find intersections for the current surface with the batch of rays
        intersectionsArray = surface.multipal_rays_intersection(rays)
        intersectionsArray = np.array(intersectionsArray)
        
        # Extract distances to intersections or set to infinity if no intersection
        intersections_t = []
        for intersection in intersectionsArray:
            if intersection is not None:
                intersections_t.append(intersection.distance)
            else:
                intersections_t.append(INFINITY)
        
        # Update the minimum t values and closest intersections
        to_change = intersections_t < min_values
        min_values = np.minimum(min_values, intersections_t)
        # Update closest intersections where new intersections are closer
        closest[to_change] = intersectionsArray[to_change]
          
    return closest

