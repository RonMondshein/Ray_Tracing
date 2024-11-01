import numpy as np
from ray import Ray
from intensity import batch_light_intensity_calc
from intersections import ray_intersections


def find_reflection_color(intersection, scene_settings, camera, surfaces, materials, lights, recursions_level) -> np.ndarray: 
    """"""""""
    get_reflection_color
    input - intersection, current_ref_level
    output - ref_color
    recursive function that calculates the reflection color using the reflection ray
    """""""""""
    #if we reached the maximum number of recursions return the background color beacuse it's part of the recursion
    max_recursions = scene_settings.max_recursions
    if recursions_level >= max_recursions:
        #base case
        return scene_settings.background_color
    ref_ray = intersection.surface.get_reflection(intersection.ray, intersection.intersection_point)
    intersections = ray_intersections(ref_ray, surfaces)
    ref_color = find_color(intersections, scene_settings, camera, surfaces, materials, lights, recursions_level + 1)
    return ref_color


def find_color(inters, scene_settings, camera, surfaces, materials, lights, recursions_level = 0) -> np.ndarray: 
    """"""""""
    find_color
    input - inters, recursions_level (0 as default)
    output - color (As np array)
    calculate the color of the intersection point
    """""""""""
    background_color = scene_settings.background_color
    color = None

    #if there are no intersections return the background color
    if inters is None or len(inters) == 0:
        return background_color
    
    for i in range(len(inters)):
        #if the material is not transparent (opaque) break, we don't need to calculate the rest of the intersections that are behind
        if inters[i].surface.get_material(materials).transparency == 0:
            inters = [inters[j] for j in range(i + 1)]
            break

    count_inters = len(inters) - 1
    while count_inters >= 0:
        #from the last intersection to the first
        current_intersection = inters[count_inters]
        #calculate the color of the intersection
        reflection_color = find_reflection_color(current_intersection, scene_settings, camera, surfaces, materials, lights, recursions_level)
        #multiply the reflection color by the material reflection color
        reflection_color = reflection_color * current_intersection.surface.get_material(materials).reflection_color
        #calculate the diffuse and specular color of the intersection
        diffuse_specular_color = find_diffuse_and_specular_color(current_intersection, scene_settings, lights, materials, surfaces)
        #calculate the transparency of the intersection
        transparency = current_intersection.surface.get_material(materials).transparency
        #calculate the color of the intersection
        color = ((1 - transparency) * diffuse_specular_color + transparency * background_color) + reflection_color
    
        background_color = color
        count_inters -= 1
    return color


def find_diffuse_and_specular_color(intersection, scene_settings, lights, materials, surfaces) -> np.ndarray:
    """""""""""""""""""""
    find_diffuse_and_specular_color
    input - intersection
    output - combinedColor
    find the diffuse and specular color of the intersection

    """""""""""""""""""""
    #if there is no intersection return the background color
    if intersection is None:
        return scene_settings.background_color
    
    specularSum = np.array([0.0, 0.0, 0.0])
    diffuseSum = np.array([0.0, 0.0, 0.0])

    # for each light source calculate the diffuse and specular color
    for lgt in lights:
        normal = intersection.surface.find_normal(intersection.intersection_point)
        lightVector = (lgt.position - intersection.intersection_point) / np.linalg.norm(lgt.position - intersection.intersection_point)
        light_normal_dot = np.dot(normal, lightVector)

        #if the light is behind the surface
        if light_normal_dot <= 0:
            continue

        #get the intensity of the light and the color of the light
        intensityOfLight, colorOfLight = batch_light_intensity_calc(lgt, intersection, scene_settings, surfaces, materials)
        eyeVector = (intersection.ray.origin - intersection.intersection_point) / np.linalg.norm(intersection.ray.origin - intersection.intersection_point)
        light_ray = Ray(lgt.position, -lightVector)
        reflectionVector = intersection.surface.get_reflection(light_ray, intersection.intersection_point).direction
        color_intensity_product = colorOfLight * intensityOfLight
        specularReflectionTerm = np.power(np.dot(reflectionVector, eyeVector), intersection.surface.get_material(materials).shininess)

        diffuseSum += color_intensity_product * light_normal_dot
        specularSum += color_intensity_product * lgt.specular_intensity * specularReflectionTerm

    materialDiffuseColor = intersection.surface.get_material(materials).diffuse_color
    materialSpecular = intersection.surface.get_material(materials).specular_color

    diffuse_color = diffuseSum * materialDiffuseColor
    specular_color = specularSum * materialSpecular

    #combine the colors
    combinedColor = diffuse_color + specular_color
    return combinedColor


