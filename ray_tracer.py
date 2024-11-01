import argparse
from PIL import Image
import numpy as np
import time

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray

from surfaces.surface import Epsilon as Epsilon
from color import find_color
from color import ray_intersections

global scene_settings
global camera
global lights
global surfaces
global materials
global width
global height

#constants
INFINITY = float('inf')

def save_image(image_array, img_name="output"):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(f"{img_name}")

def build(output, camera0):
    """""""""""""""""""""
    build the final image
    input - output, camera, scene_settings, objects
    output - output image
    """""""""""""""""""""
    global width
    global height

    #init camera
    global camera
    camera = camera0
    camera.set_ratio(width)

    for pixel_row in range(height):
        for pixel_column in range(width):
            #for each pixel in the image find all intresections rays and find the color
            ray = Ray(camera, pixel_row, pixel_column, width, height)
            intersections = ray_intersections(ray, surfaces)
            color = find_color(intersections, scene_settings, camera, surfaces, materials, lights, 0)
            #update the pixel color
            output[pixel_row, pixel_column] = color

    output = np.clip(output, 0, 1)
    return output * 255


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def init_objects(objects):
    """""""""""""""""""""""
    init_objects
    input - objects
    output - surfaces, lights, materials as global variables
    """""""""""""""""""""""
    global surfaces
    global lights
    global materials
    surfaces = [obj for obj in objects if obj.__type__() == 'Sphere' or obj.__type__() == 'InfinitePlane' or obj.__type__() == 'Cube']
    lights = [obj for obj in objects if obj.__type__() == 'Light']
    materials = [obj for obj in objects if obj.__type__() == 'Material']

    
    if len(lights) == 0:
        raise ValueError("No light sources in the scene")
    if len(materials) == 0:
        raise ValueError("No materials in the scene")
    if len(surfaces) == 0:
        raise ValueError("No surfaces in the scene")


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')

    args = parser.parse_args()
    parameters = vars(args)
    global width
    global height
    global scene_settings
    width = parameters['width']
    height = parameters['height']
    # Parse the scene file
    camera, scene_settings0, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    #Add the objects
    init_objects(objects)
    scene_settings = scene_settings0
    output = np.zeros((height, width, 3), dtype='float')
    image_array = build(output, camera)

    # Save the output image
    save_image(image_array, args.output_image)
    #check timing
    print(time.time()-start)


if __name__ == '__main__':
    main()
