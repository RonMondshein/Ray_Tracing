import numpy as np
from camera import Camera as Camera

origin = None
direction = None

camera = None
pixel_row = 0.0
pixel_column = 0.0
screen_width = 0.0
screen_height = 0.0

class Ray:
    def __init__(self, *args):
        if len(args) == 2:
            # Constructor with 2 parameters
            self.origin = args[0]            
            self.direction = args[1] / np.linalg.norm(args[1])
        elif(len(args) == 5):
            # Constructor with 5 parameters
            assert isinstance(args[0], Camera), "param1 must be an instance of Camera but it is type " + str(type(args[0]))
            self.Camera = args[0]
            self.pixel_row = args[1]
            self.pixel_column = args[2]
            self.screen_width = args[3]
            self.screen_height = args[4]

            pixel_width = -(self.pixel_column - self.screen_width // 2) 
            pixel_height = (self.pixel_row - self.screen_height // 2) 
            # pixel_relative_camera_position is the position of the pixel relative to the camera
            pixel_relative_camera_position = self.Camera.screen_center + (pixel_width * self.Camera.dir_up_perpendicular - pixel_height * self.Camera.up_vector )* self.Camera.ratio
            self.origin = self.Camera.position
            self.direction = (pixel_relative_camera_position - self.Camera.position) / np.linalg.norm(pixel_relative_camera_position - self.Camera.position)

        else:
            raise ValueError("Invalid number of parameters")
        



            
            
