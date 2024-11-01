import numpy as np

screen_center = None
direction = None
dir_up_perpendicular = None
ratio = -1

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position, dtype=np.float64)
        self.look_at = np.array(look_at, dtype=np.float64)
        self.up_vector = np.array(up_vector, dtype=np.float64)
        self.screen_distance = screen_distance
        self.screen_width = screen_width


        #direction is the unit vector pointing from the camera position to the look_at point
        self.direction = (self.look_at - self.position)/np.linalg.norm(self.look_at - self.position)
        #screen_center is the point in the middle of the screen
        self.screen_center = self.position + self.screen_distance * self.direction
        #dir_up_perpendicular perpendicular to both direction and up_vector
        self.dir_up_perpendicular = (np.cross(self.direction, self.up_vector))/np.linalg.norm(np.cross(self.direction, self.up_vector))
        #up_vector is the unit vector pointing in the up direction
        self.up_vector = (np.cross(self.dir_up_perpendicular, self.direction))/np.linalg.norm(np.cross(self.dir_up_perpendicular, self.direction))

    def set_ratio(self, width):
        self.ratio = self.screen_width / width
        return self.ratio
    
    def get_ratio(self):
        return self.ratio



