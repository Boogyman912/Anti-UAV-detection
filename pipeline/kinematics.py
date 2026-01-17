import math

class Kinematics:
    def __init__(self):
        self.prev_pos = None

    def distance(self, box_height, focal=700, real_height=0.5):
        if box_height <= 0:
            return 0
        return (real_height * focal) / box_height

    def velocity(self, center, distance, fps=30):
        if self.prev_pos is None:
            self.prev_pos = center
            return 0

        dx = center[0] - self.prev_pos[0]
        dy = center[1] - self.prev_pos[1]

        self.prev_pos = center

        pixel_speed = math.sqrt(dx*dx + dy*dy) * fps
        return (pixel_speed * distance) / 700
