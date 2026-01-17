import cv2

class MotionDetector:
    def __init__(self):
        self.prev = None

    def get_mask(self, gray):
        if self.prev is None:
            self.prev = gray.copy()
            return gray * 0

        diff = cv2.absdiff(self.prev, gray)
        self.prev = gray.copy()

        _, mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)

        return mask
