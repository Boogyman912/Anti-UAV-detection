import cv2

class DualVideoLoader:
    def __init__(self, rgb_path, ir_path):
        self.rgb_cap = cv2.VideoCapture(rgb_path)
        self.ir_cap = cv2.VideoCapture(ir_path)

    def read(self):
        ok1, rgb = self.rgb_cap.read()
        ok2, ir  = self.ir_cap.read()

        if not ok1 or not ok2:
            return None, None, None

        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ir_gray  = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        return rgb, rgb_gray, ir_gray

    def release(self):
        self.rgb_cap.release()
        self.ir_cap.release()
