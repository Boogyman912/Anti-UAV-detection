import cv2
from pipeline.video_loader import DualVideoLoader
from pipeline.motion_detector import MotionDetector
from pipeline.kinematics import Kinematics

RGB_VIDEO = "data/test_rgb.mp4"
IR_VIDEO  = "data/test_ir.mp4"

loader = DualVideoLoader(RGB_VIDEO, IR_VIDEO)
motion = MotionDetector()
kin = Kinematics()

# Get actual FPS
fps = loader.rgb_cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

DELAY = int(1000 / fps)

print(f"Running Detection at {fps:.2f} FPS...")

while True:
    rgb, rgb_gray, ir_gray = loader.read()
    if rgb is None:
        break

    # Motion mask
    mask = motion.get_mask(rgb_gray)

    # Find contours on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)

        # Center of bounding box
        cx = x + w / 2
        cy = y + h / 2

        # Distance & velocity
        D = kin.distance(h)
        V = kin.velocity((cx, cy), D, fps=fps)

        label = f"MOTION | {round(D,1)}m | {round(V,1)} m/s"

        # 🔴 DRAW ON ORIGINAL RGB FRAME
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # 🔴 DISPLAY ORIGINAL VIDEO WITH BOXES
    cv2.imshow("UAV + Bird Detection", rgb)

    if cv2.waitKey(DELAY) & 0xFF == ord("q"):
        break

loader.release()
cv2.destroyAllWindows()

print("Detection finished.")
