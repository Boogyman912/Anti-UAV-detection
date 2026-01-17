import cv2

def draw(frame, box, label, D, V):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    text = f"{label} | D={D:.1f}m | V={V:.2f}m/s"
    cv2.putText(frame, text, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame
