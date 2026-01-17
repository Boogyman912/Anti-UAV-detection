import cv2

def get_bboxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w*h < 10:   # very small threshold
            continue

        boxes.append((x, y, x+w, y+h))

    return boxes

def box_height(box):
    return box[3] - box[1]
