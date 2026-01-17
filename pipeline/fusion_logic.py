UAV_THRESH = 0.45
BIRD_THRESH = 0.25
RGB_LOW_CONF = 0.35
UAV_TINY_SIZE = 20

def classify(score):
    if score >= UAV_THRESH:
        return "UAV"
    if score >= BIRD_THRESH:
        return "Bird"
    return None

def fuse(rgb_score, ir_score, box_h):
    if box_h < UAV_TINY_SIZE or rgb_score < RGB_LOW_CONF:
        return classify(ir_score)
    return classify(rgb_score)
