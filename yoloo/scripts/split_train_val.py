import os
import random
import shutil

IMG_DIR = "data/yolo/images/train"
LBL_DIR = "data/yolo/labels/train"

VAL_IMG = "data/yolo/images/val"
VAL_LBL = "data/yolo/labels/val"

VAL_RATIO = 0.2

images = sorted(os.listdir(IMG_DIR))
random.shuffle(images)

val_count = int(len(images) * VAL_RATIO)
val_images = images[:val_count]

for img in val_images:
    src_img = os.path.join(IMG_DIR, img)
    dst_img = os.path.join(VAL_IMG, img)
    shutil.move(src_img, dst_img)

    lbl = img.replace(".jpg", ".txt")
    src_lbl = os.path.join(LBL_DIR, lbl)
    dst_lbl = os.path.join(VAL_LBL, lbl)

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print(f"Moved {val_count} images to validation set")
