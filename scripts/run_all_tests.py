import os
import subprocess

TEST_DIR = "data/test"
OUT_DIR  = "results"

os.makedirs(OUT_DIR, exist_ok=True)

for set_name in sorted(os.listdir(TEST_DIR)):
    set_path = os.path.join(TEST_DIR, set_name)

    rgb = os.path.join(set_path, "visible.mp4")
    ir  = os.path.join(set_path, "infrared.mp4")
    out = os.path.join(OUT_DIR, f"{set_name}_pred.json")

    if os.path.exists(rgb) and os.path.exists(ir):
        print("Running test on:", set_name)
        subprocess.run(["python", "main.py", rgb, ir, out])
