import argparse
import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/xdq")
args = parser.parse_args()

cnt = defaultdict(int)
stride = 50
for file_path in tqdm(glob(os.path.join(args.data_dir, "*/*/*_norm.json"))):
    anno = json.load(open(file_path))
    for obj in anno["lidar_objs"]:
        box = np.array(obj["3d_box"])
        center_x, center_y, _ = box.mean(axis=0)
        dist = (center_x**2 + center_y**2) ** 0.5
        cnt[int(dist / stride)] += 1

print(f"num of boxes: {sum(cnt.values())}")
for k, v in cnt.items():
    print(f"num of boxes in range [{k*50}, {(k+1)*50}]: {v}")
