import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir")
args = parser.parse_args()
data_dir = args.data_dir

norm_offsets = {
    "2": [-29.47, 32.36, 45.5],
    "3": [-14.57, 73.2, 45.24],
    "12": [181.5, -80.63, 45.93],
    "21": [13.57, 73.88, 45.45],
    "32": [56.18, 5.57, 45.58],
    "33": [-57.96, -7.58, 45.62],
    "34": [-6.65, -23.98, 45.46],
    "35": [63.9, 51.86, 45.73],
}

for tag in os.listdir(data_dir):
    if "txt" in tag or "csv" in tag or "calib" in tag:
        continue

    subdir = os.path.join(data_dir, tag)
    if not os.path.isdir(subdir):
        continue

    norm_offset = norm_offsets[tag]

    print(f"=> processing {tag}...")
    for pcd_file in tqdm(glob(os.path.join(data_dir, tag, "longmao/*.npy"))):
        points = np.load(pcd_file)
        points[:, :3] += np.array(norm_offset)
        save_path = pcd_file.replace(".npy", "_norm.npy")
        np.save(save_path, points)
