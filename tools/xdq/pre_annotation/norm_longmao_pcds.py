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
    "1": [22, -70, 45.75],
    "2": [-23.32, 35.89, 45.75],
    "3": [-25, 98, 45.34],
    "7": [-65, -31, 45.19],
    "12": [290, -120, 45.99],
    "16": [-50, 62, 46.91],
    "17": [8, -8, 45.41],
    "19": [39, -68, 46.21],
    "21": [19, 73, 45.6],
    "32": [55, 12, 45.93],
    "33": [-75, 1, 45.97],
    "34": [-9, -5, 45.71],
    "35": [55, 55, 46.03],
}

for tag in os.listdir(data_dir):
    if "txt" in tag or "csv" in tag or "calib" in tag:
        continue

    subdir = os.path.join(data_dir, tag)
    if not os.path.isdir(subdir):
        continue

    if tag not in norm_offsets:
        continue
    norm_offset = norm_offsets[tag]

    print(f"=> processing {tag}...")
    for pcd_file in tqdm(glob(os.path.join(data_dir, tag, "longmao/*.npy"))):
        points = np.load(pcd_file)
        points[:, :3] += np.array(norm_offset)
        save_path = pcd_file.replace(".npy", "_norm.npy")
        np.save(save_path, points)
