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
    "1": [-4.46, -34.1, 45.75],
    "2": [-23.32, 35.89, 45.75],
    "3": [-4.64, 71.39, 45.34],
    "7": [-17.48, -19.43, 45.19],
    "12": [184.74, -78.32, 45.99],
    "16": [33.16, 83.63, 46.91],
    "17": [-18.74, -35.58, 45.41],
    "19": [-12.4, -52.47, 46.21],
    "21": [12.77, 71.73, 45.6],
    "32": [48.12, -0.02, 45.93],
    "33": [-53.45, -6.59, 45.97],
    "34": [-8.23, -31.04, 45.71],
    "35": [63.23, 49.72, 46.03],
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
