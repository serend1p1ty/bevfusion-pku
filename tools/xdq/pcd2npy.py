import argparse
import numpy as np
import os.path as osp
import pypcd
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root-dir", default="data/xdq/source/20220825")
args = parser.parse_args()

for pcd_file in tqdm(glob(osp.join(args.root_dir, "*/*.pcd"))):
    np_file = pcd_file.replace("pcd", "npy")
    pc = pypcd.PointCloud.from_path(pcd_file)
    np.save(
        np_file,
        np.stack((pc.pc_data["x"], pc.pc_data["y"], pc.pc_data["z"], pc.pc_data["intensity"])).T,
    )
