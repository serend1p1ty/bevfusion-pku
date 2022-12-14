import argparse
import os
import json
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copyfile


def copy_files(file_paths, dst_dir):
    print("Copying files...")
    os.makedirs(dst_dir, exist_ok=True)
    for file_path in tqdm(file_paths):
        file_name = os.path.basename(file_path)
        dst_file = os.path.join(dst_dir, file_name)
        if not os.path.exists(dst_file):
            copyfile(file_path, dst_file)
        else:
            print(f"Warning: file {dst_file} exists!")


def main(args):
    nid2samplenum = {
        "1": 641,
        "2": 3073,
        "3": 5958,
        "7": 44,
        "12": 1290,
        "16": 1,
        "17": 195,
        "19": 1846,
        "21": 2634,
        "32": 3243,
        "33": 828,
        "34": 2251,
        "35": 202,
    }

    all_file_paths = []
    all_annos = []
    for file_path in glob(os.path.join(args.data_dir, "*/*_norm.json")):
        if "20220720" in file_path:
            continue
        all_file_paths.append(file_path)
        all_annos.append(json.load(open(file_path)))
    print(f"Num of total files: {len(all_file_paths)}")

    def sample(nid, ratio=0.1, min_box_num=4):
        nid_file_paths = []
        for file_path, anno in zip(all_file_paths, all_annos):
            if anno["nid"] != nid or len(anno["lidar_objs"]) < min_box_num:
                continue
            in_range_cnt = 0
            for obj in anno["lidar_objs"]:
                box = np.array(obj["3d_box"])
                center_x, center_y, _ = box.mean(axis=0)
                dist = (center_x**2 + center_y**2) ** 0.5
                if dist >= 100 and dist < 150:
                    in_range_cnt += 1
            if in_range_cnt >= 0.3 * len(anno["lidar_objs"]):
                nid_file_paths.append(file_path)
        total_num = len(nid_file_paths)
        sample_num = int(total_num * ratio)
        print(f"nid: {nid}, sample {sample_num} files from {total_num} files")
        return random.sample(nid_file_paths, sample_num)

    all_test_files = set()
    for nid, samplenum in nid2samplenum.items():
        if samplenum > 2000:
            nid_test_files = sample(nid)
            all_test_files = all_test_files | set(nid_test_files)
            copy_files(nid_test_files, os.path.join(args.data_dir, "test"))
    print(len(all_test_files))

    all_train_files = set(all_file_paths) - all_test_files
    print(f"Total {len(all_train_files)} train files, {len(all_test_files)} test files.")
    copy_files(all_train_files, os.path.join(args.data_dir, "train"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/xdq/annotation")
    args = parser.parse_args()
    main(args)
