import argparse
import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copyfile


def filter_range(file_paths, min_dist, max_dist):
    print(f"Filtering [{min_dist}m, {max_dist}m) samples...")
    ret_files = []
    for file_path in tqdm(file_paths):
        anno = json.load(open(file_path))
        in_range_cnt = 0
        for obj in anno["lidar_objs"]:
            box = np.array(obj["3d_box"])
            center_x, center_y, _ = box.mean(axis=0)
            dist = (center_x**2 + center_y**2) ** 0.5
            if dist >= min_dist and dist < max_dist:
                in_range_cnt += 1
        if in_range_cnt > 0.3 * len(anno["lidar_objs"]):
            ret_files.append(file_path)
    return ret_files


def copy_files(file_paths, dst_dir):
    print("Copying files...")
    os.makedirs(dst_dir, exist_ok=True)
    for file_path in tqdm(file_paths):
        file_name = os.path.basename(file_path)
        copyfile(file_path, os.path.join(dst_dir, file_name))


def main(args):
    # choose 100m-150m samples from 20221016/20221018/20221024/20221026
    all_files = []
    for file_path in glob(os.path.join(args.data_dir, "*/*_norm.json")):
        if "202210" not in file_path:
            continue
        all_files.append(file_path)
    test_files = filter_range(all_files, 100, 150)
    print(f"test 100-150m: {len(test_files)}")

    all_files = []
    for file_path in glob(os.path.join(args.data_dir, "*/*_norm.json")):
        if "20220720" in file_path:
            continue
        # skip pre-defined testset
        if "20221024" in file_path or "20221026" in file_path:
            continue
        all_files.append(file_path)
    train_files = set(all_files) - set(test_files)
    print(f"train: {len(train_files)}")

    copy_files(test_files, os.path.join(args.data_dir, "test_100-150m"))
    copy_files(train_files, os.path.join(args.data_dir, "train"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/xdq/annotation")
    args = parser.parse_args()
    main(args)
