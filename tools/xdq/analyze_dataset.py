import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from mmdet3d.datasets.xdq_dataset import mapper


def get_all_files(data_dir):
    all_files = []
    for file_path in glob(os.path.join(data_dir, "*/*/*_norm.json")):
        if "20220720" in file_path or "test" in file_path or "train" in file_path:
            continue
        all_files.append(file_path)
    print(f"Num of total files: {len(all_files)}")
    return all_files


def print_bbox_range(data_dir):
    cnt = defaultdict(int)
    stride = 50
    for file_path in tqdm(get_all_files(data_dir)):
        anno = json.load(open(file_path))
        for obj in anno["lidar_objs"]:
            box = np.array(obj["3d_box"])
            center_x, center_y, _ = box.mean(axis=0)
            dist = (center_x**2 + center_y**2) ** 0.5
            cnt[int(dist / stride)] += 1

    print(f"num of boxes: {sum(cnt.values())}")
    for k, v in cnt.items():
        print(f"num of boxes in range [{k*50}, {(k+1)*50}]: {v}")


def plot_bbox_center(data_dir):
    cls_names = ["car", "truck", "pedestrian", "bicycle"]
    cls_cnts = [0, 0, 0, 0]
    fig, axs = plt.subplots(2, 2)
    for file_path in tqdm(get_all_files(data_dir)):
        anno = json.load(open(file_path))
        for obj in anno["lidar_objs"]:
            box = np.array(obj["3d_box"])
            if obj["type"] == "Unknown":
                continue
            cls_name = mapper[obj["type"]]
            if cls_name not in cls_names:
                continue
            center_x, center_y, _ = box.mean(axis=0)
            idx = cls_names.index(cls_name)
            cls_cnts[idx] += 1
            axs[idx // 2, idx % 2].plot(center_x, center_y, "o", color="blue", markersize=0.1)
    axs[0, 0].set_title(f"car ({cls_cnts[0]})", fontdict={"fontsize": 10})
    axs[0, 1].set_title(f"truck ({cls_cnts[1]})", fontdict={"fontsize": 10})
    axs[1, 0].set_title(f"pedestrian ({cls_cnts[2]})", fontdict={"fontsize": 10})
    axs[1, 1].set_title(f"bicycle ({cls_cnts[3]})", fontdict={"fontsize": 10})
    fig.tight_layout()
    plt.savefig("bbox_centers.png")


def plot_unknown_bbox(data_dir):
    cnt = 0
    for file_path in tqdm(get_all_files(data_dir)):
        anno = json.load(open(file_path))
        for obj in anno["lidar_objs"]:
            box = np.array(obj["3d_box"])
            if obj["type"] != "Unknown":
                continue
            cnt += 1
            center_x, center_y, _ = box.mean(axis=0)
            plt.plot(center_x, center_y, "o", color="blue", markersize=0.1)
    plt.title(f"num of unknown boxes: {cnt}")
    plt.savefig("unknown_bboxes.png")


def plot_hard_bbox(data_dir):
    cnt = 0
    for file_path in tqdm(get_all_files(data_dir)):
        anno = json.load(open(file_path))
        for obj in anno["lidar_objs"]:
            box = np.array(obj["3d_box"])
            if obj["num_pts"] >= 5:
                continue
            cnt += 1
            center_x, center_y, _ = box.mean(axis=0)
            plt.plot(center_x, center_y, "o", color="blue", markersize=0.1)
    plt.title(f"num of hard boxes: {cnt}")
    plt.savefig("hard_bboxes.png")


def print_nid_samplenum(data_dir):
    nid2samplenum = defaultdict(int)
    for file_path in tqdm(get_all_files(data_dir)):
        anno = json.load(open(file_path))
        nid = anno["nid"]
        nid2samplenum[nid] += 1
    print(nid2samplenum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/xdq")
    parser.add_argument("--print-bbox-range", action="store_true")
    parser.add_argument("--print-nid-samplenum", action="store_true")
    parser.add_argument("--plot-bbox-center", action="store_true")
    parser.add_argument("--plot-unknown-bbox", action="store_true")
    parser.add_argument("--plot-hard-bbox", action="store_true")
    args = parser.parse_args()

    if args.print_bbox_range:
        print_bbox_range(args.data_dir)
        exit(0)

    if args.print_nid_samplenum:
        print_nid_samplenum(args.data_dir)
        exit(0)

    if args.plot_bbox_center:
        plot_bbox_center(args.data_dir)
        exit(0)

    if args.plot_unknown_bbox:
        plot_unknown_bbox(args.data_dir)
        exit(0)

    if args.plot_hard_bbox:
        plot_hard_bbox(args.data_dir)
        exit(0)
