import argparse
import os
import math
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from mmdet3d.datasets.xdq_dataset import mapper

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


def get_all_files(data_dir, verbose=False):
    all_files = []
    for file_path in glob(os.path.join(data_dir, "annotation/*/*/*_norm.json")):
        if "20220720" in file_path or "test" in file_path or "train" in file_path:
            continue
        all_files.append(file_path)
    if verbose:
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
    nid_set = set()
    print("Counting nid num...")
    for file_path in tqdm(get_all_files(data_dir)):
        anno = json.load(open(file_path))
        nid_set.add(anno["nid"])
    nid_cnt = len(nid_set)
    print(f"Total nid num: {nid_cnt}")

    cols = 3
    rows = math.ceil(nid_cnt / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(9, 15))

    for i, nid in enumerate(nid_set):
        row_idx = i // cols
        col_idx = i % cols
        cnt = 0
        print(f"Processing nid: {nid}...")
        for file_path in tqdm(get_all_files(data_dir)):
            anno = json.load(open(file_path))
            if anno["nid"] != nid:
                continue
            for obj in anno["lidar_objs"]:
                if obj["num_pts"] >= 5:
                    continue
                box = np.array(obj["3d_box"])
                cnt += 1
                center_x, center_y, _ = box.mean(axis=0)
                axs[row_idx, col_idx].plot(center_x, center_y, "o", color="blue", markersize=0.1)
        axs[row_idx, col_idx].set_title(f"nid {nid}: {cnt}")
    fig.tight_layout()
    fig.savefig("hard_bboxes.png")


def print_nid_samplenum(data_dir):
    nid2samplenum = defaultdict(int)
    for file_path in tqdm(get_all_files(data_dir)):
        anno = json.load(open(file_path))
        nid = anno["nid"]
        nid2samplenum[nid] += 1
    print(nid2samplenum)


def create_frustum():
    # make grid in image plane
    ogfH, ogfW = (1080, 1920)
    fH, fW = ogfH // 8, ogfW // 8
    dbound = [22.0, 90.0, 1.0]
    ds = torch.arange(*dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
    D, _, _ = ds.shape
    xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

    # D x H x W x 3
    frustum = torch.stack((xs, ys, ds), -1)
    return frustum


def get_geometry(frustum, rots, trans, norm_offset):
    B, N, _ = trans.shape

    points = frustum.repeat(B, N, 1, 1, 1, 1).unsqueeze(-1)  # B x N x D x H x W x 3 x 1

    # cam_to_ego
    points = torch.cat(
        (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5
    )
    points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += trans.view(B, N, 1, 1, 1, 3)

    norm_offset = torch.Tensor(norm_offset).to(points.device)
    # After normalize, -3.1446 <= points_z <= 7.3388, is it normal?
    points += norm_offset

    return points


def plot_frustum(file_path):
    anno = json.load(open(file_path))
    rots, trans = [], []
    for camera_info in anno["cams"]:
        # camera intrinsics, pad to 4x4
        viewpad = np.eye(4).astype(np.float32)
        intrinsic = np.float32(camera_info["intrinsic"])
        viewpad[:3, :3] = intrinsic

        # lidar to camera
        extrinsic = np.float32(camera_info["extrinsic"])
        l2c_R, l2c_t = extrinsic[:3, :3], extrinsic[:3, 3:]
        lidar2camera = np.eye(4).astype(np.float32)
        lidar2camera[:3, :3] = l2c_R
        lidar2camera[:3, 3:] = l2c_t

        # lidar to image
        lidar2image = viewpad @ lidar2camera

        mat = torch.Tensor(lidar2image)
        rots.append(mat.inverse()[:3, :3])
        trans.append(mat.inverse()[:3, 3].view(-1))
    rots = torch.stack(rots, dim=0).unsqueeze(0)
    trans = torch.stack(trans, dim=0).unsqueeze(0)
    frustum = create_frustum()
    geom = get_geometry(frustum, rots, trans, norm_offsets[anno["nid"]])
    geom = geom.reshape(-1, 3).numpy()
    plt.plot(geom[:, 0], geom[:, 1], "o", markersize=0.01)
    plt.savefig("frustum.png")


def print_timestamp_nid_samplenum(data_dir):
    anno_dir = os.path.join(data_dir, "annotation")
    timestamps = os.listdir(anno_dir)
    for timestamp in timestamps:
        print(f"### {timestamp}")
        nid2num = defaultdict(int)
        for anno_file in glob(os.path.join(anno_dir, timestamp, "*/*_norm.json")):
            anno = json.load(open(anno_file))
            nid = anno["nid"]
            nid2num[nid] += 1
        print({k: nid2num[k] for k in sorted(nid2num)})


def print_avg_num_pts_per_range(data_dir):
    ranges = [[0, 50], [50, 100], [100, 150], [150, 999999]]
    class_cnt = [defaultdict(int) for _ in range(4)]
    avg_num_pts = [0, 0, 0, 0]
    count = [0, 0, 0, 0]
    total = 0
    for file_path in tqdm(get_all_files(data_dir)):
        test_kids = [
            "20220921/21/",
            "20220930/34/",
            "20221018/2/",
            "20221018/3/",
            "20221024/19/",
        ]
        valid = False
        for test_kid in test_kids:
            if test_kid in file_path:
                valid = True
                break
        if not valid:
            continue

        anno = json.load(open(file_path))
        for obj in anno["lidar_objs"]:
            box = np.array(obj["3d_box"])
            center_x, center_y, _ = box.mean(axis=0)
            for idx, range_i in enumerate(ranges):
                min_dist = range_i[0]
                max_dist = range_i[1]
                dist = (center_x**2 + center_y**2) ** 0.5
                if dist >= min_dist and dist <= max_dist:
                    break
            if obj["type"] == "Unknown" or obj["num_pts"] < 5:
                continue
            total += 1
            count[idx] += 1
            avg_num_pts[idx] += obj["num_pts"]
            class_cnt[idx][mapper[obj["type"]]] += 1
    avg_num_pts = [num_pts / (count[i] + 1e-5) for i, num_pts in enumerate(avg_num_pts)]
    print(f"Avg num pts: {avg_num_pts}")
    print(f"Count: {count}")
    print(f"Total boxes: {total}")
    print(f"Class count per range: {[dict(cnt) for cnt in class_cnt]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/xdq")
    parser.add_argument("--print-bbox-range", action="store_true")
    parser.add_argument("--print-nid-samplenum", action="store_true")
    parser.add_argument("--print-timestamp-nid-samplenum", action="store_true")
    parser.add_argument("--print-avg-num-pts-per-range", action="store_true")
    parser.add_argument("--plot-bbox-center", action="store_true")
    parser.add_argument("--plot-unknown-bbox", action="store_true")
    parser.add_argument("--plot-hard-bbox", action="store_true")
    parser.add_argument("--plot-frustum", default=None, type=str)
    args = parser.parse_args()

    if args.print_bbox_range:
        print_bbox_range(args.data_dir)
        exit(0)

    if args.print_nid_samplenum:
        print_nid_samplenum(args.data_dir)
        exit(0)

    if args.print_timestamp_nid_samplenum:
        print_timestamp_nid_samplenum(args.data_dir)
        exit(0)

    if args.print_avg_num_pts_per_range:
        print_avg_num_pts_per_range(args.data_dir)
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

    if args.plot_frustum:
        plot_frustum(args.plot_frustum)
        exit(0)
