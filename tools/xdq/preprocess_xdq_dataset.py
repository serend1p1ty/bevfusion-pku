import argparse
import json
import numpy as np
import os
import re
from copy import deepcopy
from glob import glob
from tqdm import tqdm

# 由cal_xyz_offset()计算得到，用来将点云归一化到以原点为中心，z=-5为接地面
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


def get_points_in_box(points, box):
    box_min_xyz = box.min(axis=0).reshape(-1)
    box_max_xyz = box.max(axis=0).reshape(-1)

    val_flag_x = np.logical_and(points[:, 0] >= box_min_xyz[0], points[:, 0] < box_max_xyz[0])
    val_flag_y = np.logical_and(points[:, 1] >= box_min_xyz[1], points[:, 1] < box_max_xyz[1])
    val_flag_z = np.logical_and(points[:, 2] >= box_min_xyz[2], points[:, 2] < box_max_xyz[2])
    val_flag_merge = np.logical_and(np.logical_and(val_flag_x, val_flag_y), val_flag_z)

    box_points = points[val_flag_merge]

    return box_points


def cal_xyz_offset():
    nid2offsets = {}
    anno_files = glob(os.path.join(anno_dir, "*/*.json"))
    for anno_file in tqdm(anno_files):
        # 0720是老版本标注格式
        if "20220720" in anno_file:
            continue

        if "norm" in anno_file:
            continue

        anno = json.load(open(anno_file))
        nid = anno["nid"]

        pcd_relative_path = re.search(
            "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace("pcd", "npy")
        ).group()
        pcd_path = os.path.join(data_dir, pcd_relative_path)
        # (N, 4)
        points = np.load(pcd_path)

        gt_boxes_corners = []
        for obj in anno["lidar_objs"]:
            # (8, 3)
            box_corners = np.asarray(obj["3d_box"], dtype=np.float64)
            # 过滤掉LiDAR点数目小于5的目标
            points_in_box = get_points_in_box(points, box_corners)
            if points_in_box.shape[0] < 5:
                continue
            gt_boxes_corners.append(box_corners)
        # 当前帧所有目标的LiDAR点数目都小于5
        if len(gt_boxes_corners) == 0:
            continue
        # (M, 8, 3) -> (Mx8, 3)
        gt_boxes_corners = np.asarray(gt_boxes_corners).reshape(-1, 3)

        boxes_min_xyz = gt_boxes_corners.min(axis=0).reshape(-1)
        boxes_max_xyz = gt_boxes_corners.max(axis=0).reshape(-1)

        offset = (
            (boxes_min_xyz[0] + boxes_max_xyz[0]) / 2,  # center x
            (boxes_min_xyz[1] + boxes_max_xyz[1]) / 2,  # center y
            boxes_min_xyz[2],  # min z
        )

        if nid not in nid2offsets:
            nid2offsets[nid] = [offset]
        else:
            nid2offsets[nid].append(offset)

    ret_offsets = {}
    for nid, offsets in nid2offsets.items():
        x_offset = round(0 - np.mean([offset[0] for offset in offsets]), 2)
        y_offset = round(0 - np.mean([offset[1] for offset in offsets]), 2)
        z_offset = round(-5 - np.mean([offset[2] for offset in offsets]), 2)
        ret_offsets[nid] = (x_offset, y_offset, z_offset)
    print(json.dumps(ret_offsets, indent=4))


def normalize_coordinate():
    total_box_cnt = invalid_box_cnt = 0
    anno_files = glob(os.path.join(anno_dir, "*/*.json"))
    for anno_file in tqdm(anno_files):
        if "20220720" in anno_file:
            continue

        if "norm" in anno_file:
            continue

        anno = json.load(open(anno_file))
        nid = anno["nid"]
        norm_offset = norm_offsets[nid]

        pcd_relative_path = re.search(
            "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace("pcd", "npy")
        ).group()
        pcd_path = os.path.join(data_dir, pcd_relative_path)
        points = np.load(pcd_path)

        # 归一化标注文件中的gt box坐标
        norm_anno = deepcopy(anno)
        norm_anno["lidar_objs"] = []
        for obj in anno["lidar_objs"]:
            if obj["type"] == "Unknown":
                continue
            box_corners = np.asarray(obj["3d_box"], dtype=np.float64)
            total_box_cnt += 1
            # 过滤掉LiDAR点数目小于5的目标
            points_in_box = get_points_in_box(points, box_corners)
            if points_in_box.shape[0] < 5:
                invalid_box_cnt += 1
                continue
            norm_box = box_corners + norm_offset
            obj["3d_box"] = norm_box.tolist()
            norm_anno["lidar_objs"].append(obj)
        # 当前帧所有目标的LiDAR点数目都小于5
        if len(norm_anno["lidar_objs"]) == 0:
            continue

        save_path = anno_file.replace(".json", "_norm.json")
        json.dump(norm_anno, open(save_path, "w"), indent=4)

        # 归一化点云坐标
        points += norm_offset + [0.0]
        save_path = pcd_path.replace(".npy", "_norm.npy")
        np.save(save_path, points)
    print(f"invalid boxes: {invalid_box_cnt}, total boxes: {total_box_cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-offset", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--dataset-dir", default="data/xdq")
    args = parser.parse_args()

    anno_dir = os.path.join(args.dataset_dir, "annotation")
    data_dir = os.path.join(args.dataset_dir, "data")

    if args.cal_offset:
        cal_xyz_offset()
        exit(0)

    if args.norm:
        normalize_coordinate()
        exit(0)
