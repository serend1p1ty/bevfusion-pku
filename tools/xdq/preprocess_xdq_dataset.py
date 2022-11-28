import argparse
import json
import numpy as np
import os
import re
from copy import deepcopy
from glob import glob
from tqdm import tqdm

# calculated by cal_xyz_offset(), used to normalize point cloud to origin-centered, z=-5 grounded
# norm_offsets = {
#     "1": [-4.46, -34.1, 45.75],
#     "2": [-23.32, 35.89, 45.75],
#     "3": [-4.64, 71.39, 45.34],
#     "7": [-17.48, -19.43, 45.19],
#     "12": [184.74, -78.32, 45.99],
#     "16": [33.16, 83.63, 46.91],
#     "17": [-18.74, -35.58, 45.41],
#     "19": [-12.4, -52.47, 46.21],
#     "21": [12.77, 71.73, 45.6],
#     "32": [48.12, -0.02, 45.93],
#     "33": [-53.45, -6.59, 45.97],
#     "34": [-8.23, -31.04, 45.71],
#     "35": [63.23, 49.72, 46.03],
# }
# hand-craft
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
        # 0720 is the old annotation version
        if "20220720" in anno_file:
            continue

        if "norm" in anno_file:
            continue

        anno = json.load(open(anno_file))
        nid = anno["nid"]

        if len(anno["lidar_objs"]) == 0:
            continue

        gt_boxes_corners = [
            np.asarray(obj["3d_box"], dtype=np.float64) for obj in anno["lidar_objs"]
        ]
        # (N, 8, 3) -> (Nx8, 3)
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
    anno_files = glob(os.path.join(anno_dir, "*/*/*.json"))
    invalid_files = []
    for anno_file in tqdm(anno_files):
        if "20220720" in anno_file:
            continue

        if "norm" in anno_file:
            continue

        anno = json.load(open(anno_file))
        nid = anno["nid"]
        norm_offset = norm_offsets[nid]

        if len(anno["lidar_objs"]) == 0:
            invalid_files.append(anno_file)
            continue

        pcd_relative_path = re.search(
            "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace("pcd", "npy")
        ).group()
        pcd_path = os.path.join(data_dir, pcd_relative_path)
        points = np.load(pcd_path)

        # normalize the coordinates of gt boxes
        norm_anno = deepcopy(anno)
        for i, obj in enumerate(anno["lidar_objs"]):
            box_corners = np.asarray(obj["3d_box"], dtype=np.float64)
            norm_box = box_corners + norm_offset
            obj["3d_box"] = norm_box.tolist()
            points_in_box = get_points_in_box(points, box_corners)
            obj["num_pts"] = points_in_box.shape[0]
            norm_anno["lidar_objs"][i] = obj

        save_path = anno_file.replace(".json", "_norm.json")
        json.dump(norm_anno, open(save_path, "w"), indent=4)

        # normalize the coordinates of point cloud
        points += norm_offset + [0.0]
        save_path = pcd_path.replace(".npy", "_norm.npy")
        np.save(save_path, points)
    print(invalid_files)


def fix_cam_lidar_mismatch(fix=False):
    anno_files = glob(os.path.join(anno_dir, "*/*/*_norm.json"))
    mismatch_num = 0
    # cam is more than lidar
    mismatch_more = []
    # cam is less than lidar
    mismatch_less = []
    for anno_file in tqdm(anno_files):
        anno = json.load(open(anno_file))
        pcd_relative_path = re.search(
            "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace(".pcd", "_norm.npy")
        ).group()
        pcd_path = os.path.join(data_dir, pcd_relative_path)
        points = np.load(pcd_path)
        assert points.shape[0] % 24000 == 0
        lidar_num = points.shape[0] / 24000
        cam_num = len(anno["cams"])
        if cam_num > lidar_num:
            mismatch_num += 1
            mismatch_more.append(anno_file)
        if cam_num < lidar_num:
            mismatch_num += 1
            mismatch_less.append(anno_file)
    print(f"Total mismatch_num: {mismatch_num}")
    print(f"##### {len(mismatch_more)} files, cam is more than LiDAR: {mismatch_more}")
    print(f"##### {len(mismatch_less)} files, cam is less than LiDAR: {mismatch_less}")

    if fix:
        for anno_file in mismatch_less + mismatch_more:
            os.remove(anno_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-offset", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--fix-mismatch", action="store_true")
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

    if args.fix_mismatch:
        fix_cam_lidar_mismatch()
        exit(0)
