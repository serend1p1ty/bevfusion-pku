import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from copy import deepcopy
from glob import glob
from tqdm import tqdm
from mmdet3d.core.bbox import box_np_ops as box_np_ops


def get_anno_files(selected_timestamps=None, norm=False):
    anno_files = glob(os.path.join(anno_dir, "*/*/*.json"))
    valid_anno_files = []
    for anno_file in anno_files:
        # 0720 is the old annotation version
        if "20220720" in anno_file:
            continue
        if norm is False:
            if "norm" in anno_file:
                continue
        else:
            if "norm" not in anno_file:
                continue
        if selected_timestamps is not None:
            is_valid = False
            for timestamp in selected_timestamps:
                if timestamp in anno_file:
                    is_valid = True
                    break
            if not is_valid:
                continue
        valid_anno_files.append(anno_file)
    return valid_anno_files


def get_points_in_box(points, box):
    box_min_xyz = box.min(axis=0).reshape(-1)
    box_max_xyz = box.max(axis=0).reshape(-1)

    val_flag_x = np.logical_and(points[:, 0] >= box_min_xyz[0], points[:, 0] < box_max_xyz[0])
    val_flag_y = np.logical_and(points[:, 1] >= box_min_xyz[1], points[:, 1] < box_max_xyz[1])
    val_flag_z = np.logical_and(points[:, 2] >= box_min_xyz[2], points[:, 2] < box_max_xyz[2])
    val_flag_merge = np.logical_and(np.logical_and(val_flag_x, val_flag_y), val_flag_z)

    box_points = points[val_flag_merge]

    return box_points


def corners2xyzwlhr(corners):
    """
      1 -------- 0
     /|         /|
    2 -------- 3 .
    | |        | |
    . 5 -------- 4
    |/         |/
    6 -------- 7

    Args:
        corners (np.array): (8, 3), [x, y, z] in lidar coords.

    Returns:
        box (np.array): (7,), [x, y, z, w, l, h, r] in lidar coords.
    """
    corners = np.array(corners)
    height_group = [(4, 0), (5, 1), (6, 2), (7, 3)]
    width_group = [(4, 5), (7, 6), (0, 1), (3, 2)]
    length_group = [(4, 7), (5, 6), (0, 3), (1, 2)]
    vector_group = [(4, 7), (5, 6), (0, 3), (1, 2)]
    height, width, length = 0.0, 0.0, 0.0
    vector = np.zeros(2, dtype=np.float32)
    for index_h, index_w, index_l, index_v in zip(
        height_group, width_group, length_group, vector_group
    ):
        height += np.linalg.norm(corners[index_h[0], :] - corners[index_h[1], :])
        width += np.linalg.norm(corners[index_w[0], :] - corners[index_w[1], :])
        length += np.linalg.norm(corners[index_l[0], :] - corners[index_l[1], :])
        vector[0] += (corners[index_v[0], :] - corners[index_v[1], :])[0]
        vector[1] += (corners[index_v[0], :] - corners[index_v[1], :])[1]

    height, width, length = height * 1.0 / 4, width * 1.0 / 4, length * 1.0 / 4
    rotation_y = -np.arctan2(vector[1], vector[0]) - np.pi / 2

    center_point = corners.mean(axis=0)
    box = np.concatenate([center_point, np.array([width, length, height, rotation_y])])

    return box


def get_points_in_rbox(points, box_corners):
    box = corners2xyzwlhr(box_corners)
    idxs = box_np_ops.points_in_rbbox(points, box[None, :], origin=(0.5, 0.5, 0.5)).squeeze()
    box_points = points[idxs]
    return box_points


def cal_norm_offset():
    nid2offsets = {}
    anno_files = get_anno_files()
    for anno_file in tqdm(anno_files):
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


def norm_coord(selected_timestamps=None):
    print(">>> Normalize coordinates...")
    anno_files = get_anno_files(selected_timestamps)
    invalid_files = []
    for anno_file in tqdm(anno_files):
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
            norm_corners = box_corners + norm_offset
            obj["3d_box"] = norm_corners.tolist()
            points_in_box = get_points_in_rbox(points, box_corners)
            obj["num_pts"] = points_in_box.shape[0]
            norm_anno["lidar_objs"][i] = obj

        save_path = anno_file.replace(".json", "_norm.json")
        json.dump(norm_anno, open(save_path, "w"), indent=4)

        # normalize the coordinates of point cloud
        points += norm_offset + [0.0]
        save_path = pcd_path.replace(".npy", "_norm.npy")
        np.save(save_path, points)
    print(f"{len(invalid_files)} invalid files: {invalid_files[:20]}")


def fix_cid(selected_timestamps=None):
    print(">>> Fixing incorrect cid...")
    anno_files = get_anno_files(selected_timestamps)
    cid_err_files = []
    for anno_file in tqdm(anno_files):
        anno = json.load(open(anno_file))
        # check cid correctness
        for cam_info in anno["cams"]:
            cid = cam_info["cid"]
            img_url = cam_info["img_path"]
            res = re.search("[^/?]+/([^/?]+)/([^/?]+)-[^/?]+jpg", img_url)
            img_path = os.path.join(args.dataset_dir, "data", res.group())
            assert os.path.exists(img_path)
            nid, kid = res.group(1), res.group(2)
            assert int(nid) <= 35 and int(kid) <= 4
            if cid != f"{nid}-{kid}":
                cid_err_files.append(anno_file)
                break
    print(f"{len(cid_err_files)} files, cid is incorrect: {cid_err_files[:20]}")
    for anno_file in cid_err_files:
        anno = json.load(open(anno_file))
        for i, cam_info in enumerate(anno["cams"]):
            cid = cam_info["cid"]
            img_url = cam_info["img_path"]
            res = re.search("[^/?]+/([^/?]+)/([^/?]+)-[^/?]+jpg", img_url)
            img_path = os.path.join(args.dataset_dir, "data", res.group())
            assert os.path.exists(img_path)
            nid, kid = res.group(1), res.group(2)
            assert int(nid) <= 35 and int(kid) <= 4
            correct_cid = f"{nid}-{kid}"
            if cid != correct_cid:
                anno["cams"][i]["cid"] = correct_cid
        with open(anno_file, "w") as f:
            json.dump(anno, f, indent=4)


def remove_cam_lidar_mismatch(selected_timestamps=None):
    print(">>> Fixing cam LiDAR mismatch...")
    anno_files = get_anno_files(selected_timestamps)
    mismatch_files = []
    for anno_file in tqdm(anno_files):
        anno = json.load(open(anno_file))

        # check cam-lidar mismatch
        pcd_relative_path = re.search(
            "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace(".pcd", ".npy")
        ).group()
        pcd_path = os.path.join(data_dir, pcd_relative_path)
        points = np.load(pcd_path)
        assert points.shape[0] % 24000 == 0
        lidar_num = points.shape[0] / 24000
        cam_num = len(anno["cams"])
        if cam_num != lidar_num:
            mismatch_files.append(anno_file)

    print(f"{len(mismatch_files)} files, cam num is mismatch with LiDAR num: {mismatch_files[:20]}")
    for anno_file in mismatch_files:
        os.remove(anno_file)


def attach_min_camz(selected_timestamps=None, plot=True, plot_minz=10, plot_maxz=150):
    print(">>> Attaching min camz...")
    anno_files = get_anno_files(selected_timestamps, norm=True)
    nid_set = set()
    for anno_file in tqdm(anno_files):
        anno = json.load(open(anno_file))
        nid_set.add(anno["nid"])
        norm_offset = norm_offsets[anno["nid"]]
        for i, obj in enumerate(anno["lidar_objs"]):
            box_corners = np.asarray(obj["3d_box"], dtype=np.float64)
            box_corners -= norm_offset
            min_camz = 1e5
            for camera_info in anno["cams"]:
                intrinsic = np.float32(camera_info["intrinsic"])
                extrinsic = np.float32(camera_info["extrinsic"])
                l2c_R, l2c_t = extrinsic[:3, :3], extrinsic[:3, 3:]
                # (3, 8)
                cam_points = l2c_R @ box_corners.T + l2c_t

                # check if the box is in the front of camera
                _, _, camz = cam_points.mean(axis=1)
                if camz <= 0:
                    continue

                # (8, 3)
                img_points = (intrinsic @ cam_points).T
                img_points[:, :2] /= img_points[:, 2:]

                # check if the box is visible in image
                h, w = 1080, 1920
                min_x, min_y = img_points[:, :2].min(axis=0)
                max_x, max_y = img_points[:, :2].min(axis=0)
                if min_x >= w or max_x < 0 or min_y >= h or max_y < 0:
                    continue

                # import cv2
                # img_relative_path = re.search(
                #     "[^/?]+/[^/?]+/[^/?]+jpg", camera_info["img_path"]
                # ).group()
                # img_path = os.path.join(data_dir, img_relative_path)
                # img = cv2.imread(img_path)
                # for img_point in img_points:
                #     img_point = img_point.astype(np.int)
                #     cv2.circle(img, img_point[:2], 10, (0, 255, 255), -1)
                # cv2.imwrite("test.png", img)
                # print(camz)
                # input()

                min_camz = min(min_camz, camz)
            min_camz = -1 if min_camz == 1e5 else min_camz
            anno["lidar_objs"][i]["min_camz"] = min_camz
        with open(anno_file, "w") as f:
            json.dump(anno, f, ensure_ascii=False, indent=4)

    if not plot:
        return

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
        for anno_file in tqdm(anno_files):
            anno = json.load(open(anno_file))
            if anno["nid"] != nid:
                continue
            for obj in anno["lidar_objs"]:
                if obj["min_camz"] < plot_minz or obj["min_camz"] > plot_maxz:
                    continue
                box = np.array(obj["3d_box"])
                cnt += 1
                center_x, center_y, _ = box.mean(axis=0)
                axs[row_idx, col_idx].plot(center_x, center_y, "o", color="blue", markersize=0.1)
        axs[row_idx, col_idx].set_title(f"nid {nid}: {cnt}")
    fig.tight_layout()
    fig.savefig("cam_train_bboxes.png")


def remove_invalid_samples(selected_timestamps=None):
    print(">>> Remove invalid samples...")
    anno_files = get_anno_files(selected_timestamps)
    delete_files = []
    for anno_file in anno_files:
        assert "norm" not in anno_file
        norm_anno_file = anno_file.replace(".json", "_norm.json")
        if not os.path.exists(norm_anno_file):
            anno = json.load(open(anno_file))
            assert len(anno["lidar_objs"]) == 0
            delete_files.append(anno_file)

    pcd_files = glob(os.path.join(data_dir, "*/*/*.pcd"))
    for pcd_file in pcd_files:
        norm_npy_file = pcd_file.replace(".pcd", "_norm.npy")
        # There may be several reasons why a pcd file does not have a corresponding _norm.npy file:
        # 1. The pcd file is not labeled.
        # 2. The annotation of the pcd file is invalid, e.g., cam-LiDAR mismatch, empty LiDAR objs.
        if not os.path.exists(norm_npy_file):
            delete_files.append(pcd_file)
            delete_files.append(pcd_file.replace(".pcd", ".npy"))

            path_prefix = "/".join(pcd_file.split("/")[:-1])
            token = pcd_file.split("/")[-1].replace(".pcd", "")
            img_files = [
                os.path.join(path_prefix, f"1-{token}.jpg"),
                os.path.join(path_prefix, f"2-{token}.jpg"),
                os.path.join(path_prefix, f"3-{token}.jpg"),
                os.path.join(path_prefix, f"4-{token}.jpg"),
            ]
            has_valid_img = False
            for img_file in img_files:
                if os.path.exists(img_file):
                    has_valid_img = True
                    delete_files.append(img_file)
            assert has_valid_img
    print(f"{len(delete_files)} delete files: {delete_files[:20]}")
    for delete_file in delete_files:
        os.remove(delete_file)


def remove_null_type_boxes(selected_timestamps=None):
    print(">>> Remove null type boxes...")
    anno_files = get_anno_files(selected_timestamps)
    num_null_boxes = 0
    for anno_file in tqdm(anno_files):
        anno = json.load(open(anno_file))
        delete_idxs = []
        for i, obj in enumerate(anno["lidar_objs"]):
            if obj["type"] is None:
                num_null_boxes += 1
                delete_idxs.append(i)
        if len(delete_idxs) == 0:
            continue
        new_objs = []
        for i, obj in enumerate(anno["lidar_objs"]):
            if i in delete_idxs:
                continue
            new_objs.append(obj)
        anno["lidar_objs"] = new_objs
        with open(anno_file, "w") as f:
            json.dump(anno, f, ensure_ascii=False, indent=4)
    print(f"Total {num_null_boxes} null boxes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-norm-offset", action="store_true")
    parser.add_argument("--norm-coord", action="store_true")
    parser.add_argument("--remove-cam-lidar-mismatch", action="store_true")
    parser.add_argument("--fix-cid", action="store_true")
    parser.add_argument("--attach-min-camz", action="store_true")
    parser.add_argument("--remove-invalid-samples", action="store_true")
    parser.add_argument("--remove-null-type-boxes", action="store_true")
    parser.add_argument("--full-process", action="store_true")
    parser.add_argument("--incremental-process", nargs="+", type=str, default=None)
    parser.add_argument("--dataset-dir", default="data/xdq")
    args = parser.parse_args()

    anno_dir = os.path.join(args.dataset_dir, "annotation")
    data_dir = os.path.join(args.dataset_dir, "data")

    # Used to normalize point cloud to origin-centered, z=-5 grounded.
    norm_offsets = {
        "1": [22, -70, 45.75],
        "2": [-23.32, 35.89, 45.75],
        "3": [-25, 98, 45.34],
        "5": [30, -110, 45.56],
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

    if args.full_process or args.incremental_process:
        selected_timestamps = args.incremental_process if args.incremental_process else None
        # 1. Remove those samples whose cam num != LiDAR num,
        # which may be caused by the bug of our data processing script.
        remove_cam_lidar_mismatch(selected_timestamps)
        # 2. Fix the incorrect cid, which is caused by a longmao bug.
        fix_cid(selected_timestamps)
        # 3. Remove those boxes whose type=null, which is caused by a longmao bug.
        remove_null_type_boxes(selected_timestamps)
        # 4. Normalize LiDAR points and annotated bboxes by
        # the calculated or hand-crafted norm_offsets.
        norm_coord(selected_timestamps)
        # 5. Remove invalid samples to save disk usage.
        remove_invalid_samples(selected_timestamps)
        # 6. Calculate the shortest distance from each box to all cameras,
        # and attach the distance to annotation file.
        attach_min_camz(selected_timestamps)
        exit(0)

    if args.cal_norm_offset:
        cal_norm_offset()
        exit(0)

    if args.norm_coord:
        norm_coord()
        exit(0)

    if args.remove_cam_lidar_mismatch:
        remove_cam_lidar_mismatch()
        exit(0)

    if args.fix_cid:
        fix_cid()
        exit(0)

    if args.remove_invalid_samples:
        remove_invalid_samples()
        exit(0)

    if args.remove_null_type_boxes:
        remove_null_type_boxes()
        exit(0)

    if args.attach_min_camz:
        attach_min_camz()
        exit(0)
