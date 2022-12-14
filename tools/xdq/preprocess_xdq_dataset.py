import argparse
import json
import numpy as np
import os
import re
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from glob import glob
from tqdm import tqdm

# Data process:
# 1. Calculate norm offsets or hand-craft
# 2. Normalize LiDAR points and annotation bboxes
# 3. Check Camera-LiDAR mismatch
# 4. Check cid-ImgURL mismatch

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


def cal_xyz_offset():
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


def normalize_coordinate():
    anno_files = get_anno_files()
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
    print(f"###### {len(invalid_files)} invalid files: {invalid_files[:100]}")


def fix_cam_lidar_mismatch(fix=False):
    anno_files = get_anno_files(norm=True)
    mismatch_num = 0
    # cam is more than lidar
    mismatch_more = []
    # cam is less than lidar
    mismatch_less = []
    # cid is incorrect
    cid_err_files = []
    for anno_file in tqdm(anno_files):
        anno = json.load(open(anno_file))

        # check cam-lidar mismatch
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
    print(f"Total mismatch_num: {mismatch_num}")
    print(f"##### {len(mismatch_more)} files, cam is more than LiDAR: {mismatch_more}")
    print(f"##### {len(mismatch_less)} files, cam is less than LiDAR: {mismatch_less}")
    print(f"##### {len(cid_err_files)} files, cid is incorrect: {cid_err_files}")

    if fix:
        for anno_file in mismatch_less + mismatch_more:
            os.remove(anno_file)
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
                    print(f"Fixed {cid} to {correct_cid}")
                    anno["cams"][i]["cid"] = correct_cid
            with open(anno_file, "w") as f:
                json.dump(anno, f, indent=4)


def attach_min_camz(plot=True, minz=10, maxz=150):
    anno_files = get_anno_files(norm=True)
    nid_set = set()
    print("Attach min camz......")
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
                # [3, 8]
                cam_points = l2c_R @ box_corners.T + l2c_t

                # check if the box is in the front of camera
                _, _, camz = cam_points.mean(axis=1)
                if camz <= 0:
                    continue

                # [8, 3]
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
                if obj["min_camz"] < minz or obj["min_camz"] > maxz:
                    continue
                box = np.array(obj["3d_box"])
                cnt += 1
                center_x, center_y, _ = box.mean(axis=0)
                axs[row_idx, col_idx].plot(center_x, center_y, "o", color="blue", markersize=0.1)
        axs[row_idx, col_idx].set_title(f"nid {nid}: {cnt}")
    fig.tight_layout()
    fig.savefig("cam_train_bboxes.png")


def remove_invalid_samples():
    anno_files = get_anno_files()
    delete_files = []
    for anno_file in anno_files:
        assert "norm" not in anno_file
        norm_anno_file = anno_file.replace(".json", "_norm.json")
        if not os.path.exists(norm_anno_file):
            anno = json.load(open(anno_file))
            pcd_relative_path = re.search(
                "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace(".pcd", ".npy")
            ).group()
            pcd_path = os.path.join(data_dir, pcd_relative_path)
            points = np.load(pcd_path)
            assert points.shape[0] % 24000 == 0
            lidar_num = points.shape[0] / 24000
            assert len(anno["lidar_objs"]) == 0 or len(anno["cams"]) != lidar_num
            delete_files.append(anno_file)

    pcd_files = glob(os.path.join(data_dir, "*/*/*.pcd"))
    for pcd_file in pcd_files:
        norm_npy_file = pcd_file.replace(".pcd", "_norm.npy")
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
            for img_file in img_files:
                if os.path.exists(img_file):
                    delete_files.append(img_file)
    print(f"{len(delete_files)} delete files: {delete_files[:100]}")
    for delete_file in delete_files:
        os.remove(delete_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-offset", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--fix-mismatch", action="store_true")
    parser.add_argument("--attach-min-camz", action="store_true")
    parser.add_argument("--remove-invalid-samples", action="store_true")
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

    if args.remove_invalid_samples:
        remove_invalid_samples()
        exit(0)

    if args.attach_min_camz:
        attach_min_camz()
        exit(0)
