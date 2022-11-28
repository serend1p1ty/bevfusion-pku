import argparse
import os
import re
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from mmdet3d.core.utils import visualize_lidar
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets.xdq_dataset import load_gt_bboxes_info, XdqDataset


def main(args):
    infos = []
    mismatch_num = 0
    for file_path in tqdm(glob(os.path.join(anno_dir, "*/*/*_norm.json"))):
        if (
            "20220930/34" not in file_path
            and "20221018/2" not in file_path
            and "20221018/3" not in file_path
            and "20221018/21" not in file_path
            and "20221024/19" not in file_path
        ):
            continue

        debug = False
        if debug:
            if file_path != "data/xdq/annotation/20221018/2/1662456528.199593306_norm.json":
                continue

        anno = json.load(open(file_path))

        pcd_relative_path = re.search(
            "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace(".pcd", "_norm.npy")
        ).group()
        pcd_path = os.path.join(data_dir, pcd_relative_path)
        points = np.load(pcd_path)
        num_pts_per_kid = 24000
        if points.shape[0] != len(anno["cams"]) * num_pts_per_kid:
            mismatch_num += 1
            continue
        kid_infos = []
        for i, cam in enumerate(anno["cams"]):
            kid_info = dict(kid=cam["cid"])
            kid_info["points"] = points[i * num_pts_per_kid : (i + 1) * num_pts_per_kid]
            kid_infos.append(kid_info)

        if debug:
            classes = XdqDataset.CLASSES
            gt_bboxes, gt_labels, _ = load_gt_bboxes_info(anno, False, False)
            gt_labels = [classes.index(l) for l in gt_labels]
            gt_bboxes = LiDARInstance3DBoxes(gt_bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
            visualize_lidar(
                "lidar.png",
                points,
                bboxes=gt_bboxes,
                labels=gt_labels,
                classes=classes,
                xlim=(-160, 160),
                ylim=(-160, 160),
            )
            for i, kid_info in enumerate(kid_infos):
                visualize_lidar(
                    f"cam-{i}.png",
                    kid_info["points"],
                    bboxes=gt_bboxes,
                    labels=gt_labels,
                    classes=classes,
                    xlim=(-160, 160),
                    ylim=(-160, 160),
                )

        min_intersect_num = 3
        intersect_obj_cnt = 0
        dist_thr = 160
        for obj in anno["lidar_objs"]:
            box_corners = np.asarray(obj["3d_box"], dtype=np.float64)
            center_x, center_y, _ = box_corners.mean(axis=0)
            if (
                center_x < -dist_thr
                or center_x > dist_thr
                or center_y < -dist_thr
                or center_y > dist_thr
            ):
                continue
            if obj["num_pts"] < 5:
                continue
            if obj["type"] == "Unknown":
                continue
            cnt = 0
            for kid_info in kid_infos:
                kid_points = kid_info["points"]
                min_x, min_y, _, _ = kid_points.min(axis=0)
                max_x, max_y, _, _ = kid_points.max(axis=0)
                x_flag = np.logical_and(box_corners[:, 0] >= min_x, box_corners[:, 0] <= max_x)
                y_flag = np.logical_and(box_corners[:, 1] >= min_y, box_corners[:, 1] <= max_y)
                valid_flag = np.logical_and(x_flag, y_flag)
                if valid_flag.sum() > 0:
                    cnt += 1
            if cnt >= min_intersect_num:
                intersect_obj_cnt += 1
        infos.append((intersect_obj_cnt, file_path))
    infos = sorted(infos)[::-1]
    np.save("lidar_intersection.npy", infos[:200])
    print(infos[:200])
    print(f"Info num: {len(infos)}")
    print(f"Mismatch num: {mismatch_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/xdq")
    args = parser.parse_args()
    anno_dir = os.path.join(args.dataset_dir, "annotation")
    data_dir = os.path.join(args.dataset_dir, "data")
    main(args)
