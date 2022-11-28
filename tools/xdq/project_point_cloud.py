import argparse
import json
import re
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def plot_anno_files(anno_files):
    num_files = len(anno_files)
    ncols = 3
    nrows = math.ceil(num_files / 3)
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 40), squeeze=False)
    for anno_id, anno_file in enumerate(tqdm(anno_files)):
        anno = json.load(open(anno_file))
        nid = anno["nid"]
        norm_offset = norm_offsets[nid]
        row_idx, col_idx = anno_id // ncols, anno_id % ncols

        pcd_relative_path = re.search(
            "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace("pcd", "npy")
        ).group()
        pcd_path = os.path.join(data_dir, pcd_relative_path)
        points = np.load(pcd_path)

        colors = ["blue", "yellow", "red", "black"]
        title_str = ""
        for cam_id, camera_info in enumerate(anno["cams"]):
            img_relative_path = re.search(
                "[^/?]+/[^/?]+/[^/?]+jpg", camera_info["img_path"]
            ).group()
            img_path = os.path.join(data_dir, img_relative_path)

            intrinsic = np.float32(camera_info["intrinsic"])
            extrinsic = np.float32(camera_info["extrinsic"])
            l2c_R, l2c_t = extrinsic[:3, :3], extrinsic[:3, 3:]

            img = cv2.imread(img_path)
            img_name = os.path.basename(img_path)
            cam_points = l2c_R @ points[:, :3].T + l2c_t

            # At least min_z meters in front of the camera
            min_z = 3
            idxs = cam_points[2, :] > min_z
            points = points[idxs, :3]
            cam_points = cam_points[:, idxs]

            proj_points = intrinsic @ cam_points
            proj_points = (proj_points[:2] / proj_points[2:]).T
            proj_points = proj_points.reshape(-1, 2)

            cnt = 0
            h, w = img.shape[:2]
            min_campt_z = 1e9
            max_campt_z = -1e9
            xs, ys, cs = [], [], []
            for k in range(proj_points.shape[0]):
                x = int(proj_points[k, 0])
                y = int(proj_points[k, 1])
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                cnt += 1
                cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
                min_campt_z = min(min_campt_z, cam_points[2, k])
                max_campt_z = max(max_campt_z, cam_points[2, k])
                xs.append(points[k, 0] + norm_offset[0])
                ys.append(points[k, 1] + norm_offset[1])
                cs.append(colors[cam_id])
            axs[row_idx, col_idx].scatter(xs, ys, s=0.1, c=cs)
            if not os.path.isdir("vis_proj_pc"):
                os.makedirs("vis_proj_pc")
            print(f"{nid}_{img_name}: {cnt}")
            cv2.imwrite(f"vis_proj_pc/{nid}_{img_name}", img)
            title_str += f"{camera_info['cid']}: [{min_campt_z:.2f}, {max_campt_z:.2f}]\n"
        axs[row_idx, col_idx].set_title(title_str)
    token = pcd_path.replace("/", "_").replace("data_xdq_data_", "")
    fig.tight_layout()
    fig.savefig(f"vis_proj_pc/bev_{token}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-files", nargs="+", type=str)
    parser.add_argument("--dataset-dir", default="data/xdq")
    args = parser.parse_args()
    anno_dir = os.path.join(args.dataset_dir, "annotation")
    data_dir = os.path.join(args.dataset_dir, "data")
    plot_anno_files(args.anno_files)
