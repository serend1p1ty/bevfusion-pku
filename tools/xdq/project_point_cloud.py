import argparse
import json
import re
import os
import cv2
import math
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from mmdet3d.core.utils.gaussian import generate_guassian_depth_target

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


def project_lidar(anno_files):
    num_files = len(anno_files)
    ncols = 3
    nrows = math.ceil(num_files / ncols)
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
            if not os.path.isdir("vis_proj_lidar"):
                os.makedirs("vis_proj_lidar")
            print(f"{nid}_{img_name}: {cnt}")
            cv2.imwrite(f"vis_proj_lidar/{nid}_{img_name}", img)
            title_str += f"{camera_info['cid']}: [{min_campt_z:.2f}, {max_campt_z:.2f}]\n"
        axs[row_idx, col_idx].set_title(title_str)
    token = pcd_path.replace("/", "_").replace("data_xdq_data_", "")
    fig.tight_layout()
    fig.savefig(f"vis_proj_lidar/bev_{token}.png")


def project_map(anno_files):
    cid_to_map_points = {}
    out_dir = "vis_proj_map"
    for anno_file in tqdm(anno_files):
        timestamp = re.search("([^/?]+)/[^/?]+/[^/?]+_norm.json", anno_file).group(1)
        anno = json.load(open(anno_file))
        nid = anno["nid"]

        for camera_info in anno["cams"]:
            img_relative_path = re.search(
                "[^/?]+/[^/?]+/[^/?]+jpg", camera_info["img_path"]
            ).group()
            img_path = os.path.join(data_dir, img_relative_path)
            cid = camera_info["cid"]

            calib_file = os.path.join(args.dataset_dir, "calib", timestamp, f"{cid}.yaml")
            calib = yaml.safe_load(open(calib_file))["transform_w2c"]
            rot, trans = calib["rotation"], calib["translation"]
            rot_vec = Rotation.from_quat([rot["x"], rot["y"], rot["z"], rot["w"]]).as_rotvec()
            w2c_R = Rotation.from_rotvec(rot_vec.reshape([-1])).as_matrix().reshape([3, 3])
            w2c_t = np.float64([trans["x"], trans["y"], trans["z"]]).reshape([3, 1])

            if cid in cid_to_map_points:
                map_points = cid_to_map_points[cid]
            else:
                map_points = np.loadtxt(
                    os.path.join(args.dataset_dir, f"offs_tp/{cid}.off"),
                    dtype=float,
                    delimiter=" ",
                    skiprows=2,
                    usecols=[0, 1, 2, 3, 4, 5],
                )
                cid_to_map_points[cid] = map_points

            img = cv2.imread(img_path)
            img_depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            cam_points = w2c_R @ map_points[:, :3].T + w2c_t

            # At least min_z meters in front of the camera
            min_z = 3
            idxs = cam_points[2, :] > min_z
            map_points = map_points[idxs, :]
            cam_points = cam_points[:, idxs]

            intrinsic = np.float32(camera_info["intrinsic"])
            proj_points = intrinsic @ cam_points
            proj_points = (proj_points[:2] / proj_points[2:]).T
            proj_points = proj_points.reshape(-1, 2)

            cnt = 0
            h, w = img.shape[:2]
            min_campt_z = 1e9
            max_campt_z = -1e9
            for k in range(proj_points.shape[0]):
                x = int(proj_points[k, 0])
                y = int(proj_points[k, 1])
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                cnt += 1
                r, g, b = map_points[k, 4:] if map_points.shape[1] == 7 else map_points[k, 3:]
                cv2.circle(img, (x, y), 1, (b, g, r * 2), -1)

                u = min(int(y + 0.5), img.shape[0])
                v = min(int(x + 0.5), img.shape[1])
                img_depth[u][v] = cam_points[2, k]
                min_campt_z = min(min_campt_z, cam_points[2, k])
                max_campt_z = max(max_campt_z, cam_points[2, k])

            ts_out_dir = f"{out_dir}/{timestamp}/{nid}"
            os.makedirs(ts_out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(ts_out_dir, f"{cid}_map_proj.png"), img)
            plt.imsave(os.path.join(ts_out_dir, f"{cid}_depth.png"), img_depth)
            cam_depth_range = [20, 90, 1]
            _, min_depth, _ = generate_guassian_depth_target(
                torch.from_numpy(img_depth).unsqueeze(0),
                stride=8,
                cam_depth_range=cam_depth_range,
                constant_std=0.5,
            )
            plt.imsave(os.path.join(ts_out_dir, f"{cid}_min_depth.png"), min_depth.squeeze())
            np.save(os.path.join(ts_out_dir, f"{cid}.npy"), min_depth)
            print(f"{timestamp}_{cid}: {cnt}")
            print(f"{camera_info['cid']}: [{min_campt_z:.2f}, {max_campt_z:.2f}]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-files", nargs="+", type=str)
    parser.add_argument("--dataset-dir", default="data/xdq")
    parser.add_argument("--lidar", action="store_true")
    parser.add_argument("--map", action="store_true")
    args = parser.parse_args()
    anno_dir = os.path.join(args.dataset_dir, "annotation")
    data_dir = os.path.join(args.dataset_dir, "data")

    print(f"{len(args.anno_files)} files")

    if args.lidar:
        project_lidar(args.anno_files)

    if args.map:
        project_map(args.anno_files)
