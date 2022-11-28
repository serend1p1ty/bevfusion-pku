import argparse
import json
import re
import os
import cv2
import numpy as np


def main(args):
    anno = json.load(open(args.anno_file))

    pcd_relative_path = re.search(
        "[^/?]+/[^/?]+/[^/?]+npy", anno["pcd_path"].replace("pcd", "npy")
    ).group()
    pcd_path = os.path.join(data_dir, pcd_relative_path)
    points = np.load(pcd_path)

    for camera_info in anno["cams"]:
        img_relative_path = re.search("[^/?]+/[^/?]+/[^/?]+jpg", camera_info["img_path"]).group()
        img_path = os.path.join(data_dir, img_relative_path)

        intrinsic = np.float32(camera_info["intrinsic"])
        extrinsic = np.float32(camera_info["extrinsic"])
        l2c_R, l2c_t = extrinsic[:3, :3], extrinsic[:3, 3:]

        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        proj_points = l2c_R @ points[:, :3].T + l2c_t
        if proj_points[2:].max() < 0:
            continue
        # tp bug
        proj_points = proj_points[:, proj_points[2, :] > 3]
        proj_points = intrinsic @ proj_points
        proj_points = (proj_points[:2] / proj_points[2:]).T
        proj_points = proj_points.reshape(-1, 2)

        cnt = 0
        h, w = img.shape[:2]
        for k in range(proj_points.shape[0]):
            x = int(proj_points[k, 0])
            y = int(proj_points[k, 1])
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            cnt += 1
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
        if not os.path.isdir("vis_output"):
            os.makedirs("vis_output")
        print(f"{img_name}: {cnt}")
        cv2.imwrite(f"vis_output/{img_name}", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file")
    parser.add_argument("--dataset-dir", default="data/xdq")
    args = parser.parse_args()
    anno_dir = os.path.join(args.dataset_dir, "annotation")
    data_dir = os.path.join(args.dataset_dir, "data")
    main(args)
