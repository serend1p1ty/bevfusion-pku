import argparse
import os
import re
import json
import yaml
import mmcv
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import init_dist, load_checkpoint

from mmdet3d.core import LiDARInstance3DBoxes, Box3DMode
from mmdet3d.datasets import Custom3DDataset
from mmdet3d.core.utils import visualize_camera, visualize_lidar
from mmdet3d.datasets import build_dataloader, XdqDataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_detector

cls_mapper = {
    "pedestrian": "行人",
    "truck": "卡车货车",
    "car": "汽车",
    "bicycle": "自行车",
    "traffic_cone": "锥桶",
}


class LongmaoDataset(Custom3DDataset):
    def __init__(self, data_dir, pipelines, use_camera=True, test_mode=True):
        self.data_dir = data_dir
        self.pipeline = Compose(pipelines)
        self.use_camera = use_camera
        self.box_type_3d = LiDARInstance3DBoxes
        self.box_mode_3d = Box3DMode.LIDAR
        self.test_mode = test_mode

        # load data from .csv file
        csv_file = glob(os.path.join(data_dir, "lists*.csv"))
        assert len(csv_file) == 1
        csv_file = csv_file[0]
        # remove the first title line
        self.data_infos = open(csv_file).readlines()[1:]

    def get_data_info(self, index):
        info = self.data_infos[index].split(",")
        # url img1 img2 img3 img4
        assert len(info) == 5

        # lidar points have been normalized
        pcd_url = info[0]
        res = re.search("([^/?]+)/[^/?]+pcd", pcd_url)
        relative_path, nid = res.group().replace(".pcd", "_norm.npy").split("/"), res.group(1)
        relative_path.insert(-1, "longmao")
        relative_path = "/".join(relative_path)
        pcd_path = os.path.join(self.data_dir, relative_path)
        input_dict = dict(
            sample_idx=pcd_path.replace("/", "_"),
            pts_filename=pcd_path,
            nid=nid,
            sweeps=[],
            timestamp=0,
        )

        if self.use_camera:
            img_paths = []
            lidar2img_rts = []
            for img_url in info[1:]:
                if img_url in ["", "\n"]:
                    continue
                res = re.search("([^/?]+)/([0-9])[^/?]+jpg", img_url)
                relative_path = res.group().split("/")
                relative_path.insert(-1, "longmao")
                relative_path = "/".join(relative_path)
                img_path = os.path.join(self.data_dir, relative_path)
                nid, kid = res.group(1), res.group(2)
                img_paths.append(img_path)

                # lidar to camera
                calib_file = os.path.join(self.data_dir, f"calib/{nid}-{kid}.yaml")
                calib = yaml.safe_load(open(calib_file))
                rotation = []
                for k in ["x", "y", "z", "w"]:
                    rotation.append(calib["transform_w2c"]["rotation"][k])
                translation = []
                for k in ["x", "y", "z"]:
                    translation.append(calib["transform_w2c"]["translation"][k])
                l2c_R = Rotation.from_quat(rotation).as_matrix()
                l2c_t = np.array(translation).reshape(3, 1)
                lidar2camera = np.eye(4).astype(np.float32)
                lidar2camera[:3, :3] = l2c_R
                lidar2camera[:3, 3:] = l2c_t

                # lidar to image
                K = np.array(calib["K"]).reshape(3, 3)
                viewpad = np.eye(4).astype(np.float32)
                viewpad[:3, :3] = K
                lidar2image = viewpad @ lidar2camera
                lidar2img_rts.append(lidar2image)

            input_dict.update(dict(img_filename=img_paths, lidar2img=lidar2img_rts))

        return input_dict


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet visualize a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--longmao-dir", help="Inference all the pcds in longmao dir.")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="pre_anno")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # By default, use the test split pipeline.
    dataset = LongmaoDataset(args.longmao_dir, cfg.data["test"]["pipeline"])

    shuffle = False
    if shuffle:
        dataset.flag = np.zeros(len(dataset), dtype=np.uint8)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=shuffle,
    )

    # build the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
    model.eval()

    for data in tqdm(data_loader):
        metas = data["img_metas"][0].data[0][0]
        name = metas["sample_idx"]
        nid = metas["nid"]
        if nid not in cfg.norm_offsets:
            if not args.vis:
                anno_dict = {"pre_anno": {"bboxes": [], "labels": []}}
                save_path = (
                    os.path.join(args.out_dir, "/".join(metas["pts_filename"].split("/")[3:]))
                    .replace("_norm.npy", ".json")
                    .replace("longmao/", "")
                )
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                json.dump(anno_dict, open(save_path, "w"), indent=4, ensure_ascii=False)
            continue

        norm_offset = cfg.norm_offsets[nid]

        with torch.inference_mode():
            outputs = model(return_loss=False, **data)[0]["pts_bbox"]

        if not args.vis:
            if outputs["boxes_3d"].tensor.shape[0] == 0:
                anno_dict = {"pre_anno": {"bboxes": [], "labels": []}}
                save_path = (
                    os.path.join(args.out_dir, "/".join(metas["pts_filename"].split("/")[3:]))
                    .replace("_norm.npy", ".json")
                    .replace("longmao/", "")
                )
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                json.dump(anno_dict, open(save_path, "w"), indent=4, ensure_ascii=False)
                continue

            boxes = outputs["boxes_3d"]
            centers = boxes.gravity_center
            xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]
            ws = boxes.tensor[:, 3]
            ls = boxes.tensor[:, 4]
            hs = boxes.tensor[:, 5]
            yaws = boxes.tensor[:, 6]
            boxes_list = []
            for x, y, z, w, l, h, yaw in zip(xs, ys, zs, ws, ls, hs, yaws):
                boxes_list.append({"x": x, "y": y, "z": z, "w": w, "l": l, "h": h, "yaw": yaw})
            anno_dict = {
                "pre_anno": {
                    "bboxes": boxes_list,
                    "labels": [
                        cls_mapper[XdqDataset.CLASSES[label_idx]]
                        for label_idx in outputs["labels_3d"]
                    ],
                }
            }
            save_path = (
                os.path.join(args.out_dir, "/".join(metas["pts_filename"].split("/")[3:]))
                .replace("_norm.npy", ".json")
                .replace("longmao/", "")
            )
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            json.dump(anno_dict, open(save_path, "w"), indent=4, ensure_ascii=False)

        if not args.vis:
            continue

        bboxes = outputs["boxes_3d"].tensor.numpy()
        scores = outputs["scores_3d"].numpy()
        labels = outputs["labels_3d"].numpy()

        if args.bbox_score is not None:
            indices = scores >= args.bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

        # Unnormalize bboxes coordinates
        if bboxes is not None:
            bboxes.tensor[:, :3] -= torch.Tensor(norm_offset)

        if "img_filename" in metas:
            for k, image_path in enumerate(metas["img_filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2img"][k],
                    classes=cfg.class_names,
                )

        if "points" in data:
            lidar_points = data["points"][0].data[0][0].numpy()
            # Unnormalize lidar points coordinates
            lidar_points -= norm_offset + [0.0]
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar_points,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.class_names,
            )


if __name__ == "__main__":
    main()
