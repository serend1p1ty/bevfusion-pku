import argparse
import mmcv
import numpy as np
import os
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import init_dist, load_checkpoint
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

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


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet visualize a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="vis")
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

    if args.mode == "pred":
        assert distributed, "The script only support distributed prediction."

    # build the dataloader
    dataset = build_dataset(cfg.data.get(args.split))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False
        )
        model.eval()

    for data in tqdm(data_loader):
        metas = (
            data["img_metas"][0].data[0][0]
            if isinstance(data["img_metas"], list)
            else data["img_metas"].data[0][0]
        )
        name = metas["sample_idx"]
        ann_info = metas["ann_info"]
        nid = metas["nid"]
        norm_offset = norm_offsets[nid]

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(return_loss=False, **data)[0]["pts_bbox"]

        if args.mode == "gt" and "gt_bboxes_3d" in ann_info:
            bboxes = ann_info["gt_bboxes_3d"].tensor.numpy()
            labels = ann_info["gt_labels_3d"]

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs:
            bboxes = outputs["boxes_3d"].tensor.numpy()
            scores = outputs["scores_3d"].numpy()
            labels = outputs["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        # Unnormalize bboxes coordinates
        if bboxes is not None:
            bboxes.tensor[:, :3] -= torch.tensor(norm_offset)

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
            lidar = (
                data["points"][0].data[0][0].numpy()
                if isinstance(data["points"], list)
                else data["points"].data[0][0].numpy()
            )
            # Unnormalize lidar points coordinates
            lidar -= norm_offset + [0.0]
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.class_names,
            )


if __name__ == "__main__":
    main()
