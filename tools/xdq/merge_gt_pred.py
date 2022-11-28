import argparse
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import copyfile


def main(args):
    gt_dir, pred_dir = args.gt_dir, args.pred_dir
    os.makedirs(args.out_dir, exist_ok=True)
    sample_idxs = os.listdir(os.path.join(gt_dir, "lidar"))
    for sample_idx in tqdm(sample_idxs):
        cnt = 0
        for kid in range(0, 4):
            if os.path.exists(os.path.join(gt_dir, f"camera-{kid}/{sample_idx}")):
                cnt += 1

        fig_cam, axs_cam = plt.subplots(cnt, 2, figsize=(40, 40))
        if cnt == 1:
            axs_cam = axs_cam[None, :]
        [ax.set_axis_off() for ax in axs_cam.ravel()]
        idx = -1
        for kid in range(0, 4):
            if not os.path.exists(os.path.join(gt_dir, f"camera-{kid}/{sample_idx}")):
                continue
            gt_img = os.path.join(gt_dir, f"camera-{kid}/{sample_idx}")
            gt_img = cv2.cvtColor(cv2.imread(gt_img), cv2.COLOR_BGR2RGB)
            pred_img = os.path.join(pred_dir, f"camera-{kid}/{sample_idx}")
            pred_img = cv2.cvtColor(cv2.imread(pred_img), cv2.COLOR_BGR2RGB)
            idx += 1
            axs_cam[idx, 0].imshow(gt_img)
            axs_cam[idx, 0].set_title(f"cam-{kid} gt", fontsize=25)
            axs_cam[idx, 1].imshow(pred_img)
            axs_cam[idx, 1].set_title("pred", fontsize=25)
        # TODO: After using imshow(), although set hspace=0, the vertical margin still exists.
        fig_cam.subplots_adjust(wspace=0.01, hspace=0)
        fig_cam.savefig(
            os.path.join(args.out_dir, sample_idx.replace(".png", "_cam.png")),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(fig_cam)

        copyfile(
            os.path.join(gt_dir, f"lidar/{sample_idx}"),
            os.path.join(args.out_dir, sample_idx.replace(".png", "_gt_lidar.png")),
        )
        copyfile(
            os.path.join(pred_dir, f"lidar/{sample_idx}"),
            os.path.join(args.out_dir, sample_idx.replace(".png", "_pred_lidar.png")),
        )

        # Merging LiDAR images is time-consuming, so we just copy it
        # fig_lidar, axs_lidar = plt.subplots(1, 2, figsize=(2, 1))
        # [ax.set_axis_off() for ax in axs_lidar.ravel()]
        # gt_img = os.path.join(gt_dir, f"lidar/{sample_idx}")
        # gt_img = cv2.cvtColor(cv2.imread(gt_img), cv2.COLOR_BGR2RGB)
        # pred_img = os.path.join(pred_dir, f"lidar/{sample_idx}")
        # pred_img = cv2.cvtColor(cv2.imread(pred_img), cv2.COLOR_BGR2RGB)
        # axs_lidar[0].imshow(gt_img)
        # # TODO: Remove the margin between title and figure.
        # # axs_lidar[0].set_title("gt", fontsize=1)
        # axs_lidar[1].imshow(pred_img)
        # fig_lidar.subplots_adjust(wspace=0.01, hspace=0)
        # # Refer to: https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots
        # fig_lidar.savefig(
        #     os.path.join(args.out_dir, sample_idx.replace(".png", "_lidar.png")),
        #     dpi=3000,
        #     bbox_inches="tight",
        #     pad_inches=0.0,
        # )
        # plt.close(fig_lidar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-dir")
    parser.add_argument("--out-dir", default="merge_vis")
    args = parser.parse_args()
    main(args)
