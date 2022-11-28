import argparse
import numpy as np
from mmdet3d.core import visualize_lidar


def main(args):
    points = np.load(args.lidar_path)
    visualize_lidar("vis_lidar.png", points, xlim=(-160, 160), ylim=(-160, 160))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar-path")
    args = parser.parse_args()
    main(args)