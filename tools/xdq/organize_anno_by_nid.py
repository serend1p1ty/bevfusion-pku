import argparse
import os
import json
from glob import glob
from shutil import move


def main(args):
    anno_dir = os.path.join(args.data_dir, "annotation")
    timestamps = os.listdir(anno_dir)
    for ts in timestamps:
        ts_dir = os.path.join(anno_dir, ts)
        for anno_file in glob(os.path.join(ts_dir, "*.json")):
            anno = json.load(open(anno_file))
            nid = anno["nid"]
            nid_dir = os.path.join(ts_dir, nid)
            os.makedirs(nid_dir, exist_ok=True)
            move(anno_file, os.path.join(nid_dir, os.path.basename(anno_file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/xdq")
    args = parser.parse_args()
    main(args)
