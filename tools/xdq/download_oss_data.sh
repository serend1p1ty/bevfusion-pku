set -ex
oss_anno_prefix="oss://city-brain-vendor/yachuang.fyc/dataset/3D/longmao/results/cross"
oss_data_prefix="oss://city-brain-vendor/yachuang.fyc/dataset/3D/longmao"
for timestamp in "$@"
do
    ossutil cp $oss_anno_prefix/$timestamp data/xdq/annotation/ -rf
    ossutil cp $oss_data_prefix/$timestamp data/xdq/data/ -rf
done