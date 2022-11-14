python -m torch.distributed.launch --nproc_per_node=1 tools/xdq/visualize.py \
configs/bevfusion/lidar_stream/hv_pointpillars_secfpn_sbn-all_4x8_2x_xdq-3d.py \
--launcher pytorch \
--checkpoint work_dirs/hv_pointpillars_secfpn_sbn-all_4x8_2x_xdq-3d/latest.pth \
--mode pred \
--out-dir vis_xdq_test_pred \
--bbox-score $1 ${@:2}