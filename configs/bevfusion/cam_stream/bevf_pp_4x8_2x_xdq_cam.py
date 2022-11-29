_base_ = [
    "../../_base_/datasets/xdq_cam_pp.py",
    "../../_base_/schedules/schedule_2x.py",
    "../../_base_/default_runtime.py",
]
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

#### modified ####
final_dim = (1080, 1920)  # HxW
downsample = 8
#### modified ####
voxel_size = [0.4, 0.4, 10]
model = dict(
    type="BEVF_FasterRCNN",
    camera_stream=True,
    lss=False,
    #### modified ####
    norm_offsets={
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
    },
    camera_depth_range=[20.0, 90.0, 1.0],
    pc_range=[-160, -160, -6, 160, 160, 4],
    grid=0.8,
    ##################
    num_views=6,
    final_dim=final_dim,
    downsample=downsample,
    img_backbone=dict(
        type="CBSwinTransformer",
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
    ),
    img_neck=dict(
        type="FPNC",
        final_dim=final_dim,
        downsample=downsample,
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        use_adp=True,
        num_outs=5,
    ),
    pts_bbox_head=dict(
        type="Anchor3DHead",
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            #### modified ####
            ranges=[
                [-160, -160, -1.80032795, 160, 160, -1.80032795],
                [-160, -160, -1.74440365, 160, 160, -1.74440365],
                [-160, -160, -1.68526504, 160, 160, -1.68526504],
                [-160, -160, -1.67339111, 160, 160, -1.67339111],
                [-160, -160, -1.61785072, 160, 160, -1.61785072],
                [-160, -160, -1.80984986, 160, 160, -1.80984986],
                [-160, -160, -1.763965, 160, 160, -1.763965],
            ],
            sizes=[
                [1.95017717, 4.60718145, 1.72270761],  # car
                [2.4560939, 6.73778078, 2.73004906],  # truck
                [2.87427237, 12.01320693, 3.81509561],  # trailer
                [0.60058911, 1.68452161, 1.27192197],  # bicycle
                [0.66344886, 0.7256437, 1.75748069],  # pedestrian
                [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
                [2.49008838, 0.48578221, 0.98297065],  # barrier
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True,
        ),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder", code_size=9),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False,
        )
    ),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500,
        )
    ),
)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)

load_img_from = "work_dirs/pre_mask_rcnn_dbswin-t_fpn_3x_nuim_cocopre/epoch_36.pth"
#### modified ####
fp16 = dict(loss_scale=32.0)
load_from = "work_dirs/pre_bevf_pp_4x8_2x_nusc_cam/new_epoch_24.pth"
##################