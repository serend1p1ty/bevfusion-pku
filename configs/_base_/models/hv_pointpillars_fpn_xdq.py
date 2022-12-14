# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
#### modified ####
voxel_size = [0.4, 0.4, 10]
model = dict(
    type="MVXFasterRCNN",
    # Used to calculate min_camz
    norm_offsets={
        "1": [22, -70, 45.75],
        "2": [-23.32, 35.89, 45.75],
        "3": [-25, 98, 45.34],
        "5": [30, -110, 45.56],
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
    pts_voxel_layer=dict(
        max_num_points=64,
        #### modified ####
        point_cloud_range=[-160.0, -160.0, -6.0, 160.0, 160.0, 4.0],
        voxel_size=voxel_size,
        max_voxels=(30000, 40000),
    ),
    pts_voxel_encoder=dict(
        type="HardVFE",
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        #### modified ####
        point_cloud_range=[-160.0, -160.0, -6.0, 160.0, 160.0, 4.0],
        norm_cfg=dict(type="naiveSyncBN1d", eps=1e-3, momentum=0.01),
    ),
    #### modified ####
    pts_middle_encoder=dict(type="PointPillarsScatter", in_channels=64, output_shape=[800, 800]),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    pts_neck=dict(
        type="FPN",
        norm_cfg=dict(type="naiveSyncBN2d", eps=1e-3, momentum=0.01),
        act_cfg=dict(type="ReLU"),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3,
    ),
    pts_bbox_head=dict(
        type="Anchor3DHead",
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1, 2, 4],
            sizes=[
                [0.8660, 2.5981, 1.0],  # 1.5/sqrt(3)
                [0.5774, 1.7321, 1.0],  # 1/sqrt(3)
                [1.0, 1.0, 1.0],
                [0.4, 0.4, 1],
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
