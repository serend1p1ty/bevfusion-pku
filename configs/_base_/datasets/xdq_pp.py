#### modified ####
point_cloud_range = [-160.0, -160.0, -6.0, 160.0, 160.0, 4.0]
#### modified ####
# XDQ point cloud has been normalized offline, and stored in disk (*_norm.npy)
# The variable is used to normalize image LSS point cloud online and
# unnormalize prediction result.
norm_offsets = {
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
}
class_names = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]
evaluation = dict(interval=2, gpu_collect=True)

dataset_type = "XdqDataset"
data_root = "data/xdq/"
input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)
#### modified ####
img_scale = (960, 540)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#### modified ####
sweeps_num = 0
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="LoadMultiViewImageFromFiles",
        project_pts_to_img_depth=True,
        cam_depth_range=[10.0, 150.0, 1.0],
        norm_offsets=norm_offsets,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="MyResize", img_scale=img_scale, keep_ratio=True),
    dict(type="MyNormalize", **img_norm_cfg),
    dict(type="MyPad", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "img", "img_depth", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
    ),
    dict(type="LoadMultiViewImageFromFiles"),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type="MyResize", img_scale=img_scale, keep_ratio=True),
            dict(type="MyNormalize", **img_norm_cfg),
            dict(type="MyPad", size_divisor=32),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points", "img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        # Map to image projection is incorrect: 20220922#35, 20221024#3, 20221026#3, 20221117-1004#3
        timestamps=[
            "20220825",
            "20220921#2",
            "20220922#3/32/33/34",
            "20220930#3/12/32",
            "20221016",
            "20221018#1/16/19/21",
            "20221024#7",
            "20221026#19",
            "20221031-0902",
            "20221031-0907",
            "20221031-0908",
            "20221031-0909",
            "20221031-0915",
            "20221031-0916",
            "20221031-0923",
            "20221031-0926",
            "20221031-0927",
            "20221031-0928",
            "20221104-0929",
            "20221107-0930",
            "20221109-1003",
            "20221117-1004#2/5/12/32/33",
        ],
        data_root=data_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
        camz_range=[10.0, 150.0],
    ),
    val=dict(
        type=dataset_type,
        timestamps=[
            "20220921#21",
            "20220930#34",
            "20221018#2/3",
            "20221024#19",
        ],
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
        camz_range=[10.0, 150.0],
    ),
    test=dict(
        type=dataset_type,
        timestamps=[
            "20220921#21",
            "20220930#34",
            "20221018#2/3",
            "20221024#19",
        ],
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
        camz_range=[10.0, 150.0],
    ),
)
