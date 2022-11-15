#### modified ####
point_cloud_range = [-160.0, -160.0, -6.0, 160.0, 160.0, 4.0]
#### modified ####
# XDQ point cloud has been normalized offline, and stored in disk (*_norm.npy)
# The variable is used to normalize image LSS point cloud online and
# unnormalize prediction result.
norm_offsets = {
    "1": [-4.46, -34.1, 45.75],
    "2": [-23.32, 35.89, 45.75],
    "3": [-4.64, 71.39, 45.34],
    "7": [-17.48, -19.43, 45.19],
    "12": [184.74, -78.32, 45.99],
    "16": [33.16, 83.63, 46.91],
    "17": [-18.74, -35.58, 45.41],
    "19": [-12.4, -52.47, 46.21],
    "21": [12.77, 71.73, 45.6],
    "32": [48.12, -0.02, 45.93],
    "33": [-53.45, -6.59, 45.97],
    "34": [-8.23, -31.04, 45.71],
    "35": [63.23, 49.72, 46.03],
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
evaluation = dict(interval=24)

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
        cam_depth_range=[22.0, 90.0, 1.0],
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
        timestamps=[
            "20220825",
            "20220922",
            "20220930",
        ],
        data_root=data_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        timestamps=[
            "20220921",
        ],
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        timestamps=[
            "20220921",
        ],
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)
