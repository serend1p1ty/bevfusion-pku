# If point cloud range is changed, the models should also change their point
# cloud range accordingly
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
# For nuScenes we usually do 10-class detection
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
dataset_type = "XdqDataset"
data_root = "data/xdq/"
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)
file_client_args = dict(backend="disk")
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))
#### modified ####
sweeps_num = 0
train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps", sweeps_num=sweeps_num, file_client_args=file_client_args
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type="GlobalRotScaleTrans",
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0],
    # ),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(type="SplitBEVLiDAR"),
    dict(
        type="LoadPointsFromMultiSweeps", sweeps_num=sweeps_num, file_client_args=file_client_args
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        timestamps=[
            "20220825",
            "20220921#2",
            "20220922",
            "20220930#3/12/32",
            "20221016",
            "20221018#1/16/19/21",
            "20221024#3/7",
            "20221026",
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
            "20221117-1004",
        ],
        data_root=data_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=False,
        # camz_range=[10, 150],
    ),
    # train=dict(
    #     type="CBGSDataset",
    #     dataset=dict(
    #         type=dataset_type,
    #         timestamps=[
    #             "20220825",
    #             "20220921",
    #             "20220922",
    #             "20220930#3/12/32",
    #             "20221016",
    #             "20221018#1/16/19",
    #             "20221024#3/7",
    #             "20221026",
    #         ],
    #         data_root=data_root,
    #         pipeline=train_pipeline,
    #         classes=class_names,
    #         modality=input_modality,
    #         test_mode=False,
    #         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    #         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    #         box_type_3d="LiDAR",
    #         with_unknown_boxes=False,
    #         with_hard_boxes=False,
    #     ),
    # ),
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
        with_hard_boxes=False,
        # camz_range=[10, 150],
    ),
    test=dict(
        type=dataset_type,
        timestamps=[
            "20220921#21",  # 755
            "20220930#34",  # 229
            "20221018#2/3",  # 388, 918
            "20221024#19",  # 167
        ],
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=False,
        # camz_range=[10, 150],
    ),
)
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=1)
