# If point cloud range is changed, the models should also change their point
# cloud range accordingly
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
        timestamps=["train"],
        data_root=data_root,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
    ),
    val=dict(
        type=dataset_type,
        timestamps=["20221024", "20221026"],
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
    ),
    test=dict(
        type=dataset_type,
        timestamps=["20221024", "20221026"],
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
        with_unknown_boxes=False,
        with_hard_boxes=True,
    ),
)
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=24)
