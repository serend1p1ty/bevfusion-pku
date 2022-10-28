_base_ = "../htc/htc_r50_fpn_20e_coco.py"
model = dict(
    pretrained="open-mmlab://msra/hrnetv2_w32",
    backbone=dict(
        _delete_=True,
        type="HRNet",
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(32, 64),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
            ),
        ),
    ),
    neck=dict(_delete_=True, type="HRFPN", in_channels=[32, 64, 128, 256], out_channels=256),
)
