import os.path as osp

import mmcv
import pytest
import torch

from mmdet import digit_version
from mmdet.models.necks import FPN, YOLOV3Neck
from .utils import ort_validate

if digit_version(torch.__version__) <= digit_version("1.5.0"):
    pytest.skip("ort backend does not support version below 1.5.0", allow_module_level=True)

# Control the returned model of neck_config()
test_step_names = {
    "normal": 0,
    "wo_extra_convs": 1,
    "lateral_bns": 2,
    "bilinear_upsample": 3,
    "scale_factor": 4,
    "extra_convs_inputs": 5,
    "extra_convs_laterals": 6,
    "extra_convs_outputs": 7,
}

data_path = osp.join(osp.dirname(__file__), "data")


def fpn_config(test_step_name):
    """Return the class containing the corresponding attributes according to
    the test_step_names."""

    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8

    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i]) for i in range(len(in_channels))
    ]

    if test_step_names[test_step_name] == 0:
        fpn_model = FPN(
            in_channels=in_channels, out_channels=out_channels, add_extra_convs=True, num_outs=5
        )
        return fpn_model, feats
    elif test_step_names[test_step_name] == 1:
        # Tests for fpn with no extra convs (pooling is used instead)
        fpn_model = FPN(
            in_channels=in_channels, out_channels=out_channels, add_extra_convs=False, num_outs=5
        )
        return fpn_model, feats
    elif test_step_names[test_step_name] == 2:
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=True,
            no_norm_on_lateral=False,
            norm_cfg=dict(type="BN", requires_grad=True),
            num_outs=5,
        )
        return fpn_model, feats
    elif test_step_names[test_step_name] == 3:
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=True,
            upsample_cfg=dict(mode="bilinear", align_corners=True),
            num_outs=5,
        )
        return fpn_model, feats
    elif test_step_names[test_step_name] == 4:
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs=True,
            upsample_cfg=dict(scale_factor=2),
            num_outs=5,
        )
        return fpn_model, feats
    elif test_step_names[test_step_name] == 5:
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs="on_input",
            num_outs=5,
        )
        return fpn_model, feats
    elif test_step_names[test_step_name] == 6:
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs="on_lateral",
            num_outs=5,
        )
        return fpn_model, feats
    elif test_step_names[test_step_name] == 7:
        fpn_model = FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            add_extra_convs="on_output",
            num_outs=5,
        )
        return fpn_model, feats


def yolo_config(test_step_name):
    """Config yolov3 Neck."""

    in_channels = [16, 8, 4]
    out_channels = [8, 4, 2]

    # The data of yolov3_neck.pkl contains a list of
    # torch.Tensor, where each torch.Tensor is generated by
    # torch.rand and each tensor size is:
    # (1, 4, 64, 64), (1, 8, 32, 32), (1, 16, 16, 16).
    yolov3_neck_data = "yolov3_neck.pkl"
    feats = mmcv.load(osp.join(data_path, yolov3_neck_data))

    if test_step_names[test_step_name] == 0:
        yolo_model = YOLOV3Neck(in_channels=in_channels, out_channels=out_channels, num_scales=3)
        return yolo_model, feats


def test_fpn_normal():
    outs = fpn_config("normal")
    ort_validate(*outs)


def test_fpn_wo_extra_convs():
    outs = fpn_config("wo_extra_convs")
    ort_validate(*outs)


def test_fpn_lateral_bns():
    outs = fpn_config("lateral_bns")
    ort_validate(*outs)


def test_fpn_bilinear_upsample():
    outs = fpn_config("bilinear_upsample")
    ort_validate(*outs)


def test_fpn_scale_factor():
    outs = fpn_config("scale_factor")
    ort_validate(*outs)


def test_fpn_extra_convs_inputs():
    outs = fpn_config("extra_convs_inputs")
    ort_validate(*outs)


def test_fpn_extra_convs_laterals():
    outs = fpn_config("extra_convs_laterals")
    ort_validate(*outs)


def test_fpn_extra_convs_outputs():
    outs = fpn_config("extra_convs_outputs")
    ort_validate(*outs)


def test_yolo_normal():
    outs = yolo_config("normal")
    ort_validate(*outs)
