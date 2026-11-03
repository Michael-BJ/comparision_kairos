"""Agile Autonomy's PlaNet (planner_learning/src/PlannerLearning/models/nets.py) in PyTorch.

Layer for layer the TensorFlow original with use_rgb false, use_depth true, seq_len 1, modes 3, state_dim 3,
out_seq_len 10:
  image   MobileNet v1 (alpha 1.0, ImageNet weights of keras.applications, include_top false, trainable) on the
          depth repeated to 3 channels, preprocess x / 127.5 - 1 -> (H/32 x W/32, 1024) -> Conv1D 128 k1 ->
          flatten -> Conv1D 128, 64, 64, 32 (k2, 'same', LeakyReLU 0.01) over the frame sequence ->
          Conv1D modes k3 'valid' over the 32 features
  state   Conv1D 64, 32, 32, 32 (k2, 'same', LeakyReLU 0.5) -> Conv1D modes k3 'valid'
  plan    concat (modes, 30 + 30) -> Conv1D 64, 128, 128 (k1, LeakyReLU 0.5) -> Conv1D 31 (k1)
  output  (B, modes, 1 + 30): alpha = predicted trajectory cost, then [x_1..x_10, y_1..y_10, z_1..z_10]
Keras Conv1D works on (B, length, channels); here the tensors stay in that layout and are permuted around
torch's Conv1d. The image is the 128 x 400 rig panorama, so MobileNet gives 4 x 12 cells (7 x 7 at 224 x 224).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

KERAS_MOBILENET = "~/saved_net/aa_kairos/mobilenet_1_0_224_tf_no_top.h5"
# keras.applications MobileNet v1: (pointwise filters, stride) of the 13 depthwise separable blocks
MOBILENET_BLOCKS = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1), (512, 1), (512, 1),
                    (512, 1), (512, 1), (1024, 2), (1024, 1)]


class MobileNetV1(nn.Module):
    """keras.applications.mobilenet.MobileNet(include_top=False, pooling=None): Conv-BN-ReLU6 blocks, BN eps 1e-3,
    stride-2 layers padded by one row / column at the bottom / right (Keras 'same' for even sizes and
    ZeroPadding2D(((0, 1), (0, 1))) + 'valid')."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
        self.dw, self.dw_bn, self.pw, self.pw_bn, self.strides = (nn.ModuleList(), nn.ModuleList(), nn.ModuleList(),
                                                                  nn.ModuleList(), [])
        c = 32
        for f, s in MOBILENET_BLOCKS:
            self.dw.append(nn.Conv2d(c, c, 3, s, padding=1 if s == 1 else 0, groups=c, bias=False))
            self.dw_bn.append(nn.BatchNorm2d(c, eps=1e-3, momentum=0.01))
            self.pw.append(nn.Conv2d(c, f, 1, bias=False))
            self.pw_bn.append(nn.BatchNorm2d(f, eps=1e-3, momentum=0.01))
            self.strides.append(s)
            c = f

    def forward(self, x):
        x = F.relu6(self.conv1_bn(self.conv1(F.pad(x, (0, 1, 0, 1)))))
        for dw, dbn, pw, pbn, s in zip(self.dw, self.dw_bn, self.pw, self.pw_bn, self.strides):
            if s == 2:
                x = F.pad(x, (0, 1, 0, 1))
            x = F.relu6(dbn(dw(x)))
            x = F.relu6(pbn(pw(x)))
        return x

    def load_keras(self, path=KERAS_MOBILENET):
        """ImageNet weights of keras.applications (mobilenet_1_0_224_tf_no_top.h5), the ones Agile Autonomy loads."""
        import h5py
        with h5py.File(os.path.expanduser(path), "r") as f:
            def w(layer, name):
                return torch.as_tensor(f[layer][layer][name + ":0"][()])

            def bn(mod, layer):
                mod.weight.data.copy_(w(layer, "gamma"))
                mod.bias.data.copy_(w(layer, "beta"))
                mod.running_mean.copy_(w(layer, "moving_mean"))
                mod.running_var.copy_(w(layer, "moving_variance"))

            self.conv1.weight.data.copy_(w("conv1", "kernel").permute(3, 2, 0, 1))
            bn(self.conv1_bn, "conv1_bn")
            for k in range(len(MOBILENET_BLOCKS)):
                b = k + 1
                self.dw[k].weight.data.copy_(w(f"conv_dw_{b}", "depthwise_kernel").permute(2, 3, 0, 1))
                bn(self.dw_bn[k], f"conv_dw_{b}_bn")
                self.pw[k].weight.data.copy_(w(f"conv_pw_{b}", "kernel").permute(3, 2, 0, 1))
                bn(self.pw_bn[k], f"conv_pw_{b}_bn")


def mobilenet_cells(h, w):
    """Number of MobileNetV1 output cells for an h x w image: 5 stride-2 layers, each padded by one and 'valid'."""
    for _ in range(5):
        h, w = (h - 2) // 2 + 1, (w - 2) // 2 + 1
    return h * w


class KConv1D(nn.Module):
    """keras Conv1D on (B, length, channels); 'same' pads as TensorFlow (the extra column on the right)."""

    def __init__(self, cin, cout, k, padding="valid", bias=True):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, bias=bias)
        self.pad = (((k - 1) // 2, k - 1 - (k - 1) // 2) if padding == "same" else (0, 0))

    def forward(self, x):
        return self.conv(F.pad(x.transpose(1, 2), self.pad)).transpose(1, 2)


def _stack(cin, widths, k, slope):
    layers = []
    for i, c in enumerate(widths):
        layers.append(KConv1D(cin, c, k, "same"))
        if slope is not None and (i < len(widths) - 1 or slope[1]):
            layers.append(nn.LeakyReLU(slope[0]))
        cin = c
    return nn.Sequential(*layers)


class PlaNet(nn.Module):
    def __init__(self, state_size=15, modes=3, out_seq_len=10, state_dim=3, img_hw=(128, 400), pretrained=True):
        super().__init__()
        self.modes = modes
        self.backbone = MobileNetV1()
        if pretrained:
            self.backbone.load_keras()
        cells = mobilenet_cells(*img_hw)
        self.resize_op = KConv1D(1024, 128, 1)
        self.img_mergenet = _stack(cells * 128, [128, 64, 64, 32], 2, (1e-2, True))
        self.resize_op_2 = KConv1D(1, modes, 3)
        self.states_conv = _stack(state_size, [64, 32, 32, 32], 2, (0.5, False))
        self.resize_op_3 = KConv1D(1, modes, 3)
        out = state_dim * out_seq_len + 1
        self.plan_module = nn.Sequential(KConv1D(60, 64, 1), nn.LeakyReLU(0.5), KConv1D(64, 128, 1),
                                         nn.LeakyReLU(0.5), KConv1D(128, 128, 1), nn.LeakyReLU(0.5),
                                         KConv1D(128, out, 1))

    def forward(self, depth3, state):
        """depth3 (B, 3, H, W) on Agile Autonomy's 0..255 scale, state (B, state_size) -> (B, modes, 31)."""
        x = self.backbone(depth3 / 127.5 - 1.0)                       # mobilenet.preprocess_input
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])  # (B, MxM, 1024), row-major as tf.reshape
        x = self.resize_op(x).reshape(x.shape[0], 1, -1)               # (B, seq_len 1, MxM x 128)
        x = self.img_mergenet(x)                                       # (B, 1, 32)
        img = self.resize_op_2(x.transpose(1, 2)).transpose(1, 2)      # (B, modes, 30)
        s = self.states_conv(state[:, None, :])                        # (B, 1, 32)
        st = self.resize_op_3(s.transpose(1, 2)).transpose(1, 2)       # (B, modes, 30)
        return self.plan_module(torch.cat([img, st], -1))              # (B, modes, 31)


def depth_input(depth_m):
    """Agile Autonomy's depth preprocessing (decode_depth_cv2) on a (B, H, W) batch in meters: depth in mm capped
    at 20000, / 80 (0..250), repeated to 3 channels. No resize: the panorama is used at its own size."""
    d = torch.clamp(depth_m * 1000.0, max=20000.0) / 80.0
    return d[:, None].expand(-1, 3, -1, -1)
