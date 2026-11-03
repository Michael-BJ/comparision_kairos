"""PlannerNet of NEO-Planner (identical to ~/Training_model_FM_planner/nn_trainer.py and neo_planner_node.py),
with the image size of the KAIROS rig panorama (neo_data.PANO_H x PANO_W) instead of 640 x 480."""
import torch
import torch.nn as nn
import torchvision.models as models

from neo_data import MOTION_INPUT_SIZE, OUTPUT_SIZE, PANO_H, PANO_W

IMG_HEIGHT, IMG_WIDTH = PANO_H, PANO_W
IMG_FEATURE_SIZE = 24
MOTION_FEATURE_SIZE = 24


class PlannerNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        try:
            self.img_backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
        except Exception as e:  # offline fallback, as nn_trainer.py
            print(f"[WARN] pretrained ResNet18 unavailable ({e}) - random init")
            self.img_backbone = models.resnet18(weights=None)
        for param in self.img_backbone.parameters():
            param.requires_grad = False  # freeze backbone
        # replaced layers are freshly created -> trainable (as in the original)
        self.img_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.img_backbone.fc = nn.Linear(self.img_backbone.fc.in_features, IMG_FEATURE_SIZE)
        self.motion_backbone = nn.Sequential(
            nn.Linear(MOTION_INPUT_SIZE, 48), nn.LeakyReLU(),
            nn.Linear(48, 24), nn.LeakyReLU(),
            nn.Linear(24, 24), nn.LeakyReLU(),
            nn.Linear(24, MOTION_FEATURE_SIZE),
        )
        self.mlp = nn.Sequential(
            nn.Linear(IMG_FEATURE_SIZE + MOTION_FEATURE_SIZE, 48), nn.LeakyReLU(),
            nn.Linear(48, 96), nn.LeakyReLU(),
            nn.Linear(96, 96), nn.LeakyReLU(),
            nn.Linear(96, OUTPUT_SIZE),
        )

    def forward(self, x):
        img = x[:, :IMG_WIDTH * IMG_HEIGHT].reshape(-1, 1, IMG_HEIGHT, IMG_WIDTH)
        vector = x[:, IMG_WIDTH * IMG_HEIGHT:]
        fused = torch.cat([self.img_backbone(img), self.motion_backbone(vector)], dim=1)
        return self.mlp(fused)
