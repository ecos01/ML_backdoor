import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    

    def __init__(self, num_classes: int):
        super().__init__()
        # Three conv blocks: channels double at each stage (32 -> 64 -> 128)
        # padding=1 keeps spatial size intact before each MaxPool halves it
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 64x64 -> 64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 16x16 -> 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 16x16 -> 8x8
        )
        # AdaptiveAvgPool forces output to 4x4 regardless of input resolution
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        # fc1 is the embedding layer: 128*4*4=2048 -> 256 dimensions
        # This is the layer analyzed by the fine-pruning defense
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        # fc2 maps the 256-dim embedding to class scores
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)          # flatten to (batch, 2048)
        emb = self.relu_fc(self.fc1(x))     # 256-dim face embedding
        logits = self.fc2(emb)
        # return_embedding=True is used by analysis_defense.py
        if return_embedding:
            return logits, emb
        return logits
