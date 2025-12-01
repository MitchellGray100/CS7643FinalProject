# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

from TNet import TNet
import torch
import torch.nn as nn

class PointResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointResBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # first convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # Second Convolution 
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip Connection
        self.shortcut = nn.Sequential()
        
        # If input and output channels differ, we must project x to match output
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # Save identity (possibly projected)
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # non-linearity
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Path B: Addition
        out += identity
        
        # Final Activation
        out = self.relu(out)
        return out

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.input_TNet = TNet(transform_dimension=3)
        self.feature_TNet = TNet(transform_dimension=64)

        # Initial lifting layer: 3 -> 64
        # We use a standard block here to get into feature space 
        # before starting residual connections
        self.initial_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Encoder 1: Processing the 64-dim features
        self.encoder_1 = nn.Sequential(
            PointResBlock(64, 64),
            PointResBlock(64, 64)
        )

        # Encoder 2: Deepening and expanding
        self.encoder_2 = nn.Sequential(
            PointResBlock(64, 64),  # Extra processing at 64
            PointResBlock(64, 128), # Expand
            PointResBlock(128, 1024) # Expand to Global Vector size
        )
        
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        batch_size = x.size(0)

        # 1. Input Transform
        transform_matrix_input = self.input_TNet(x)
        x = torch.bmm(transform_matrix_input, x).contiguous()

        # 2. Lift to 64-dim space
        x = self.initial_conv(x)
        
        # 3. Feature Transform (Applied early, on the "base" features)
        transform_matrix_feature = self.feature_TNet(x)
        x = torch.bmm(transform_matrix_feature, x).contiguous()
        
        # 4. Deep Residual Encoding
        x = self.encoder_1(x)
        x = self.encoder_2(x)

        # 5. Global Pool
        x = self.maxpool(x)
        x = x.reshape(batch_size, 1024)

        return x, transform_matrix_feature