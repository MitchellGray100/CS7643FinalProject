# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

from TNet import TNet
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.input_TNet = TNet(transform_dimension=3)
        self.feature_TNet = TNet(transform_dimension=64)

        self.convolution_layer_1 = nn.Conv1d(3, 64, 1)
        self.convolution_layer_2 = nn.Conv1d(64, 64, 1)
        self.convolution_layer_3 = nn.Conv1d(64, 64, 1)
        self.convolution_layer_4 = nn.Conv1d(64, 128, 1)
        self.convolution_layer_5 = nn.Conv1d(128, 1024, 1)

        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.batch_norm_2 = nn.BatchNorm1d(64)
        self.batch_norm_3 = nn.BatchNorm1d(64)
        self.batch_norm_4 = nn.BatchNorm1d(128)
        self.batch_norm_5 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        batch_size = x.size(0)

        transform_matrix_input = self.input_TNet(x)
        x = torch.bmm(transform_matrix_input, x).contiguous()

        # mlp
        # 3 to 64
        x = self.convolution_layer_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        # 64 to 64
        x = self.convolution_layer_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        
        transform_matrix_feature = self.feature_TNet(x)
        x = torch.bmm(transform_matrix_feature, x).contiguous()

        # 64 to 64
        x = self.convolution_layer_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)

        # 64 to 128
        x = self.convolution_layer_4(x)
        x = self.batch_norm_4(x)
        x = self.relu(x)

        # 128 to 1024
        x = self.convolution_layer_5(x)
        x = self.batch_norm_5(x)
        x = self.relu(x)

        # global
        x = self.maxpool(x)
        x = x.reshape(batch_size, 1024)

        return x, transform_matrix_feature
