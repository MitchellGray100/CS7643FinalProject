# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

from PointNet import PointNet
import torch.nn as nn

class PointNetClassification(nn.Module):
    def __init__(self, num_classes=10, dropout_probability=0.3):
        nn.Module.__init__(self)

        self.feature_extraction = PointNet()

        self.fully_connected_layer_1 = nn.Linear(1024, 512)
        self.fully_connected_layer_2 = nn.Linear(512, 256)
        self.fully_connected_layer_3 = nn.Linear(256, num_classes)

        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.batch_norm_2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout_probability)

    def forward(self, x):
        # (batch_size, 3, num_points)
        x, transform_matrix_feature = self.feature_extraction(x)   # x is (batch_size, 1024)

        # fc 1024 to 512
        x = self.fully_connected_layer_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        # fc 512 to 256
        x = self.fully_connected_layer_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        # fc 256 to num_classes logits
        x = self.fully_connected_layer_3(x)

        return x, transform_matrix_feature