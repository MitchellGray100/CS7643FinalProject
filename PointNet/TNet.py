# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

import torch
import torch.nn as nn

class TNet(nn.Module):
    def __init__(self, transform_dimension=3):
        nn.Module.__init__(self)
        self.transform_dimension = transform_dimension

        self.convolution_layer_1 = nn.Conv1d(transform_dimension, 64, 1)
        self.convolution_layer_2 = nn.Conv1d(64, 128, 1)
        self.convolution_layer_3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, transform_dimension ** 2)
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)
        
        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.batch_norm_3 = nn.BatchNorm1d(1024)
        self.batch_norm_4 = nn.BatchNorm1d(512)
        self.batch_norm_5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, learned_matrix):
        batch_size = learned_matrix.size(0)
        
        # convolution
        learned_matrix = self.convolution_layer_1(learned_matrix)
        learned_matrix = self.batch_norm_1(learned_matrix)
        learned_matrix = self.relu(learned_matrix)

        learned_matrix = self.convolution_layer_2(learned_matrix)
        learned_matrix = self.batch_norm_2(learned_matrix)
        learned_matrix = self.relu(learned_matrix)

        learned_matrix = self.convolution_layer_3(learned_matrix)
        learned_matrix = self.batch_norm_3(learned_matrix)
        learned_matrix = self.relu(learned_matrix)

        # max pool
        learned_matrix = self.maxpool(learned_matrix)
        # flatten
        learned_matrix = learned_matrix.reshape(batch_size, 1024)
        
        # fc (batch_size, 512)
        learned_matrix = self.fc1(learned_matrix)
        learned_matrix = self.batch_norm_4(learned_matrix)
        learned_matrix = self.relu(learned_matrix)

        learned_matrix = self.fc2(learned_matrix)
        learned_matrix = self.batch_norm_5(learned_matrix)
        learned_matrix = self.relu(learned_matrix)

        # final fc (batch_size, initial_dim * initial_dim)
        learned_matrix = self.fc3(learned_matrix)
        
        # identity matrix to add to learned matrix (initial_dim, initial_dim)
        identity_matrix = torch.eye(self.transform_dimension, device=learned_matrix.device)
        identity_matrix = identity_matrix.reshape(1, self.transform_dimension * self.transform_dimension)
        learned_matrix = learned_matrix + identity_matrix
        learned_matrix = learned_matrix.reshape(batch_size, self.transform_dimension, self.transform_dimension)
        
        return learned_matrix