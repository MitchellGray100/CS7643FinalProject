# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

import torch

def t_net_regularization_loss(transform_matrix_feature):
    # (batch_size, k, k)
    batch_size = transform_matrix_feature.size(0)
    transform_dimension = transform_matrix_feature.size(1)

    identity_matrix = torch.eye(transform_dimension, device=transform_matrix_feature.device)
    identity_matrix = identity_matrix.reshape(1, transform_dimension, transform_dimension)

    # A A^T
    matrix_product = torch.bmm(transform_matrix_feature, transform_matrix_feature.transpose(1, 2))
    difference = matrix_product - identity_matrix
    difference = difference.reshape(batch_size, transform_dimension * transform_dimension)

    loss = torch.norm(difference, p='fro', dim=1)** 2
    loss = loss.mean()
    return loss
