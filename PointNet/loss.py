# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

import torch

def t_net_regularization_loss(transform_matrix_feature):
    # (batch_size, k, k)
    transform_dimension = transform_matrix_feature.size(1)

    # element A @ A^T
    matrix_product = torch.bmm(transform_matrix_feature, transform_matrix_feature.transpose(1, 2))

    identity_matrix = torch.eye(transform_dimension, device=transform_matrix_feature.device)

    difference = matrix_product - identity_matrix

    loss_per_sample = (difference**2).sum(dim=(1, 2))
    loss = loss_per_sample.mean()
    return loss
