# REFERENCE:
# Official point net Github with TensorFlow: https://github.com/charlesq34/pointnet

import torch

def apply_random_y_rotation(data):
    points = data.pos
    random_angle = torch.rand(1)*2*torch.pi
    cos_value = torch.cos(random_angle)
    sin_value = torch.sin(random_angle)
    rotation_matrix = torch.tensor([[cos_value.item(), 0.0, sin_value.item()],[0.0, 1.0, 0.0],[-sin_value.item(), 0.0, cos_value.item()]], dtype=points.dtype, device=points.device)
    rotated_points = points @ rotation_matrix.t()
    data.pos = rotated_points
    return data

def apply_random_jitter(data, standard_deviation=0.02, clip_value=0.05):
    points = data.pos
    noise = torch.normal(mean=0.0,std=standard_deviation,size=points.shape,dtype=points.dtype,device=points.device)
    noise = noise.clamp(-clip_value, clip_value)
    noisy_points = points + noise
    data.pos = noisy_points
    return data