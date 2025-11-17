from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import SamplePoints

file_path = '../../data/'
number_of_points = 1024

# ModelNet10
train_10 = ModelNet(root=file_path+"ModelNet10/", name='10', train=True, pre_transform=SamplePoints(number_of_points), force_reload=True)
test_10 = ModelNet(root=file_path+"ModelNet10/", name='10', train=False, pre_transform=SamplePoints(number_of_points), force_reload=True)

print(f"ModelNet10: {train_10.num_classes}")
print(f"train n samples: {len(train_10)}")
print(f"test n samples: {len(test_10)}")

loader = DataLoader(train_10, batch_size=5, shuffle=False)
batch_iterator = iter(loader)
batch = next(batch_iterator)
print(batch.pos.shape)
# class
print(batch.y)
# sample assignment
print(batch.batch)

# ModelNet40
train_40 = ModelNet(root=file_path+"ModelNet40/", name='40', train=True, pre_transform=SamplePoints(number_of_points), force_reload=True)
test_40 = ModelNet(root=file_path+"ModelNet40/", name='40', train=False, pre_transform=SamplePoints(number_of_points), force_reload=True)

print(f"ModelNet40: {train_40.num_classes}")
print(f"train n samples: {len(train_40)}")
print(f"test n samples: {len(test_40)}")

loader = DataLoader(train_40, batch_size=5, shuffle=False)
batch_iterator = iter(loader)
batch = next(batch_iterator)
print(batch.pos.shape)
# class
print(batch.y)
# sample assignment
print(batch.batch)