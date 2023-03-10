import os.path as osp
import tqdm

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
# from torch_geometric.data.Dataset import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
from ocpmodels.models.scn.scn_qm9 import *


target = 0
dim = 64
batch_size = 128


class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
dataset = QM9(path, transform=transform).shuffle()

# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

# Split datasets.
test_dataset = dataset[:10000]
val_dataset = dataset[10000:20000]
train_dataset = dataset[20000:]

test_loader = DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
model = HDGNN().to(device)
# model = 
# model2 = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=1,
                                                       min_lr=0.00001)



def train(epoch):
    model.train()
    loss_all = 0
    k = 0
    for data in tqdm.tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        # if k % 500 == 0:
            # print(f'Loss: {loss:.7f}') 
    return loss_all / len(train_loader.dataset)

def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R

def test(loader, MAD=False):
    model.eval()
    error = 0
    MAD_error = 0
    k = 0
    for data in tqdm.tqdm(loader):
        data = data.to(device)
        error_temp = (model(data) * std - data.y * std).abs()
        if MAD:
            rotation_data = data.clone()
            rot = torch.Tensor(data=random_rotation(np.pi)).to(device)
            rotation_data.pos = torch.einsum('ck, bk -> bc', rot, rotation_data.pos)
            MAD_error_temp = (error - (model(rotation_data) * std - data.y * std).abs()).abs()
            MAD_error += MAD_error_temp.sum().item() # MAD 
        error += error_temp.sum().item()  # MAE
        # error += (model(data) - data.y).abs().sum().item()
    return error / len(loader.dataset), MAD_error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 50):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error, _ = test(val_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error, MAD_error = test(test_loader, MAD=True)
        best_val_error = val_error

    print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
          f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}, Test MAD:{MAD_error:.7f}')