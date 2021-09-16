import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

"""
    Add_block:z_order
    Input: xyz BxNx3
    Output xyz,points BxN BxNxC  
"""
def round_to_int_32(data):
    #Morton Code is a 64-bit integer,So x,y,z should be a 21-bit integer
    data = 256*(data + 1)
    data = np.round(2 ** 21 - data).astype(dtype=np.int32)
    return data

def split_by_3(x):
    x &= 0x1fffff  # only take first 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249
    return x

def get_z_order(x, y, z):
    res = 0
    res |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2
    return res

def get_z_values(data):
    data = data.cpu().numpy()
    data = round_to_int_32(data)  # convert to int
    z = get_z_order(data[:, :, 0], data[:, :, 1], data[:, :, 2])
    z = torch.from_numpy(z)
    return z

def z_order_sorting(data):
    num,idx=torch.sort(data,dim=-1)
    return num,idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    # view_shape = list(idx.shape)
    # view_shape[1:] = [1] * (len(view_shape) - 1)
    # repeat_shape = list(idx.shape)
    # repeat_shape[0] = 1
    # batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # new_points = points[batch_indices, idx, :]
    if len(idx.shape) == 2:
        batch_idx = torch.arange(B).unsqueeze(-1).to(device)
        new_points = points[batch_idx, idx]
        return new_points
    else:
        batch_idx = torch.arange(B).unsqueeze(-1).to(device).unsqueeze(-1)
        new_points = points[batch_idx, idx]
        return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2=nn.Linear(d_model,d_points)
        self.fc3=nn.Linear(3,1024)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self,x):
        features= x[:, 3:, :]
        xyz = x[:, :3, :]
        xyz = xyz.permute(0, 2, 1)
        features = features.permute(0, 2, 1)
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        new_feature= torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        new_feature=self.fc2(new_feature)+pre
        #print(new_feature.size())
        new_feature=self.fc3(new_feature)
        new_feature=new_feature.permute(0,2,1)
        return  new_feature

class PointNet(nn.Module):
    """
    输入为N个点的 C（坐标信息）+D（特征信息） 输出为S个点的C（坐标信息)+D'(特征信息）
            Input:
                xyz:  [B, C, N]
            Return:
                xyz:  [B,1024,N]
    """
    def __init__(self,channel):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 2)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
       # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class CNN(nn.Module):
    """
    输入为N个点的 C（坐标信息）+D（特征信息） 输出为S个点的C（坐标信息)+D'(特征信息）
            Input:
                xyz:  [B, C, N]
            Return:
                xyz:  [B,1024,N]
    """
    def __init__(self,channel):
        super(CNN, self).__init__()
        self.fc1=nn.Linear(channel,64)
        self.fc2=nn.Linear(64,128)
        self.fc3=nn.Linear(128,1024)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)

        self.conv1 = torch.nn.Conv1d(1024, 512,64,stride=63)
        self.conv2 = torch.nn.Conv1d(512, 256,4,stride=3 )
        self.conv3 = torch.nn.Conv1d(256,128,5)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.relu(self.fc1(x))
        x =self.relu(self.dropout(self.fc2(x)))
        x =self.relu(self.dropout(self.fc3(x)))
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x



def z_order(x):
    norm = x[:, 3:, :]
    xyz = x[:, :3, :]
    xyz = xyz.permute(0, 2, 1)
    pointcode = get_z_values(xyz)
    #print('pointcode',pointcode)
    _, idx = z_order_sorting(pointcode)
    #print('idx',idx,idx.size())
    xyz = index_points(xyz, idx)

    norm = norm.permute(0, 2, 1)
    norm = index_points(norm, idx)
    x = torch.cat([xyz,norm], dim=-1)
    x=x.permute(0,2,1)
    xyz = xyz.permute(0, 2, 1)
    norm = norm.permute(0, 2, 1)
    return x,xyz,norm

if __name__ == '__main__':
    x=torch.randn(3,10,6)
    #print('x',x)
    x,_,_=z_order(x)