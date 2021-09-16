import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
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
        self.fc2 = nn.Linear(d_model, d_points)
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

    def forward(self, xyz, features):
        xyz=xyz.permute(0,2,1)
        dists = square_distance(xyz, xyz)
        #print(dists.size())
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k    提取索引
        #print(knn_idx.size())
        knn_xyz = index_points(xyz, knn_idx)

        pre = features.permute(0,2,1)
        features=pre
        #print(features.size())
        x = self.fc1(features)
        #print(x.size())
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        #print('q', q.size(), 'k', k.size(), 'v', v.size())

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        #print(pos_enc.size())

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
       # print(attn.size())
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        #print(attn.size())

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        #print(res.size())
        res = self.fc2(res) + pre
        #print(res.size())
        return res, attn

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

        self.conv1 = torch.nn.Conv1d(1024, 512,3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv1d(512, 256,3,stride=2,padding=1)
        self.conv3 = torch.nn.Conv1d(256,128,3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


class CNN2(nn.Module):
    def __init__(self,channel):
        super(CNN2, self).__init__()
        self.fc1=nn.Linear(channel,64)
        self.fc2=nn.Linear(64,128)

        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.4)
        self.conv3 = torch.nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv1d(256, 512,3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv1d(512, 1024,3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool=nn.MaxPool1d(2)
        self.pool2=nn.MaxPool1d(256)

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv3(x)))
        #print(x.size())
        x=self.pool(self.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.contiguous().view(x.size()[0], -1)
        #print(x.size())
        return x

class CNN3(nn.Module):
    def __init__(self,channel):
        super(CNN3, self).__init__()
        self.fc1=nn.Linear(channel,64)
        self.fc2=nn.Linear(64,128)

        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.4)
        self.conv3 = torch.nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv1d(256, 512,3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv1d(512, 1024,3,stride=1,padding=1)
        self.conv4=torch.nn.Conv1d(1024, 2048,3,stride=1,padding=1)
        self.conv5 = torch.nn.Conv1d(2048, 4096, 3, stride=1, padding=1)
        self.pool=nn.MaxPool1d(2)
        self.pool2=nn.MaxPool1d(64)

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv3(x)))
        #print(x.size())
        x=self.pool(self.relu(self.conv1(x)))
        #print(x.size())
        x = self.pool(self.relu(self.conv2(x)))
        #print(x.size())
        x = self.pool(self.relu(self.conv4(x)))
        #print(x.size())
        x = self.pool2(self.relu(self.conv5(x)))
        #print(x.size())
        x = x.contiguous().view(x.size()[0], -1)
        #print(x.size())
        return x
class VGG(nn.Module):

    def __init__(self, channel):
        super(VGG, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 128, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(256, 512, 3, stride=1, padding=1)
        #self.conv5 = torch.nn.Conv1d(512, 1024, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv1d(64, 64, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv1d(256, 256, 3, stride=1, padding=1)
        self.conv10 = torch.nn.Conv1d(512, 512, 3, stride=1, padding=1)
        self.pooling=torch.nn.MaxPool1d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pooling(F.relu(self.conv7(x)))
        x = F.relu(self.conv2(x))
        x = self.pooling(F.relu(self.conv8(x)))
        x = self.pooling(F.relu(self.conv8(x)))
        x = F.relu(self.conv3(x))
        x = self.pooling(F.relu(self.conv9(x)))
        x = self.pooling(F.relu(self.conv9(x)))
        x = F.relu(self.conv4(x))
        x=self.pooling(F.relu(self.conv10(x)))
        x = F.relu(self.conv10(x))
        x = self.pooling(F.relu(self.conv10(x)))
        return x

class Twod_CNN(nn.Module):
    """
    输入为N个点的 C（坐标信息）+D（特征信息） 输出为S个点的C（坐标信息)+D'(特征信息）
            Input:
                xyz:  [B, C, N]
            Return:
                xyz:  [B,1024,N]
    """
    def __init__(self,channel):
        super(Twod_CNN, self).__init__()
        self.conv1 =nn.Conv2d(6, 64,3,1,1 )
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 512, 3,1,1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=x.permute(0,2,1)
        x=x.reshape(-1,32,32,6)
        x = x.permute(0, 3, 1,2)
        x = self.pool (self.relu(self.conv1(x)))
        #print(x.size())
        x =self.pool (self.relu(self.conv2(x)))
        #print(x.size())
        x = self.pool (self.relu(self.conv3(x)))
        #print(x.size())
        x = self.pool (self.relu(self.conv4(x)))
        #print(x.size())
        x=x.contiguous().view(x.size()[0],-1)
        #print(x.size())
        return x

class TwodCNNnew(nn.Module):
    """
    输入为N个点的 C（坐标信息）+D（特征信息） 输出为S个点的C（坐标信息)+D'(特征信息）
            Input:
                xyz:  [B, C, N]
            Return:
                xyz:  [B,1024,N]
    """
    def __init__(self,channel):
        super(TwodCNNnew, self).__init__()
        self.fc1=nn.Linear(channel,64)
        self.fc2=nn.Linear(64,128)
        self.fc3=nn.Linear(128,512)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)

        self.conv1 = torch.nn.Conv2d(512, 256,3,stride=2)
        self.conv2 = torch.nn.Conv2d(256, 128,3,stride=2 )
        self.conv3 = torch.nn.Conv2d(128,64,3,stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.relu(self.fc1(x))
        x =self.relu(self.dropout(self.fc2(x)))
        x =self.relu(self.dropout(self.fc3(x)))
        x=x.reshape(-1,32,32,x.size()[2])
        x = x.permute(0, 3, 1,2)
        #print(x.size())
        x = self.relu(self.bn1(self.conv1(x)))
        #print(x.size())
        x = self.relu(self.bn2(self.conv2(x)))
        #print(x.size())
        x = self.relu(self.conv3(x))
        print(x.size())
        x = x.contiguous().view(x.size()[0], -1)
        print(x.size())
        return x

class TwodCNNnew_partseg(nn.Module):
    """
    输入为N个点的 C（坐标信息）+D（特征信息） 输出为S个点的C（坐标信息)+D'(特征信息）
            Input:
                xyz:  [B, C, N]
            Return:
                xyz:  [B,1024,N]
    """
    def __init__(self,channel):
        super(TwodCNNnew_partseg, self).__init__()
        self.fc1=nn.Linear(channel,64)
        self.fc2=nn.Linear(64,128)
        self.fc3=nn.Linear(128,512)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)

        self.conv1 = torch.nn.Conv2d(512, 256,3,stride=2)
        self.conv2 = torch.nn.Conv2d(256, 128,3,stride=2 )
        self.conv3 = torch.nn.Conv2d(128,64,3,stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv1=nn.ConvTranspose2d(64,128,3,2)
        self.tconv2=nn.ConvTranspose2d(128,256,3,2)
        self.tconv3 = nn.ConvTranspose2d(256, 512, 3, 2)
    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.relu(self.fc1(x))
        x =self.relu(self.dropout(self.fc2(x)))
        x =self.relu(self.dropout(self.fc3(x)))
        x=x.reshape(-1,32,32,x.size()[2])
        x = x.permute(0, 3, 1,2)
        print(x.size())
        x = self.relu(self.bn1(self.conv1(x)))
        print(x.size())
        x = self.relu(self.bn2(self.conv2(x)))
        print(x.size())
        x = self.relu(self.conv3(x))
        print(x.size())
        x = self.tconv1(x)
        print(x.size())
        x = self.tconv2(x)
        print(x.size())
        x = self.tconv3(x)
        print(x.size())
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