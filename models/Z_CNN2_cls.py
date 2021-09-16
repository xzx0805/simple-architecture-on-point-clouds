import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from Z_CNN import TransformerBlock,PointNet,z_order,CNN,VGG,CNN2

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.normal_channel=True
        self.cnn=CNN2(channel=channel)
        self.fc1= nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3=nn.Linear(256,k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x,xyz,feature=z_order(x)
        x=self.cnn(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss

if __name__ == '__main__':

    x=torch.randn(64,6,1024)
    #print('x',x)
    model=get_model()
    x=model(x)