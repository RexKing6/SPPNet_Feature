import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from scipy import io

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[128, 128, 128], std=[1, 1, 1])
    # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

d_label = {}
with open('./list/cList.txt','r') as f:
    for line in f.readlines():
        tmp = line.split('/')[-1].strip('\n').strip('\r').split(' ')
        d_label[tmp[0]] = tmp[1]

class VOC07(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(img_path)
        label = d_label[img_path.split('/')[-1]]
        if self.transforms:
            data = self.transforms(data)
        if data.shape[0] == 1:
            data = torch.cat([data, data, data], 0)
        return data, label

    def __len__(self):
        return len(self.imgs)

class SppNet(nn.Module):
    def __init__(self, batch_size=1, out_pool_size=[1,2,4], class_number=2):
        super(SppNet, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        vgg = nn.Sequential(*list(vgg.children())[:-1])
        self.out_pool_size = out_pool_size
        self.batch_size = batch_size
        self.encoder = vgg
        self.spp = self.make_spp(batch_size=batch_size, out_pool_size=out_pool_size)
        sum0 = 0
        for i in out_pool_size:
            sum0 += i**2
        self.fc = nn.Sequential(nn.Linear(512*sum0, 1024), nn.ReLU(inplace=True))
        self.score = nn.Linear(1024, class_number)
    
    def make_spp(self, batch_size=1, out_pool_size=[1,2,4]):
        func = []
        for i in range(len(out_pool_size)):
            func.append(nn.AdaptiveAvgPool2d(output_size=(out_pool_size[i], out_pool_size[i])))
        return func
    
    def forward(self, x):
        assert x.shape[0] == 1 , 'batch size need to set to be 1'
        encoder = self.encoder(x)
        spp=[]
        for i in range(len(self.out_pool_size)):
            spp.append(self.spp[i](encoder))
        fc = torch.cat(spp, dim=1)
        return fc


dataset = VOC07('./c/', transforms=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=3)
spp = SppNet(out_pool_size=[6], class_number=20)
spp = spp.cuda()
# features = torch.zeros((1, 18432))
features = torch.zeros((1, 512, 6, 6))
features = features.cuda()
labels = []
for batch_data, batch_label in dataloader:
    batch_data = Variable(batch_data.cuda())
    out = spp(batch_data)
    out_np = out.data
    labels.append(int(batch_label[0]))
    features = torch.cat((features, out_np), 0)
    print(features.shape)
feature = features[1:,:]
feature = feature.cpu().numpy()
feature.shape
label = np.matrix(labels)
io.savemat('c_features.mat', {'c_feature': feature})
io.savemat('c_labels.mat', {'c_label': label})

