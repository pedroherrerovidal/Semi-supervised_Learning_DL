import torch
import torchvision
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.image as img
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from torch.autograd import Variable
import numpy
import time
import datetime
from torch.utils.data.sampler import *
import sys
import argparse

output_dir = "./model"
d = 20
width = 96
channel = 3

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def image_loader(path, batch_size, subset_indices):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ]
    )
    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
#         shuffle=True,
        num_workers=0,
        sampler=SubsetRandomSampler(subset_indices)
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=4, stride=2),  
            nn.ReLU(True),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 40, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 60, kernel_size=4, stride=2),  
            nn.ReLU(True),
            nn.BatchNorm2d(60),
            nn.Conv2d(60, 70, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(70)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(70, 60, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(60),
            nn.ConvTranspose2d(60, 40, kernel_size=5, stride=2),  
            nn.ReLU(True),
            nn.BatchNorm2d(40),
            nn.ConvTranspose2d(40, 20, kernel_size=3, stride=1),  
            nn.ReLU(True),
            nn.BatchNorm2d(20),
            nn.ConvTranspose2d(20, 3, kernel_size=4, stride=2),  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class newBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(newBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out    

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_less_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_less_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(newBlock(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    

    def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
class combined(nn.Module):
    def __init__(self):
        super(combined, self).__init__()
        self.resnet = ResNet()
        self.conv = nn.Conv2d(70, 128, kernel_size=3, stride=1) 
        self.bn = nn.BatchNorm2d(128)
        self.vae = autoencoder()
        
        self.vae.train(False)

        for param in self.vae.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.vae.encoder(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.resnet.forward(x)
        return x
    
if __name__ == "__main__":

    
    ## parse arguments
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--ae_dir', type=str, default="./model/ae_weight_100.pth", help='location of pretrain ae mode')
    parser.add_argument('--sample_size',  type=int, default=64, help='how many sample per label')

    args = parser.parse_args()



    sample_size = args.sample_size
    subset_indices = [j+i for i in range(sample_size)for j in range(0,64000,64)]

    ## set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## load data
    sup_train, sup_val, unsup = image_loader('../ssl_data_96', 100, subset_indices)

    ## set training parameters

    model = combined()
    model.to(device)

    ae_path = args.ae_dir
    model.vae.load_state_dict(torch.load(ae_path))

    # res_path = "./model/mRes.pth"
    # model.resnet.load_state_dict(torch.load(res_path))


    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    criterion = nn.CrossEntropyLoss().cuda()

    ## set random seeds

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    epochs = 30
    tot = len(sup_train)
    top_k=5

    bestAcc = 0

    file = open("./out/train_combined_"+str(sample_size)+"_sample_record.txt","w") 

    start_time = time.time()
    file.write("training starts "+str(datetime.datetime.now())+"\n")
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0
        count = 0
        last_time = time.time()
        for x, y in sup_train:
            count += 1
            if count % 100 == 0:
                file.write("training data "+str(count)+" out of "+str(tot)+" at epoch "+str(epoch)+", time = "+str(time.time()-last_time)+"\n")
                file.write(("loss = "+str(loss.item())+"\n"))
                last_time = time.time()
            x = x.to(device)
            # ===================forward=====================
            y_hat = model(x)
            y = y.to(device)
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        file.write(f'====> Epoch: {epoch} Average loss: {train_loss / len(sup_train):.4f}')
        file.write("\n")

        
        # Testing
        
        with torch.no_grad():
            model.eval()
            test_loss = 0
            n_samples = 0.
            n_correct_top_1 = 0
            n_correct_top_k = 0
            
            for x, y in sup_val:
                x = x.to(device)
                y = y.to(device)
                
                batch_size = x.size(0)
                n_samples += batch_size
                # ===================forward=====================
                y_hat = model(x)
                test_loss += criterion(y_hat, y).item()
                # ===================log========================
                # Top 1 accuracy
                pred_top_1 = torch.topk(y_hat, k=1, dim=1)[1]
                n_correct_top_1 += pred_top_1.eq(y.view_as(pred_top_1)).int().sum().item()

                # Top k accuracy
                pred_top_k = torch.topk(y_hat, k=top_k, dim=1)[1]
                target_top_k = y.view(-1, 1).expand(batch_size, top_k)
                n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()

        # Accuracy
        top_1_acc = n_correct_top_1/n_samples
        top_k_acc = n_correct_top_k/n_samples

        # Log
        file.write(f'top 1 accuracy: {top_1_acc:.4f}'+"\n")
        file.write(f'top {top_k} accuracy: {top_k_acc:.4f}'+"\n")
        
        test_loss /= len(sup_val)
        file.write(f'====> Test set loss: {test_loss:.4f}'+"\n")
        end_time = time.time()
        file.write("training ends: "+str(datetime.datetime.now())+"\n")
        file.write("total time: "+str(end_time-start_time)+"\n")
        
        if (top_1_acc>bestAcc):
            bestAcc = top_1_acc
            model_file = os.path.join(output_dir, 'model_combined_'+str(sample_size)+'_sample.pth'.format(epoch))
            file.write("save model "+"\n\n")
            torch.save(model.state_dict(), model_file)