from CAE import autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets, transforms
import numpy as np

import argparse
import json
import torch
from torchvision.utils import save_image
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 96, 96)
    return x

def image_loader(path, batch_size):
    
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ]))
    
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader_unsup


def train(model, device, train_loader, criterion,  optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data, batch_idx in train_loader:
            img = data.to(device)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch+1, num_epochs, loss.data.item())) # loss.data[0]
            if epoch % 10 == 0:
                pic = to_img(output.cpu().data)
                save_image(pic, './dc_img/image_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN_training')
    parser.add_argument('--data_dir', type=str, default='/beegfs/pmh314-share/ssl_data_96/',
                        help='location of data')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1008, metavar='S',
                        help='random seed')
    parser.add_argument('--epoch', type=int, default=15,
                        help='number of epochs trained (default: 15')
    parser.add_argument('--result_dir', type=str, default='/beegfs/pmh314-share/CAE/result.txt',
                        help='location of result')
    parser.add_argument('--model_name', type=str, default='CAE',
                        help='model name')

    # Parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    args.device = torch.device("cuda" if args.cuda else "cpu")


    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    unsup = image_loader(args.data_dir, args.batch_size)

    model = autoencoder()
    model.to(args.device)

    with open(args.result_dir, "w") as f:
        f.write(args.model_name+'--start\n')

    # Optimizer: SGD with learning rate of 1e-2 and momentum of 0.5
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.MSELoss()
    # Training loop with 15 epochs
    train(model, args.device, unsup, criterion, optimizer, args.epoch)

    with open(args.result_dir, "a") as f:
        f.write(args.model_name+'--end\n\n')
    torch.save(model.state_dict(), 'weight_100g.pth')


    
