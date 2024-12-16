import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as dset
from torchvision import datasets, transforms
from torchinfo import summary

import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class AE_Encoder(nn.Module):
    def __init__(self, dim):
        super(AE_Encoder,self).__init__()
        self.FC  = nn.Sequential(nn.Linear(dim[0], int(dim[0]/2)),
                                 nn.Tanh(),
                                 nn.Linear(int(dim[0]/2), int(dim[0]/2)),
                                 nn.Linear(int(dim[0]/2), dim[1]),
                                 nn.Tanh(),
                                 )
    def forward(self, x):
        x = self.FC(x)
        return x

class AE_Decoder(nn.Module):
    def __init__(self, dim):
        super(AE_Decoder, self).__init__()
        self.FC = nn.Sequential(nn.Linear(dim[1], int(dim[0]/2)),
                                nn.Linear(int(dim[0]/2), int(dim[0]/2)),
                                nn.Tanh(),
                                nn.Linear(int(dim[0]/2), dim[0]),
                                #nn.Sigmoid(),
                                )

    def forward(self, x):
        x = self.FC(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, dim):
        super(AutoEncoder, self).__init__()
        self.encoder = AE_Encoder(dim)
        self.decoder = AE_Decoder(dim)

    def forward(self, x):
        codes = self.encoder(x)
        decoded = self.decoder(codes)
        return codes, decoded

def Training(AE, optimizer, train_data, test_data, loss_function, epoch, batch, device, scheduler = None):
    over_fitting = 0
    for j in range(epoch):
        perm_in = torch.randperm(len(train_data))
        train_data = [train_data[k] for k in perm_in]
        num = len(train_data)

        for i in tqdm(range(int(num/batch))):
            up_lim = (i+1)*batch
            if (i+1)*batch > num:
                up_lim = num

            optimizer.zero_grad()

            x = torch.tensor(train_data[i*batch : up_lim], dtype=torch.float32).to(device)
            c,out = AE(x)

            loss = loss_function(out, x)
            loss.backward()

            optimizer.step()

            if((i + 1) % int(num/64/4) == 0):
                with torch.no_grad():
                    perm_in = torch.randperm(len(test_data))
                    y = torch.tensor([test_data[k] for k in perm_in[0:128]], dtype=torch.float32).to(device).unsqueeze(1)
                    _,out_test = AE(y)
                    loss_test = loss_function(out_test, y).item()

                    y = torch.tensor([train_data[k] for k in perm_in[0:128]], dtype=torch.float32).to(device).unsqueeze(1)
                    _,out_train = AE(y)
                    loss_train = loss_function(out_train, y).item()

                    tqdm.write(f"[TRAIN] Epoch: {(j)}, Test_Loss: {round(loss_test,4)}, Train_Loss: {round(loss_train,4)}")

                    if loss_test/1.5 > loss_train:
                        over_fitting += 1
                    else:
                        over_fitting = 0

        if over_fitting > 4:
            break

        if scheduler != None and j % 2 == 0:
            scheduler.step()

def Testing(AE, test_data, batch, device):
    out_put = []
    out_out = []
    with torch.no_grad():
        num = len(test_data)
        for i in tqdm(range(int((num+batch-1)/batch))):

            up_lim = (i+1)*batch
            if (i+1)*batch > num:
                up_lim = num

            data = torch.tensor(test_data[i*batch : up_lim], dtype=torch.float32).to(device)
            x = data.to(device)
            c, out = AE(x)
            c = c.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            for j in range(len(c)):
                out_put.append(c[j])
            for j in range(len(c)):
                out_out.append(out[j])
    return [out_put, out_out]



