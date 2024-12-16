import sys
from IPython.core.ultratb import ColorTB

sys.excepthook = ColorTB()

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms
from torchinfo import summary
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt
import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tqdm import tqdm
from torch.utils.data import Dataset

device = torch.device("cuda")

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

trainSet = datasets.MNIST(root='MNIST', download=True, train=True) # transform=transform
testSet = datasets.MNIST(root='MNIST', download=True, train=False) # transform=transform

#trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True, num_workers = 2)
#testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.Tanh(),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.Tanh(),)

    def forward(self,x):
        return self.conv(x)

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y)
        self.n_samples = x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples

class AE_Encoder(nn.Module):
    def __init__(self, dim):
        super(AE_Encoder, self).__init__()

        model = []
        model2 = []
        size_cnn = [1,32,32,64]
        size_fc = [64*2*2,64,dim]
        for i in range(len(size_cnn)-2):
            model.append(DoubleConv(size_cnn[i], size_cnn[i+1]))
            model.append(nn.MaxPool2d(kernel_size=2))
        model.append(nn.Conv2d(size_cnn[-2], size_cnn[-1], kernel_size=4, stride=1))
        model.append(nn.BatchNorm2d(size_cnn[-1]))
        model.append(nn.Tanh())
        model.append(nn.MaxPool2d(kernel_size=2))

        for i in range(len(size_fc)-1):
            model2.append(nn.Linear(size_fc[i], size_fc[i+1]))
            model2.append(nn.Tanh())

        self.cnn_model = nn.Sequential(*model)
        self.fc_model = nn.Sequential(*model2)

    def forward(self, x):
        x = self.cnn_model(x).view(x.shape[0],64*2*2)
        x = self.fc_model(x)
        return x

class AE_Decoder(nn.Module):
    def __init__(self, dim):
        super(AE_Decoder, self).__init__()
        model = []
        model2 = []

        size_cnn = [64,32,32,9]
        size_fc = [dim,64,64*2*2]
        model.append(nn.ConvTranspose2d(size_cnn[0], size_cnn[0], kernel_size=2, stride=2))
        model.append(nn.ConvTranspose2d(size_cnn[0], size_cnn[0], kernel_size=2, stride=2))
        model.append(nn.Conv2d(size_cnn[0], size_cnn[1], kernel_size=4, stride=1, padding=1))
        model.append(nn.BatchNorm2d(size_cnn[1]))
        model.append(nn.Tanh())
        for i in range(1,len(size_cnn)-1):
            model.append(nn.ConvTranspose2d(size_cnn[i], size_cnn[i], kernel_size=2, stride=2))
            model.append(DoubleConv(size_cnn[i], size_cnn[i+1]))

        for i in range(len(size_fc)-1):
            model2.append(nn.Linear(size_fc[i], size_fc[i+1]))
            model2.append(nn.Tanh())

        model.append(nn.Conv2d(size_cnn[-1], 1, kernel_size=1))
        #model.append(nn.Sigmoid())
        model.append(nn.Tanh())
        self.cnn_model = nn.Sequential(*model)
        self.fc_model = nn.Sequential(*model2)

    def forward(self, x):
        x = self.fc_model(x).view(x.shape[0],64,2,2)
        x = self.cnn_model(x).view(x.shape[0],1,28,28)
        return x

class AE_Encoder_Fc(nn.Module):
    def __init__(self):
        super(AE_Encoder_Fc,self).__init__()

        self.fc = nn.Linear(784, 256)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.tanh(self.fc(x.view(x.shape[0],784)))
        return x

class AE_Decoder_Fc(nn.Module):
    def __init__(self):
        super(AE_Decoder_Fc, self).__init__()

        self.fc = nn.Linear(256, 784)

    def forward(self, x):
        x = self.fc(x).view(x.shape[0],1,28,28)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, dim):
        super(AutoEncoder, self).__init__()
        if dim == 256:
            self.encoder = AE_Encoder_Fc()
            self.decoder = AE_Decoder_Fc()
        else:
           self.encoder = AE_Encoder(dim)
           self.decoder = AE_Decoder(dim)

    def forward(self, x):
        codes = self.encoder(x)
        decoded = self.decoder(codes)
        return codes, decoded

def Training(AE, optimizer, train_data, test_data, loss_function, epoch, batch, device, scheduler = None):
    for j in range(epoch):
        perm_in = torch.randperm(len(train_data))
        train_data = [train_data[k] for k in perm_in]

        num = len(train_data)

        for i in tqdm(range(int(num/batch))):
            up_lim = (i+1)*batch
            if (i+1)*batch > num:
                up_lim = num

            x = torch.tensor(train_data[i*batch : up_lim], dtype=torch.float32).to(device).unsqueeze(1)
            c,out = AE(x)

            optimizer.zero_grad()
            loss = loss_function(out, x)
            loss.backward()

            optimizer.step()

            if (i % (int(num/64/4)) == 0):
                perm_in = torch.randperm(len(test_data))
                test_data = [test_data[k] for k in perm_in]
                with torch.no_grad():
                    y = torch.tensor(test_data[0:64], dtype=torch.float32).to(device).unsqueeze(1)
                    c,out_test = AE(y)
                    test_loss = loss_function(out_test, y).item()

                    y = torch.tensor(train_data[0:64], dtype=torch.float32).to(device).unsqueeze(1)
                    c,out_train = AE(y)
                    train_loss = loss_function(out_train, y).item()
                    tqdm.write(f"[TRAIN] Epoch: {(j)}, test_loss: {round(test_loss,4)}, train_loss: {round(train_loss,4)}")

        if scheduler != None:
            scheduler.step()

def Cover_To_dim8(AE, trainLoader, testLoader, input_s, file, dim, device):
    train_dim8 = {}
    test_dim8 = {}
    batch = 64
    transform = T.Resize(size = (16,16))
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((16,16)),
        T.ToTensor()
    ])
    with torch.no_grad():
        for N in range(10):
            if str(N) not in input_s:
                continue

            num = len(trainLoader[str(N)])
            S = np.zeros([num,dim])
            if dim == 16 or dim == 8:
                for i in tqdm(range(int((num+batch-1)/batch))):
                    up_lim = (i+1)*batch
                    if (i+1)*batch > num:
                        up_lim = num

                    data = torch.tensor(trainLoader[str(N)][i*batch : up_lim], dtype=torch.float32).to(device).unsqueeze(1)
                    x = data.to(device)
                    c = (AE(x)[0]).detach().cpu().numpy()*3.14
                    S[i*batch : up_lim] = c
                train_dim8[N] = S.tolist()
            elif dim == 256:
                for i in range(num):
                    img = (tf(trainLoader[str(N)][i].astype(np.float32)).numpy())
                    img = [img[0][p][q] for p in range(16) for q in range(16)]
                    S[i] = img
                train_dim8[N] = S.tolist()

    with torch.no_grad():
        for N in range(10):
            if str(N) not in input_s:
                continue

            num = len(testLoader[str(N)])
            S = np.zeros([num,dim])
            if dim == 16 or dim == 8:
                for i in tqdm(range(int((num+batch-1)/batch))):
                    up_lim = (i+1)*batch
                    if (i+1)*batch > num:
                        up_lim = num

                    data = torch.tensor(testLoader[str(N)][i*batch : up_lim], dtype=torch.float32).to(device).unsqueeze(1)
                    x = data.to(device)
                    c = (AE(x)[0]).detach().cpu().numpy()*3.14
                    S[i*batch : up_lim] = c
                test_dim8[N] = S.tolist()
            elif dim == 256:
                for i in range(num):
                    img = (tf(testLoader[str(N)][i].astype(np.float32)).numpy())
                    img = [img[0][p][q] for p in range(16) for q in range(16)]
                    S[i] = img
                test_dim8[N] = S.tolist()

    with open(file[0], 'w') as f:
        json.dump(train_dim8, f)
    with open(file[1], 'w') as f:
        json.dump(test_dim8, f)

def Show_Umap(X_result, Y_target, input_s, file_path, show, save):
    import umap
    reducer = umap.UMAP()
    umap_results = reducer.fit_transform(X_result)
    for i in range(len(input_s)):
        plt.scatter(umap_results[Y_target == i, 0], umap_results[Y_target == i, 1], label=input_s[i], alpha=1)
    plt.title(f'UMAP')
    plt.legend()
    if file_path != None and save:
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.cla()

trainSet_list = (trainSet.data.numpy() / 255 - 0.1307) / 0.3081
testSet_list = (testSet.data.numpy() / 255 - 0.1307) / 0.3081

trainSet_list_targets = trainSet.targets.numpy()
testSet_list_targets = testSet.targets.numpy()

mnist_train = {}
mnist_test = {}

for i in "0123456789":
    OutB = np.zeros([len(trainSet_list_targets)], dtype=bool)
    for j in range(len(trainSet_list_targets)):
        if trainSet_list_targets[j] == int(i):
            OutB[j] = 1
    mnist_train[i] = trainSet_list[OutB]

    OutB = np.zeros([len(testSet_list_targets)], dtype=bool)
    for j in range(len(testSet_list_targets)):
        if testSet_list_targets[j] == int(i):
            OutB[j] = 1
    mnist_test[i] = testSet_list[OutB]

out_dim_list = [8,16]               # Compressed size
input_s_list = ["01","018","0123"]  # Mnist Number
for out_dim in out_dim_list:

    loss_function_BCE = nn.BCELoss().to(device)
    loss_function_MSE = nn.MSELoss().to(device)

    for input_s in input_s_list:
        print(f"Compressed size = {out_dim}, Mnist Number = {input_s}, Train Num = {[len(mnist_train[i]) for i in input_s]}, Test Num = {[len(mnist_test[i]) for i in input_s]}")

        file_train = f"Data\\Dim{out_dim}_mnist\\Mnist_train_{input_s}_t.json"
        file_test = f"Data\\Dim{out_dim}_mnist\\Mnist_test_{input_s}_t.json"
        AE_PATH = f"Data\\Dim{out_dim}_mnist\\AE_CNN_{input_s}_t.pth"
        umap_save = f"Data\\Dim{out_dim}_mnist\\{input_s}_t.png"
        AE = AutoEncoder(out_dim)
        AE = AE.to(device)

        file = [file_train, file_test]

        optimizer_AE = optim.AdamW(AE.parameters(), lr=0.001)
        scheduler_AE = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_AE, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)

        train_data = [b for a in [mnist_train[index] for index in input_s] for b in a]
        test_data = [b for a in [mnist_test[index] for index in input_s] for b in a]

        Load_Network = False
        if os.path.isfile(AE_PATH) and Load_Network:
            print("Find_AE_Network.  Load...")
            checkpoint = torch.load(AE_PATH)
            AE.load_state_dict(checkpoint['Net'])
        else:
            Training(AE, optimizer_AE, train_data, test_data, loss_function_MSE, 10, 64, device, scheduler_AE)
            if os.path.exists(f"Data\\Dim{out_dim}_mnist") == False:
                os.mkdir(f"Data\\Dim{out_dim}_mnist")
            torch.save({'Net': AE.state_dict()}, AE_PATH)
            Cover_To_dim8(AE, mnist_train, mnist_test, input_s, file, out_dim, device)

        X_output, X_output_train = [], []
        target = []
        batch = 128
        for i in range(len(input_s)):
            t_len = len(mnist_train[input_s[i]])
            perm_in = torch.randperm(t_len)
            X_input = [mnist_train[input_s[i]][index] for index in perm_in[0:t_len]]
            with torch.no_grad():
                y = torch.tensor(X_input, dtype=torch.float32).to(device).unsqueeze(1)
                c,out = AE(y)

                if len(X_output_train) == 0:
                    X_output_train = c.cpu().numpy()
                else:
                    X_output_train = np.append(X_output_train, c.cpu().numpy(), axis=0)
            target = target + [i for _ in range(t_len)]

        Show_Umap(X_output_train, np.array(target), input_s, umap_save, False, True)

        print(input_s)



