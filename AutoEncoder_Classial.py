import json
import os
import numpy as np
from time import time

import torch
import matplotlib.pyplot as plt

import Class_Training

def Ouput_Figure(X_input, target, input_s, file_path, show, save):

    for i in range(len(input_s)):
        plt.scatter(X_input[target == i, 0], X_input[target == i, 1], label=input_s[i], alpha=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if file_path != None and save:
        plt.savefig(file_path)
    if show:
        plt.show()
    plt.cla()

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

def Calculation_SVM(train_result, train_target, test_result, test_target, input_s):
    from sklearn.svm import SVC
    svm = SVC(degree = len(input_s))
    svm.fit(train_result, train_target)

    Prob = svm.score(test_result, test_target)
    return Prob

def data_linit_0_1(data, min_d, max_d):
    return (data-min_d)/max_d

def ClassResult(X_Train, X_Test, input_s, dim, dim_to, min_d, max_d, device):
    import torch.nn as nn

    X_batch, X_test = [], []
    for i in range(len(input_s)):
        X_batch = X_batch + X_Train[input_s[i]]
        X_test = X_test + X_Test[input_s[i]]

    X_batch = data_linit_0_1(np.array(X_batch), min_d, max_d)
    X_test = data_linit_0_1(np.array(X_test), min_d, max_d)

    Class_AE = Class_Training.AutoEncoder([dim,dim_to]).to(device)
    #loss_function = nn.BCELoss().to(device)
    loss_function = nn.MSELoss().to(device)

    optimizer_AE = torch.optim.AdamW(Class_AE.parameters(), lr=0.002, weight_decay=0.001)
    scheduler_AE = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_AE, T_0=10, T_mult=1, eta_min=0.01, last_epoch=-1)

    Class_Training.Training(Class_AE, optimizer_AE, X_batch, X_test, loss_function, 8, 64, device, scheduler_AE)

    return Class_AE

input_s_list = ["01"]
if __name__ == '__main__':
    dim = 8
    dim_to = 1
    input_s = "01"
    num_of_test = 200*2
    device = torch.device("cuda")

    for input_s in input_s_list:

        with open(f'Data/Dim{dim}_mnist/mnist_train_{input_s}.json', newline='') as jsonfile:
            X_Train = json.load(jsonfile)
        with open(f'Data/Dim{dim}_mnist/mnist_test_{input_s}.json', newline='') as jsonfile:
            X_Test = json.load(jsonfile)
        min_d = np.array(X_Test["0"]).min()
        max_d = np.array(X_Test["0"]).max()
        print(min_d, max_d,"\n\n")
        min_d, max_d = 0, 0.4

        for it in range(1):
            start = time()
            input_s = "01"
            Class_AE = ClassResult(X_Train, X_Test, input_s, dim, dim_to, min_d, max_d, device)

            for i in range(8):
                input_s = f"0{i+1}"
                X_output_test, X_output_train = [], []
                X_0_output, X_o_output = [], []
                target = []
                batch = int(num_of_test/len(input_s))
                for i in range(len(input_s)):
                    perm_in = torch.randperm(len(X_Train[input_s[i]]))
                    batch = len(X_Train[input_s[i]])
                    X_input = [X_Train[input_s[i]][index] for index in perm_in[0:batch]]
                    X_input = data_linit_0_1(np.array(X_input), min_d, max_d)

                    X_output_train = X_output_train + Class_Training.Testing(Class_AE, X_input, batch, device)[0]

                    perm_in = torch.randperm(len(X_Test[input_s[i]]))
                    batch = len(X_Test[input_s[i]])
                    X_input = [X_Test[input_s[i]][index] for index in perm_in[0:batch]]
                    X_o_output = X_o_output + X_input                                                       #-------------
                    X_input = data_linit_0_1(np.array(X_input), min_d, max_d)
                    X_output_test = X_output_test + Class_Training.Testing(Class_AE, X_input, batch, device)[0]
                    X_0_output = X_0_output + Class_Training.Testing(Class_AE, X_input, batch, device)[1]   #-------------

                    target = target + [i for _ in range(batch)]


                X_output_test = np.array(X_output_test)
                X_output_train = np.array(X_output_train)
                target = np.array(target)

                result_data_output_dir = f"Result_class//{dim}_{dim_to}_{input_s}"
                if os.path.exists(result_data_output_dir) == False:
                    os.makedirs(result_data_output_dir)

                save_result = True
                show_fig = True
                save_fig = True

                if dim_to != 1:
                    output_data_name = os.path.join(result_data_output_dir, f"umap_{it}.png")
                    Show_Umap(X_output_test, target, input_s, output_data_name, show_fig, save_fig)

                    output_data_name = os.path.join(result_data_output_dir, f"Classical_umap_{it}.png")
                    Ouput_Figure(X_output_test, target, input_s, output_data_name, show_fig, save_fig)

                Prob = Calculation_SVM(X_output_test, target, X_output_test, target, input_s)
                print(f"Correct Probility: {Prob}\n")


            if save_result == True:
                with open(os.path.join(result_data_output_dir, "result.txt"), "a") as file:
                    file.write(f"Correct Probility: {Prob}" + "\n")
                    file.write(f"\n")

            stop = time()
            print(f'finished running.\ntime spent:{stop - start}s')
