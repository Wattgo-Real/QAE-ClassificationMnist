import re
import pennylane as qml
#from pennylane import numpy as np
import numpy as np
from tqdm import tqdm
from time import time

import Training
import json
import os

import torch
import QuantumCircuit.QAE as QAE_circuit

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt

#import umap

def Get_File_Data(file, it, request = "Params"):
    text_split = []
    with open(file, 'r') as f:
        text = [line for line in f.readlines()]

    Interval = 0
    for i in range(20):
        if text[i] == '\n':
            Interval = i+1
            break

    for i in range(int((len(text)+1) / Interval)):
        if it == i:
            text_split = [text[j].replace("\n","") for j in range(i*Interval, (i+1)*Interval-1)]
            break
    for i in range(len(text_split)):
        if text_split[i][0] == "[":
            text_split[i] = text_split[i].replace("[","").replace("]","").replace(" ","").replace("\'","").split(",")

    if request == "Params":
        Params = [[],[],[]]
        for i in range(3):
            if text_split[i][0] != '':
                Params[i] = [float(text_split[i][j].replace("tensor(","").replace(")","")) for j in range(len(text_split[i]))]
        return Params
    elif request == "Loss":
        return [float(text_split[3][j]) for j in range(len(text_split[3]))]
    elif request == "Number":
        return text_split[-3].rsplit(" ",1)[-1]
    elif request == "Turn":
        if text_split[-3][0] == "R":
            return [float(text_split[-3].rsplit(" ",1)[-1]), float(text_split[-2].rsplit(" ",1)[-1])]
        else:
            return [float(text_split[-2][j]) for j in range(len(text_split[-2]))]
    elif request == "Prob":
        return float(text_split[-1].rsplit(" ",1)[-1])

def Show_Umap(X_result, Y_target, input_s, file_path):
    return
    reducer = umap.UMAP()
    umap_results = reducer.fit_transform(X_result)
    for i in range(len(input_s)):
        plt.scatter(umap_results[Y_target == i, 0], umap_results[Y_target == i, 1], label=input_s[i], alpha=1)
    plt.title(f'UMAP')
    plt.legend()
    plt.savefig(file_path)
    plt.show()

def Calculation_Prob(Circuit, X_Train, X_Test, input_s, num_of_set, file_path, use_turn = True, show_umap = True):
    if use_turn == False:
        Test_result, Test_target = Quantum_Result(Circuit, X_Test, input_s, num_of_set, use_turn = False)
        Train_result, Train_target = Quantum_Result(Circuit, X_Train, input_s, num_of_set, use_turn = False)
        if show_umap == True:
            Show_Umap(Train_result, Train_target, input_s, file_path)

        Prob = Calculation_SVM(Train_result, Train_target, Test_result, Test_target, input_s)

    elif use_turn == True:
        if Circuit.Turn == None or show_umap == True:
            Train_result, Train_target = Quantum_Result(Circuit, X_Train, input_s, num_of_set, use_turn = False)
            if show_umap == True:
                Show_Umap(Train_result, Train_target, input_s, file_path)

            if Circuit.Turn == None:
                turn = []
                for i in range(int(len(Train_result[0])/3)):
                    t = Calculation_SVM_Turn(Train_result[:,i*3:(i+1)*3], Train_target)
                    turn.append(t[0])
                    turn.append(t[1])
                Circuit.Turn = turn
        Test_result, Test_target = Quantum_Result(Circuit, X_Test, input_s, num_of_set, use_turn = True)

        if len(Circuit.Turn) == 2:
            num_correct = 0
            for k in range(len(Test_result)):
                if (Test_result[k] > 0 and Test_target[k] == 0) or (Test_result[k] < 0 and Test_target[k] == 1):
                    num_correct += 1
            Prob =  num_correct / len(Test_result)
            if Prob < 0.5:
                Prob = 1 - Prob
        else:
            Train_result, Train_target = Quantum_Result(Circuit, X_Test, input_s, num_of_set, use_turn = True)
            Prob = Calculation_SVM(Train_result, Train_target, Test_result, Test_target, input_s)

    return Prob

def Calculation_SVM_Turn(train_result, train_target):
    from sklearn.svm import LinearSVC
    svm = LinearSVC(fit_intercept = False)
    svm.fit(train_result, train_target)
    svm_coef = np.array(svm.coef_[0])
    svm_coef = svm_coef / np.linalg.norm(svm_coef)
    rx = np.arctan(svm_coef[1] / (np.sqrt(svm_coef[0]**2 + svm_coef[2]**2)))
    ry = np.arctan(-svm_coef[0] / (np.sqrt(svm_coef[1]**2 + svm_coef[2]**2)))
    if svm_coef[2] < 0:
        rx, ry = -rx, -ry

    return [rx.item(), ry.item()]

def Calculation_SVM(train_result, train_target, test_result, test_target, input_s):
    from sklearn.svm import SVC
    svm = SVC(degree = len(input_s))
    svm.fit(train_result, train_target)

    Prob = svm.score(test_result, test_target)
    return Prob

def Calculation_Reconstruction_Success_Rate(Circuit, X_Test, input_s, num_of_set):
    result = Quantum_Result(Circuit, X_Test, input_s, num_of_set, use_turn = False, get = "All")
    result = np.sum(result, axis = 0) / len(result)
    return 1-np.sum(result)/len(result)

def Quantum_Result(Circuit, X_input, input_s, num_of_test, use_turn = False, get = "Encoding"):
    batch = int(num_of_test / len(input_s))

    X_batch = []
    Output_target = []
    for i in range(len(input_s)):
        perm_in = torch.randperm(len(X_input[input_s[i]]))
        if batch > len(perm_in):
            round = int(batch/len(perm_in))
            for j in range(round):
                X_batch = X_batch + [X_input[input_s[i]][index] for index in perm_in]
            X_batch = X_batch + [X_input[input_s[i]][index] for index in perm_in[0:batch - round*len(perm_in)]]
        else:
            X_batch = X_batch + [X_input[input_s[i]][index] for index in perm_in[0:batch]]
        Output_target = Output_target + [i for j in range(batch)]
    X_batch = torch.Tensor(X_batch)

    Output_target = np.array(Output_target)
    Output_result = []
    if get == "Encoding":
        if Circuit.Turn == None or use_turn == False:
            for i in tqdm(range(len(X_batch))):
                result = Circuit.QAE(X_batch[i], 'Encoding')
                Output_result.append(np.array([np.sin(result[i].item() * np.pi / 2) for i in range(len(result))]))
                for j in range(len(result)):
                    Output_result[i][j*3:(j+1)*3] = Output_result[i][j*3:(j+1)*3] / np.linalg.norm(Output_result[i][j*3:(j+1)*3])
        else:
            for i in tqdm(range(len(X_batch))):
                result = Circuit.QAE(X_batch[i], 'Turn_Encoding')
                Output_result.append(np.array([result[i].item() for i in range(len(result))]))
        return np.array(Output_result), np.array(Output_target)
    elif get == "All":
        for i in tqdm(range(len(X_batch))):
            result = Circuit.QAE(X_batch[i], get)
            Output_result.append(np.array([result[i][1].item() for i in range(len(result))]))
        return np.array(Output_result)

def Ouput_BlochSphere(Circuit, X_Input, input_s, num_of_set, file_path):
    for type in range(2):
        result, target = Quantum_Result(Circuit, X_Input, input_s, num_of_set, use_turn = False)
        print(result)
        import qutip
        b = qutip.Bloch()
        b.clear()
        theta, phi  = -60, 10
        b.view = [theta, phi]
        init = np.array([[0],[1],[0]])
        rotate_xaxis = np.array([[1, 0, 0],
                                [0, np.cos(phi*np.pi/180), -np.sin(phi*np.pi/180)],
                                [0, np.sin(phi*np.pi/180), np.cos(phi*np.pi/180)]])
        rotate_zaxis = np.array([[np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180), 0],
                                [np.sin(theta*np.pi/180), np.cos(theta*np.pi/180), 0],
                                [0, 0, 1]])
        view_plane = rotate_zaxis @ rotate_xaxis @ init

        for i in range(len(input_s)):
            x,y,z,size = [],[],[],[]
            z_add = 0
            interval = int(num_of_set/len(input_s))
            for j in range(interval*i, interval*(i+1)):
                x.append(result[j][0])
                y.append(result[j][1])
                z.append(result[j][2])
                size.append(15)
                z_add += result[j][2]
            b.point_marker = 'o'
            b.sphere_alpha = 0.3
            b.point_size = size
            b.add_points([x, y, z])

        color = ['r', 'b']

        b.color = color
        if Circuit.Turn == None:
            b.save(dirc=file_path, format = "Bloch_NoTurn.png")
        else:
            b.save(dirc=file_path, format = "Bloch_WithTurn.png")
        QCNN_QAE_Circuit.Turn = None


if __name__ == "__main__":
    # Structure settings
    dim = 8                                     # 8, 256
    embedding_type = 'Angle'                    # Angle, Amplitude
    model = 'QCNN_Half'                         # QCNN_Half, QCNN_Full, RX_Half, RX_Full, UGate_Half, UGate_Full
    U_structure = 'U_1'                         # U_1, U_2, U_3, None (15, 4, 6, 0)
    V_structure = 'ZX'                          # Generalized, ZX, None (6, 2, 0)
    layers = 3                                  # 1, 2, 3
    N_params = [15*1*layers, 2*1*layers, 0]     # QCNN_Half->[15*1*layers, 2*1*layers, 0], RX_Half->[8*1, 7*8*1, 8*1], UGate_Half->[4*28*2*1, 0, 0]
    Qubit_Num = 8                               # 8, 16


    # Training settings
    Step = 4
    Batch_size = 4
    Y_target = None
    input_s = "01"  # Mnist Number


    # Result settings
    use_turn = True
    show_umap = False
    show_reconstruction_sr = True
    save_result = True
    load_from_result = False
    if len(input_s) != 2:
        use_turn = False


    # Loading data set
    input_s_in = input_s
    with open(f'Data/Dim{dim}_mnist/mnist_train_{input_s_in}.json', newline='') as jsonfile:
        X_Train = json.load(jsonfile)
    with open(f'Data/Dim{dim}_mnist/mnist_test_{input_s_in}.json', newline='') as jsonfile:
        X_Test = json.load(jsonfile)


    start = time()


    for it in range(1):
        # Folder Selete
        if model == 'RX_Full' or model == 'RX_Half':
            U_structure = 'None'
            V_structure = 'None'
            Get_Data_path = f'Result_Quantum/qml_refenerce/{model}.{embedding_type}.{Qubit_Num}.{layers}.{Step}&BCE&{input_s_in}/result.txt'
            Result_Data_path = f'Result/{model}.{embedding_type}.{Qubit_Num}.{layers}.{Step}&BCE&{input_s_in}'
        else:
            Get_Data_path = f'Result_Quantum/qml_refenerce/{model}.{embedding_type}.{U_structure}.{V_structure}.{Qubit_Num}.{layers}.{Step}&BCE&{input_s_in}/result.txt'
            Result_Data_path = f'Result/{model}.{embedding_type}.{U_structure}.{V_structure}.{Qubit_Num}.{layers}.{Step}&BCE&{input_s_in}'


        QCNN_QAE_Circuit = QAE_circuit.QuantumAutoEncoder(embedding_type, model, U_structure, V_structure, Qubit_Num, layers, N_params)
        #QCNN_QAE_Circuit.DataEdit = lambda x: x
        #QCNN_QAE_Circuit.N_PerLayers = 2

        print(f"{it} start")
        if load_from_result == False:
            loss_history = Training.circuit_training_torch(QCNN_QAE_Circuit, X_Train, Y_target, input_s, batch_size = Batch_size, steps = Step)


        # Load Result File Parameters  request =  "Params", "Loss", "Turn", "Prob"
        if load_from_result == True:
            print(f"-----Get {Get_Data_path} Data-----")
            it_to = 0  #it
            QCNN_QAE_Circuit.Params = Get_File_Data(Get_Data_path, it_to, request = "Params")
            #QCNN_QAE_Circuit.Turn = Get_File_Data(Get_Data_path, it_to, request = "Turn")
            loss_history = Get_File_Data(Get_Data_path, it_to, request = "Loss")
            input_s = Get_File_Data(Get_Data_path, it_to, request = "Number")


        # Output
        if os.path.exists(Result_Data_path) == False:
            os.makedirs(Result_Data_path)
        # Result_Img_path = os.path.join(Result_Data_path)
        # Ouput_BlochSphere(QCNN_QAE_Circuit, X_Test, input_s, 200*len(input_s), Result_Img_path)
        output_data_name = os.path.join(Result_Data_path, f"{it}.png")
        Prob = Calculation_Prob(QCNN_QAE_Circuit, X_Train, X_Test, input_s, 100*len(input_s), output_data_name, use_turn = use_turn, show_umap = show_umap)
        if show_reconstruction_sr == True:
            Prob_s = Calculation_Reconstruction_Success_Rate(QCNN_QAE_Circuit, X_Test, input_s, 100*len(input_s))
            print(f"{it}, Reconstruction Success Rate: {Prob_s}")

        if save_result == True:

            with open(os.path.join(Result_Data_path, "result.txt"), "a") as file:
                file.write("["+", ".join(str(element.item()) for element in QCNN_QAE_Circuit.Params[0]) + "]" + "\n")
                file.write("["+", ".join(str(element.item()) for element in QCNN_QAE_Circuit.Params[1]) + "]" + "\n")
                file.write("["+", ".join(str(element.item()) for element in QCNN_QAE_Circuit.Params[2]) + "]" + "\n")
                file.write("["+", ".join(str(element) for element in loss_history) + "]" + "\n")
                file.write(f"input_s = {input_s}" + "\n")
                if show_reconstruction_sr == True:
                    file.write(f"Reconstruction Success Rate: {Prob_s}" + "\n")
                if QCNN_QAE_Circuit.Turn == None:
                    file.write("[]" + "\n")
                else:
                    file.write("["+", ".join(str(element) for element in QCNN_QAE_Circuit.Turn) + "]" + "\n")
                file.write(f"Correct Probility: {Prob}" + "\n")
                file.write(f"\n")

        print(f"{it}, Correct Probility: {Prob}")

    stop = time()
    print(f'Finished running.\nTime spent:{stop - start}s')



