
import torch
from tqdm import tqdm
import QuantumCircuit.QAE as QAE_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp

def MSELoss_torch(labels, predictions):
    loss = 0
    for i in range(len(predictions)):
        for l, p in zip(labels[i], predictions[i]):  # p[0] -> Probability in |0>,  p[1] -> Probability in |1>
            c_entropy  = ((l - p[1]) * (l - p[1])) ** 0.5
            loss += c_entropy
    return loss

def BCELoss_torch(labels, predictions):
    loss = 0
    for i in range(len(predictions)):
        for j in range(len(predictions[i])): # p[0] -> Probability in |0>,  p[1] -> Probability in |1>
            pred_clip = torch.clamp(predictions[i][j][1], min = 1e-7, max = 1 - 1e-7)
            c_entropy = labels[i][j] * torch.log(pred_clip) + (1 - labels[i][j]) * torch.log(1 - pred_clip)
            loss += c_entropy
    return -1 * loss

def circuit_training_torch(Circuit, X_input, Y_target, input_s, batch_size = 32, steps = 64):
    learning_rate = 0.005

    if Circuit.Params == None:
        Circuit.Params = [torch.randn(Circuit.N_params[0], requires_grad=True), torch.randn(Circuit.N_params[1], requires_grad=True), torch.randn(Circuit.N_params[2], requires_grad=True)]
    else:
        Circuit.Params = [torch.tensor(Circuit.Params[0], requires_grad=True), torch.tensor(Circuit.Params[1], requires_grad=True), torch.tensor(Circuit.Params[2], requires_grad=True)]

    opt = torch.optim.SGD(Circuit.Params, learning_rate, momentum=0.9)

    loss_history, perm_in = [],[]
    batch = int(batch_size/len(input_s))
    for i in range(len(input_s)):
        perm_in.append(torch.randperm(len(X_input[input_s[i]])))

    for it in tqdm(range(steps)):
        X_batch = []
        for i in range(len(input_s)):
            from_, to_ =  (batch * it) % len(perm_in[i]), (batch * (it+1)) % len(perm_in[i])
            if from_ > to_:
                to_ += len(perm_in[i])
            X_batch = X_batch + [X_input[input_s[i]][index] for index in perm_in[i][from_:to_]]
        X_batch = torch.tensor(X_batch)

        if Y_target == None:
            Y_batch = [torch.zeros((Circuit.QubitNum-int(Circuit.QubitNum / 2**Circuit.Layers)) + Circuit.QubitNum ) for i in range(len(X_batch))]
        else:
            Y_batch = [Y_target for i in range(len(X_batch))]

        opt.zero_grad()
        result = [Circuit.QAE(X_batch[i], 'Training') for i in range(batch_size)]
        #loss = MSELoss_torch(Y_batch, result)
        loss = BCELoss_torch(Y_batch, result)
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if it % 10 == 0:
            print(f"Iteration: {it}, Loss: {loss.item()}")

    #Circuit.Params = [Circuit.Params[0].detach().numpy(), Circuit.Params[1].detach().numpy(), Circuit.Params[2].detach().numpy()]
    Circuit.Params = [Circuit.Params[0].detach(), Circuit.Params[1].detach(), Circuit.Params[2].detach()]
    return loss_history