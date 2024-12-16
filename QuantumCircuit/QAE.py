import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
import QuantumCircuit.QCNN_Unitary as set

class QuantumAutoEncoder():
    def __init__(self, Embedding_type, Model, U_structure, V_structure, QubitNum, Layers, N_params):
        self.Embedding_type = Embedding_type
        self.Model = Model
        self.U_structure = U_structure
        self.V_structure = V_structure
        self.QubitNum = QubitNum
        self.Layers = Layers
        self.N_PerLayers = 1
        self.N_params = N_params
        self.Params = None
        self.Turn = None
        self.DataEdit = None

    def QAE(self, Input_X, get = "Encoding"):
        params_num = [0,0]   #[U_params_num, V_params_num]
        if self.U_structure == 'U_1':  # 15 params
            U = set.U_1
            params_num[0] = 15
        elif self.U_structure == 'U_2':  # 4 params
            U = set.U_2
            params_num[0] = 6
        elif self.U_structure == 'U_3':  # 6 params
            U = set.U_3
            params_num[0] = 4
        elif self.U_structure == 'U_4':  # 2 params
            U = set.U_4
            params_num[0] = 2
        elif self.U_structure == 'None':
            U = None
        else:
            print("Convolution_structure error")
            return "None"

        if self.V_structure == 'Generalized':
            V = set.Generalized_Pooling
            params_num[1] = 6
        elif self.V_structure == 'ZX':
            V = set.ZX_Pooling
            params_num[1] = 2
        elif self.V_structure == 'None':
            V = None
        else:
            print("Pooling_structure error")
            return "None"

        if self.Turn != None:
            turn = self.Turn
        else:
            turn = None

        if self.DataEdit != None:
            Input_X = [self.DataEdit(it) for it in Input_X]

        QubitNum = self.QubitNum
        if self.Model.split("_")[0] == 'QCNN':
            return MakeCircuits(Input_X, self.Embedding_type, self.Model, U, V, params_num, self.Params, self.Layers, get, QubitNum, turn)
        elif self.Model.split("_")[0] == 'RX':
            return MakeCircuits(Input_X, self.Embedding_type, self.Model, U, V, [QubitNum, QubitNum*(QubitNum-1), QubitNum], self.Params, self.Layers, get, QubitNum, turn)
        elif self.Model.split("_")[0] == 'UGate':
            return MakeCircuits(Input_X, self.Embedding_type, self.Model, U, V, [params_num[0]*int(QubitNum*(QubitNum-1)/2), 0, 0], self.Params, self.Layers, get, QubitNum, turn)

def Encoder_QCNN(U, V, params_num, params, layers, Qubits, Encoding_qubit, turn = None, qb_id = [0,1,2,3,4,5,6,7]):    #params=[convolution, pooling]  U_params = [15, 2 or 6]
    if Qubits == 16:
        qb_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    Has_pool = False
    U_param_list = [params[0][i*params_num[0] : (i+1)*params_num[0]] for i in range(layers)]
    if V != None:
        Has_pool = True
        V_param_list = [params[1][i*params_num[1] : (i+1)*params_num[1]] for i in range(layers)]

    for layer in range(layers):
        jump = 2**layer
        for i in range(-1+jump, Qubits, 2*jump):
            U(U_param_list[layer], wires=[qb_id[(i)%Qubits], qb_id[(i + jump)%Qubits]])
        if Qubits/jump != 2:
            for i in range(-1+2*jump, Qubits+jump, 2*jump):
                U(U_param_list[layer], wires=[qb_id[(i)%Qubits], qb_id[(i + jump)%Qubits]])
        if Has_pool == True:
            for i in range(-1+jump, Qubits, 2*jump):
                V(V_param_list[layer], wires=[qb_id[(i + jump)%Qubits], qb_id[(i)%Qubits]])
    if turn != None:
        for i in range(len(Encoding_qubit)):
            qml.RX(turn[i*2], wires = qb_id[Encoding_qubit[i]])
            qml.RY(turn[i*2+1], wires = qb_id[Encoding_qubit[i]])

def Encoder_RX(params, layers, Qubits, Encoding_qubit, turn = None, qb_id = [0,1,2,3,4,5,6,7]):
    for i in range(Qubits):
        qml.RY(params[0][i], wires=qb_id[i])

    c = 0
    for i in range(Qubits):
        for j in range(Qubits):
            if (i != j):
                qml.CRZ(params[1][c], wires=[qb_id[i],qb_id[j]])
                c += 1

    for i in range(Qubits):
        qml.RY(params[2][i], wires=qb_id[i])

    if turn != None:
        for i in range(len(Encoding_qubit)):
            qml.RX(turn[i*2], wires = Encoding_qubit[i])
            qml.RY(turn[i*2+1], wires = Encoding_qubit[i])

def Encoder_UGate(U, U_len, params, Qubits, Encoding_qubit, turn = None, qb_id = [0,1,2,3,4,5,6,7]):
    c = 0
    for i in range(1,Qubits):
        for j in range(Qubits-i):
            U(params[0][c * U_len : (c+1) * U_len], wires=[qb_id[j], qb_id[j+i]])
            c += 1

    it = 0
    if turn != None:
        for i in range(len(Encoding_qubit)):
            qml.RX(turn[i*2], wires = Encoding_qubit[i])
            qml.RY(turn[i*2+1], wires = Encoding_qubit[i])


dev = qml.device("default.qubit", wires=16)
@qml.qnode(dev, interface = 'torch') #interface = 'torch' # , diff_method = "parameter-shift"
def MakeCircuits(Input_X, embedding_type, model, U, V, params_num, params, layers, get, Qubits, turn):
    # Data Embedding
    if embedding_type == "Angle":
        if model.split("_")[0] == 'UGate':
            AngleEmbedding(Input_X, wires=range(Qubits), rotation='X')
        else:
            AngleEmbedding(Input_X, wires=range(Qubits), rotation='Y')
    elif embedding_type == "Amplitude":
        AmplitudeEmbedding(Input_X, wires=range(Qubits), normalize=True)


    # Encoder
    if model.split("_")[0] == 'QCNN':
        params_to = [params[j][0:layers*params_num[j]] for j in range(len(params_num))]
    else:
        params_to = [params[j][0:params_num[j]] for j in range(len(params_num))]

    Encoding_qubit = [i for i in range(-1+2**(layers), Qubits, 2*(2**(layers-1)))]
    if model.split("_")[0] == 'QCNN':
        Encoder_QCNN(U, V, params_num, params_to, layers, Qubits, Encoding_qubit, turn)
    elif model.split("_")[0] == 'RX':
        Encoder_RX(params_to, layers, Qubits, Encoding_qubit, turn)
    elif model.split("_")[0] == 'UGate':
        Encoder_UGate(U, 4, params_to, Qubits, Encoding_qubit, turn)


    # Decoder
    if get == "All" or get == "Training" and (model.split("_")[1] == 'Full'):
        if model.split("_")[0] == 'QCNN':
            if layers == 3:
                qb_id = [14,13,12,11,10,9,8,7]
            elif layers == 2:
                qb_id = [13,12,11,3,10,9,8,7]
        else:
            qb_id = [14,13,12,11,10,9,8,7]

        if model.split("_")[1] == 'Half':
            params_to = params
        elif model.split("_")[1] == 'Full':
            if model.split("_")[0] == 'QCNN':
                params_to = [params[i][layers*params_num[i]:layers*params_num[i]*2] for i in range(len(params_num))]
            else:
                params_to = [params[i][params_num[i]:params_num[i]*2] for i in range(len(params_num))]
        if model.split("_")[0] == 'QCNN':
            qml.adjoint(Encoder_QCNN)(U, V, params_num, params_to, layers, Qubits, Encoding_qubit, turn, qb_id = qb_id)
        elif model.split("_")[0] == 'RX':
            qml.adjoint(Encoder_RX)(params_to, layers, Qubits, Encoding_qubit, turn, qb_id = qb_id)
        elif model.split("_")[0] == 'UGate':
            qml.adjoint(Encoder_UGate)(U, 4, params_to, Qubits, Encoding_qubit, turn, qb_id = qb_id)

        if embedding_type == "Angle":
            if model.split("_")[0] == 'UGate':
                qml.adjoint(AngleEmbedding)(Input_X, wires=qb_id, rotation='X')
            else:
                qml.adjoint(AngleEmbedding)(Input_X, wires=qb_id, rotation='Y')

        elif embedding_type == "Amplitude":
            qml.adjoint(AmplitudeEmbedding)(Input_X, wires=qb_id, normalize=True)


    # Return
    if get == "Encoding":
        result = []
        for it in Encoding_qubit:
            result.append(qml.expval(qml.X(it)))
            result.append(qml.expval(qml.Y(it)))
            result.append(qml.expval(qml.Z(it)))
        return result
    elif get == "Turn_Encoding":
        return [qml.expval(qml.Z(it)) for it in Encoding_qubit]
    elif get == "Training":
        if model == "QCNN_Half" or model == "RX_Half" or model == "UGate_Half":
            return [qml.probs(wires = i) for i in range(0,Qubits) if i not in Encoding_qubit]
        elif model == 'QCNN_Full' or model == 'RX_Full' or model == 'UGate_Full':
            return [qml.probs(wires = i) for i in range(0,Qubits*2-len(Encoding_qubit)) if i not in Encoding_qubit]
    elif get == "All":
        return [qml.probs(wires = i) for i in range(Qubits)]
