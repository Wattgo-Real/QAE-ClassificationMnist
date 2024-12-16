'''
import json

with open(f'Data/Dim{8}_mnist/mnist_train.json', newline='') as jsonfile:
    X_Train = json.load(jsonfile)
with open(f'Data/Dim{8}_mnist/mnist_test.json', newline='') as jsonfile:
    X_Test = json.load(jsonfile)


maxx = 0
minx = 0
IP = "0123456789"
for i in IP:
    for j in range(len(X_Train[i])):
        maxx = max(max(X_Train[i][j]), maxx)
        minx = min(min(X_Train[i][j]), minx)
    for j in range(len(X_Test[i])):
        maxx = max(max(X_Test[i][j]), maxx)
        minx = min(min(X_Test[i][j]), minx)

print(maxx, minx)
'''
'''
from libsvm.svmutil import *

y = [1, 0, 1, 0]
x = [[1, 3, 1], [-1, 4, -1], [1, 4 ,2], [-1, 2, 0]]

# 训练 SVM 模型
model = svm_train(y, x, '-t 0 -c 1 -b 1')

# 预测
y_test = [1]
x_test = [[1, 0, 1]]
p_label, p_acc, p_val = svm_predict(y_test, x_test, model)

sv_coef = model.get_sv_coef()
sv_indices = model.get_SV()
weights = [0] * len(x_test[0])
for i in range(len(sv_indices)):
    for j in range(len(sv_indices[i])):
        weights[j] += sv_coef[i][0] * sv_indices[i][j+1]  # sv_indices[i][0] 是支持向量的索引

for i in range(len(x)):
    print(x[i][0] * weights[0] + x[i][1] * weights[1] + x[i][2] * weights[2])

print("预测标签:", p_label)
print("预测准确度:", p_acc)
print("预测值:", p_val)
'''