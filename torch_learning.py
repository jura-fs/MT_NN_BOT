import os.path

import torch
from sympy.codegen.ast import float16
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import embedding, dropout
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import json
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as tfs



'''                             задача 1             '''
'''import torch

x = torch.tensor([[10, 20], [30, 40], [50, 60]])

print(x)
print(x.shape)
print(x.device)
print(x.dtype)
'''

'''                                 задача 2                    '''

'''x = torch.tensor([1.5, 2.5, 3.5])
print(x.dtype)
'''

'''                                 задача 3                   '''

'''
def tensor_sdd(x):
    x = torch.tensor(x)
    print(x.shape)
    print(x.device)
    print(x.dtype)
'''


'''             2. Создание тензоров: torch.tensor, torch.zeros, torch.ones, torch.randn            '''

'''                             задача 1             '''

'''ten = torch.tensor(5,)
ten_z = torch.tensor(5,)
ten_o = torch.tensor(5, )

print(ten)
print(ten_o)
print(ten_z)'''

'''                             задача 2             '''

'''x = torch.randn(3, 4)

print(x, x.dtype)

x = x.to(torch.float16)

print(x, x.dtype)'''

'''                             задача 3             '''


'''normal_tensor = torch.tensor([[1, 2], [3, 4]])

not_normal_tensor_через_ебанное_заполнение_бля_какое_тупо_задание_какой_тупой_грок = torch.zeros(2, 2, dtype=torch.int64)

not_normal_tensor_через_ебанное_заполнение_бля_какое_тупо_задание_какой_тупой_грок[0,0] = 1
not_normal_tensor_через_ебанное_заполнение_бля_какое_тупо_задание_какой_тупой_грок[0,1] = 2
not_normal_tensor_через_ебанное_заполнение_бля_какое_тупо_задание_какой_тупой_грок[1,0] = 3
not_normal_tensor_через_ебанное_заполнение_бля_какое_тупо_задание_какой_тупой_грок[1,1] = 4

print(normal_tensor)
print(not_normal_tensor_через_ебанное_заполнение_бля_какое_тупо_задание_какой_тупой_грок)'''


'''                                             3. shape, dtype, device                             '''

'''                             задача 1 and 2             '''

'''x = torch.randn(4, 5)

print(x, x.dtype, x.device, x.shape)

x = x.to(torch.int64)
x = x.to(torch.float32)
print(x)
'''



'''                             4. Индексация и slicing                         '''

'''                             задача 1 and 2             '''

'''x = torch.arange(20).reshape(4, 5)
print(x)

center = x[1:3, 1:4]
print(center)

x[x % 2 == 0] = -1
print(x)'''

'''                             задача 3                     '''

'''x = torch.arange(3 * 4 * 5).reshape(3,4,5)
print(x)

print()

print(x[1,1,1])'''



'''                 5. Broadcasting               '''

'''                             задача 1                     '''

'''a = torch.randn(5, 1)
b = torch.randn(1, 3)
c = a + b
print(c, c.shape)
'''

'''                             задача 2                     '''


'''a = torch.randn(32, 10)
print(a)
print()
b = torch.randn(10)
print(b)
print()
c = a - b

print(c)
'''

'''                         7. Матричное умножение: @ и torch.matmul                        '''

'''                             задача 1                    '''

'''
a = torch.randn(5, 3)
b = torch.randn(3, 7)
c = a @ b
print(c)
'''

'''                             задача 2                     '''


'''A = torch.randn(16, 10, 20)
B = torch.randn(16, 20, 5)
C = A @ B
print(C)
'''

'''                             задача 3                     '''


'''             8. sum, mean, softmax, argmax                   '''

'''                             задача 1                     '''

'''a = torch.randn(4, 5)
print(a.sum(dim=0))
print(a.sum(dim=1))
print(a.mean(dim=0))
print(a.mean(dim=1))
'''

'''                             задача 2                     '''

'''
x = torch.randn(2, 3)
print(x.softmax(dim=-1))
'''

'''                             задача 3                     '''

'''
x = torch.randn(3 ,4)
print(x)
print()
arg = x.argmax(dim=1)
print(x[0, arg[0]], x[1, arg[1]], x[2, arg[2]])'''

'''                                             9. cat и stack                                          '''

'''a = torch.randn(3,4)
print(a)
print()
b = torch.randn(3,4)
print(b)
print()

print(torch.cat([a ,b], dim=0))
print(torch.cat([a ,b], dim=1))
'''


'''a = torch.randn(3,4)
b = torch.randn(3,4)

c = torch.stack([a,b], dim=0)
d = torch.stack([a,b], dim=2)

print(c)
print()
print(d)
print()
print(c.shape, d.shape)'''

'''a1 = torch.randn(10,8)
a2 = torch.randn(10,8)
a3 = torch.randn(10,8)
a4 = torch.randn(10,8)
a5 = torch.randn(10,8)

a6 = torch.stack([a1, a2, a3, a4, a5])

print(a6.shape)'''
'''====================================================================================================================================
========================================================================================================================
========================================================================================================================'''


'''                                             ЭТАП 2                                                  '''


'''                                             1. requires_grad                                        '''
'''x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(2.0)


print(x.requires_grad)
print(y.requires_grad)


x  = torch.tensor(2.3)
print(x.requires_grad)
x.requires_grad_(True)
print(x.requires_grad)
'''


'''                                              2. grad                                     '''


'''                        пример 
Это специальная полочка у тензора, куда PyTorch кладёт градиент после вызова backward()

x = torch.tensor(3.0, requires_grad=True)
y = x * x          # y = 9
y.backward()
print(x.grad)'''


'''                                             3. backward()                                           '''



'''x = torch.tensor(4.4, requires_grad=True)


y = (x + 5) ** 2
y.backward()
print(x.grad)

x, y, z = torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)

sum_kw = x ** 2 + y ** 2 + z ** 2
sum_kw.backward()
print(x.grad)
print(y.grad)
print(z.grad)

x = torch.randn(2,2)
y = x + 5
y.backward()
print(y)'''



'''                                   4. Computational Graph (вычислительный граф)                    '''


'''a = torch.tensor(5.0, requires_grad=True)
b = a * 2
c = b + 3
d = c ** 2

d.backward()

print(a.grad)


f = d ** 3
print(f)'''



'''                                     5. torch.no_grad()                                      '''
'''#x = torch.tensor(5.0, requires_grad=True)

#with torch.no_grad():
    #y = x ** 2
    #y.backward()
    #print(y)

import time
x = torch.tensor(1.0, requires_grad=True)

y = 5 + x - x ** 2 * x * 6 + (x ** 3 - 4 + x)
a = time.time()
y.backward()
print(x.grad, time.time() - a)

b = time.time()
y_grad = 2 - 15 * x ** 2 
print(y_grad, time.time() - b)

a = 0.005182027816772461
b = 0.0005125999450683594
print(a > b)'''


'''                                     6. detach()                                             '''


'''
x = torch.tensor(3.0, requires_grad=True)
y = x.detach()

print(y.requires_grad)

y = x ** 2
y.backward()
print(x.grad)



y = x.detach()

y.backward()
'''


'''                                 7. nn.Module                                            '''



'''
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()



    def return_none(self, x):
        return x


M = MyModel()
x = torch.tensor([1.0,2.0,3.0])
output = M(x)
print(x)
print(output)


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


model = SimpleLinearModel()
print(f"Модель: {model}")
print(f"\nКоличество параметров: {sum(p.numel() for p in model.parameters())}")
# 10 весов + 1 bias = 11 параметров

x = torch.randn(5, 10)
output = model(x)
print(f"\nВход shape: {x.shape}")
print(f"Выход shape: {output.shape}")


'''


'''                                         8. nn.Parameter                                     '''


''' тут сразу 3 задания. создание класса, добавление нового параметра и вывод всех параметров, 
а также проверка будет ли self.w параметром - нет не будет тк нету функции nn.Parameter'''



'''class ModelParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.weightt = nn.Parameter(torch.randn(4, 4))

        self.w = torch.randn(4, 4)





M = ModelParameter()
print([param for param in M.parameters()])
'''


'''                                     9. forward()                                            '''



'''class Mod(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        ret_x = x * 2


m = Mod()
x = torch.tensor(1.0)

print(m(x))
print(m.forward(x))'''


'''                                         10. model.parameters()                                      '''


'''                                 кол во параметров класса                                        '''

'''
class ModelLinear2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def return_parameters(self):
        return self.parameters()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

model = ModelLinear2()

print(f"   Всего: {model.count_parameters()}")


###########################################################################




class ModelLinear2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

model = ModelLinear2()

# Передаем model.parameters() в SGD оптимизатор
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)'''

'''
torch.manual_seed(42)

x_train = torch.linspace(-5, 5, 100).reshape(-1, 1)
y_true = 2 * x_train + 1 + 0.1 * torch.randn_like(x_train)




class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = NN()
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
num_epochs = 200

for epoch in range(num_epochs):
    y_pred = model(x_train)

    loss = criterion(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Шаг {epoch + 1:3d}/{num_epochs}: loss = {loss.item():.6f}")

print("\nОбучение завершено!")

# --- 5. СМОТРИМ ЧЕМУ НАУЧИЛАСЬ МОДЕЛЬ ---
print("\n--- Результаты обучения ---")
# Получаем параметры модели
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"Изученные параметры: weight = {w:.4f}, bias = {b:.4f}")
print(f"Истинные параметры:  weight = 2.0000, bias = 1.0000")
print(f"Ошибка в weight: {abs(w - 2.0):.4f}")
print(f"Ошибка в bias:   {abs(b - 1.0):.4f}")

# --- 6. ПРОВЕРКА НА НОВЫХ ДАННЫХ ---
print("\n--- Проверка на новых данных ---")
x_new = torch.tensor([[1.5], [-2.0], [3.0], [-4.5], [0.0]])
with torch.no_grad():  # Отключаем градиенты для предсказания
    y_new = model(x_new)

print("  x     -> предсказание (истинное)")
for i in range(len(x_new)):
    x_val = x_new[i].item()
    y_pred_val = y_new[i].item()
    y_true_val = 2 * x_val + 1  # Истинное значение без шума
    print(f"{x_val:6.2f} -> {y_pred_val:8.4f}  ({y_true_val:8.4f})")'''


'''====================================================================================================================================
========================================================================================================================
========================================================================================================================'''


'''                                             ЭТАП 3                                                  '''


'''                                     1. nn.Linear — самый важный слой                            '''

'''linear = nn.Linear(10, 5)

ten = torch.randn(8, 10)
y = linear(ten)

print(ten.shape)
print(y.shape)

print(linear.weight.shape)
print(linear.bias.shape)

linear_no_bias = nn.Linear(10,5, bias=False)
print(linear_no_bias.bias)
'''



'''                                     2. nn.Embedding — превращение номеров в векторы                 '''


'''embeddings = nn.Embedding(num_embeddings=500, embedding_dim=32)

tokens = torch.tensor([3, 10, 25])
vectors  = embeddings(tokens)

print(tokens.shape)
print()
print(vectors.shape)
print()
print(embeddings.weight.shape)
print()
# print(embeddings.weight.shape) > torch.Size([500, 32]) это означает что у нас 500 токенов и 32 это размер каждого токена

batch = torch.tensor([4, 12])
vector_batch = embeddings(batch)
print(vector_batch.shape)'''



'''                                         3. nn.LayerNorm — нормализация                              '''

'''norm = nn.LayerNorm(normalized_shape=512)

ten = torch.randn(3, 20, 512)

y = norm(ten)


layer_norm_with_elementwise_affine = nn.LayerNorm(normalized_shape=512 ,elementwise_affine=False)

print(layer_norm_with_elementwise_affine.weight)

mean_before = ten.mean(dim=-1)
std_before = ten.std(dim=-1)

mean_after = y.mean(dim=-1)
std_after = y.std(dim=-1)

print("=== ДО LayerNorm ===")
print(f"Среднее: {mean_before[0, 0].item():.4f}")  # ≈ 0.0 (для randn)
print(f"Стандартное отклонение: {std_before[0, 0].item():.4f}")  # ≈ 1.0

print("\n=== ПОСЛЕ LayerNorm ===")
print(f"Среднее: {mean_after[0, 0].item():.4f}")  # ≈ 0.0
print(f"Стандартное отклонение: {std_after[0, 0].item():.4f}")  # ≈ 1.0

layer_norm_no_affine = nn.LayerNorm(normalized_shape=512, elementwise_affine=False)
print(f"\nlayer_norm_no_affine.weight: {layer_norm_no_affine.weight}")  # None
print(f"layer_norm_no_affine.bias: {layer_norm_no_affine.bias}") '''



'''             модель на 64 параметра переписанна с numpy на pytorch не финальная версия тк нет 
                наследования от nn.Module, torch.nn.Linear, torch.nn.ReLU'''
'''


class MSET: # T значит что класс написан на PYTORCH
    @staticmethod
    def loss(pred, y):
        return torch.mean((pred - y) ** 2)

    @staticmethod
    def gradient(pred, y):
        return 2 * (pred - y) / len(y)

class ReLUT: # T значит что класс написан на PYTORCH
    def forward(self, X):
        self.cache = X
        return torch.clamp(X, min=0)

    def backward(self, grad_output, learning_rate):
        return grad_output * (self.cache > 0)

class LayerT: # T значит что класс написан на PYTORCH
    def __init__(self, size_input, size_output):
        self.cache = None
        self.weights = torch.randn(size_input, size_output) * 0.1
        self.biases = torch.zeros(size_output)

    def forward(self, X):
        self.cache = X
        return X @ self.weights + self.biases

    def backward(self, grad_output, learning_rate):
        grad_weights = self.cache.T @ grad_output
        grad_biases = torch.sum(grad_output, dim=0)

        grad_input = grad_output @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input

class NeuralNetworkT: # T значит что класс написан на PYTORCH
    def __init__(self, layer_sizes):
        self.layers = []
        self.activations = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LayerT(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.activations.append(ReLUT())
            else:
                self.activations.append(None)

    def forward(self, X):
        out = X
        for layer, activation in zip(self.layers, self.activations):
            out = layer.forward(out)
            if activation is not None:
                out = activation.forward(out)
        return out

    def backward(self, grad_output, learning_rate):
        for layer, activation in zip(reversed(self.layers), reversed(self.activations)):
            if activation is not None:
                grad_output = activation.backward(grad_output, learning_rate)
            grad_output = layer.backward(grad_output, learning_rate)
        return grad_output

    def train_step(self, X, y, learning_rate):
        pred = self.forward(X)
        loss = MSET.loss(pred, y)
        grad = MSET.gradient(pred, y)
        self.backward(grad, learning_rate)
        return loss, grad

torch.manual_seed(42)  # правильный seed для PyTorch

X = torch.randn(1000, 2)
y = ((X[:, 0] * X[:, 1] > 0).float().reshape(-1, 1))

model = NeuralNetworkT([2, 8, 8, 1])

epochs = 500
learning_rate = 0.1
batch_size = 32

print("Начинаем обучение...")
loss_history = []

for epoch in range(epochs):
    indices = torch.randperm(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    total_loss = 0

    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        loss, _ = model.train_step(X_batch, y_batch, learning_rate)
        total_loss += loss.item()  # .item() для скаляра

    avg_loss = total_loss / (len(X) / batch_size)
    loss_history.append(avg_loss)
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, средняя ошибка: {avg_loss:.6f}")

# --- ТЕСТ ---
test_X = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32)
test_y = ((test_X[:, 0] * test_X[:, 1] > 0).float().reshape(-1, 1))
preds = model.forward(test_X)

print("\nРезультат на тесте:")
print("Истина:              ", test_y.flatten().int())
print("Предсказания (округл):", torch.round(preds).flatten().int())
print("Вероятности:         ", preds.flatten())

# --- ГРАФИК ---
plt.plot(loss_history)
plt.title("График ошибки (MSE) во время обучения")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.show()'''

'''     финальная модель на 64 параметра на torch с nn.Module, torch.nn.Linear, torch.nn.ReLU   '''

'''     это моя версия она не правильная по похожа, 
        не работает тк не оч еще шарю в торче и сложно понимать как все работает'''

'''
class ReLUT:
    pass # бесполезен тк мы уже используем готовый от torch



class NeuralNetwork(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.activations = []
        self.layers = []
        self.optimizer = optim.SGD( lr=0.01)
        self.criterion = nn.MSELoss()

        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            if i < len(layer_size) - 2:
                self.activations.append(nn.ReLU())
            else:
                self.relu.append(None)

    def forwardd(self, X):
        out = X
        for layer, activation in zip(self.layers, self.activations):
            out = layer.forward(out)
            if activation is not None:
                out = activation.forward(out)
        return out



    def train_step(self, X, y, learning_rate):
        pred = self.forwardd(X)
        loss = self.criterion(pred, y)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step(learning_rate)
        return loss




torch.manual_seed(42)  # правильный seed для PyTorch

X = torch.randn(1000, 2)
y = ((X[:, 0] * X[:, 1] > 0).float().reshape(-1, 1))

model = NeuralNetwork([2, 8, 8, 1])

epochs = 500
learning_rate = 0.1
batch_size = 32

print("Начинаем обучение...")
loss_history = []

for epoch in range(epochs):
    indices = torch.randperm(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    total_loss = 0

    for i in range(0, len(X), batch_size):
        X_batch = torch.tensor((X_shuffled[i], X_shuffled[i+1:i+batch_size]), requires_grad=True)
        y_batch = y_shuffled[i:i+batch_size]

        loss, _ = model.train_step(X_batch, y_batch, learning_rate)
        total_loss += loss.item()  # .item() для скаляра

    avg_loss = total_loss / (len(X) / batch_size)
    loss_history.append(avg_loss)
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, средняя ошибка: {avg_loss:.6f}")

# --- ТЕСТ ---
test_X = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32)
test_y = ((test_X[:, 0] * test_X[:, 1] > 0).float().reshape(-1, 1))
preds = model.forward(test_X)

print("\nРезультат на тесте:")
print("Истина:              ", test_y.flatten().int())
print("Предсказания (округл):", torch.round(preds).flatten().int())
print("Вероятности:         ", preds.flatten())

# --- ГРАФИК ---
plt.plot(loss_history)
plt.title("График ошибки (MSE) во время обучения")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.show()'''


''' это правильнная версия тут все правильно работает и тут макс много торча'''

'''
# nn.Module, nn.ModuleList, nn.Identity(), nn.ReLU(), nn.Linear, nn.MSELoss(), optim.SGD,

class NeuralNetwork(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()


        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            if i < len(layer_size) - 2:
                self.activations.append(nn.ReLU())
            else:
                self.activations.append(nn.Identity())

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.1)

    def forward(self, X):
        out = X

        for layer, activation in zip(self.layers, self.activations):
            out = layer(out)
            out = activation(out)

        return out

    def train_step(self, X, y):
        pred = self.forward(X)
        loss = self.criterion(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

torch.manual_seed(42)

X = torch.randn(3000, 2)
y = ((X[:, 0] * X[:, 1] > 0).float().reshape(-1, 1))


model = NeuralNetwork([2,8,8,1])

epochs = 600
batch_size = 32


loss_history = []

for epoch in range(epochs):
    indices = torch.randperm(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    total_loss = 0
    n_batches = 0

    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        loss = model.train_step(X_batch, y_batch)
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    avg_loss = total_loss / n_batches
    loss_history.append(avg_loss)
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, средняя ошибка: {avg_loss:.6f}")

# --- ТЕСТ ---
test_X = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32)
test_y = ((test_X[:, 0] * test_X[:, 1] > 0).float().reshape(-1, 1))
preds = model.forward(test_X)

print("\nРезультат на тесте:")
print("Истина:              ", test_y.flatten().int())
print("Предсказания (округл):", torch.round(preds).flatten().int())
print("Вероятности:         ", preds.flatten())

# --- ГРАФИК ---
plt.plot(loss_history)
plt.title("График ошибки (MSE) во время обучения")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.show()


'''

'''
class NeuralNetwork(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activaions = nn.ModuleList()
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            if i < len(layer_size) - 2:
                self.activaions.append(nn.Tanh())
            else:
                self.activaions.append(nn.Identity())

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, 
        weight_decay=0.0001) # здесь добавлен l2-регулизатор

    def forward(self, X):
        out = X
        for layer, activation in zip(self.layers, self.activaions):
            out = layer(out)
            out = activation(out)

        return out

    def train_step(self, X, y):
        pred = self.forward(X)
        loss = self.criterion(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss




torch.manual_seed(42)

model = NeuralNetwork([10,32,32,10,1])

n = 10000

X = torch.randn(n, 10) # [1,2,3...,10] то есть 10000 список и раждый список 10 знач
X[:, 0] *= 100 # вместо [1,2,3...10], [11,12,13...20] делает [100,2,3...10], [1100,12,13...20] тд
X[:, 1] *= 0.01 # все что ниже так же
X[:, 2] *= 10
X[:, 3] *= 1000

#                                       что бы исправить данные норм их
print(X[:, 0], 'X:0')
print(X[0, :], 'X0:')


y = (
  3 * X[:, 0] + 20 * X[:, 1] - 2 * X[:, 2] + 0.01 * X[:, 3] + 5 * X[:, 4] * X[:, 5]
  + 3 * torch.sin(X[:, 6]) - 4 * torch.cos(X[:, 7]) + X[:, 8] - 2 * X[:, 9]).reshape(-1, 1)
y += torch.randn_like(y) * 10

#                                       тоже самое для y

print(y[:, 0], 'y:0')
print(y[0, :], 'y0:')

n = len(X)
train_size = int(0.7 * n)
val_size   = int(0.2 * n)
test_size  = n - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val,   y_val   = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test,  y_test  = X[train_size + val_size:], y[train_size + val_size:]

X_mean = X_train.mean(dim=0)
X_std  = X_train.std(dim=0)
y_mean = y_train.mean(dim=0)
y_std  = y_train.std(dim=0)

X_train = (X_train - X_mean) / X_std
y_train = (y_train - y_mean) / y_std

X_val = (X_val - X_mean) / X_std
y_val = (y_val - y_mean) / y_std

X_test = (X_test - X_mean) / X_std
y_test = (y_test - y_mean) / y_std

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


train_data = DataLoader(train_dataset, batch_size=30, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=30, shuffle=False)
test_data = DataLoader(test_dataset, batch_size=30, shuffle=False)


epochs = 500


loss_history = []

for epoch in range(epochs):
    model.train()
    n_batches = 0
    total_loss = 0

    for X_batch, y_batch in train_data:
        loss = model.train_step(X_batch, y_batch)
        total_loss += loss.item()
        n_batches += 1

    if epoch == 1:
        print(loss.item() ,loss)

    avg_loss = total_loss / n_batches
    total_loss = 0
    n_batches = 0
    loss_history.append(avg_loss)
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, средняя ошибка: {avg_loss:.6f}")

    if epoch % 25 == 0:
    ################################# валидация #######################

        total_loss_val = 0
        criterion = nn.MSELoss()
        n = 0
        with torch.no_grad():
            model.eval()
            for x, y in val_data:
                pred = model(x)
                loss = criterion(pred, y)
                total_loss_val += loss.item()
                n += 1
            print('############################ валидация ############################')
            print()
            print()
            print(total_loss_val / n)

        total_loss_val = 0
        n = 0







total_loss_test = 0
criterion = nn.MSELoss()
n = 0
with torch.no_grad():
    model.eval()
    for x, y in test_data:
        pred = model(x)
        loss = criterion(pred, y)
        total_loss_test += loss.item()
        n += 1
print('############################ тест ############################')
print()
print()
print(total_loss_test / n)


# крч как я пончял надо сделать ADAM тк он стабильнее SGD,
# также надо данные привести к 0 или 1 тк признаки разныве один 10000000 а другой 0.0001
# мб сделать lr меньше
# меньший lr помог но прям чутка


'''



import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SunDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = len(self.format)
        self.files = tuple(self.format.keys())
        self.targets = tuple(self.format.values())

    def __getitem__(self, item):
        path_file = os.path.join(self.path, self.files[item])
        img = Image.open(path_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.targets[item], dtype=torch.float32)

    def __len__(self):
        return self.length


model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 8, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 4, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(4096, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
d_train = SunDataset("dataset_reg", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_function = nn.MSELoss()
epochs = 5
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1/lm_count * loss.item() + (1 - 1/lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

st = model.state_dict()
torch.save(st, 'model_sun_2.tar')

d_test = SunDataset("dataset_reg", train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

# тестирование обученной НС
Q = 0
count = 0
model.eval()

test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        Q += loss_function(p, y_test).item()
        count += 1

Q /= count
print(Q)
