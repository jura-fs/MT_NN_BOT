'''=====================================================================================================================
========================================================================================================================
====================================================================================================================='''
import numpy as np
import matplotlib.pyplot as plt
import torch
from sympy.codegen.ast import float16
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import embedding, dropout
import torch.nn.functional as F




'''                                               ЭТАП 1                                              '''




'''                                         ЗАДАЧА 1                                         '''


'''import numpy as np
import matplotlib.pyplot as plt



x = np.random.rand()
lr = 0.1
iteration = 50

history_x = []
history_function = []

for i in range(iteration):
    c_v = (x - 5) ** 2
    gradient = 2 * (x - 5) # это у нас производнеая от c_v
    x = x - gradient * lr # это обновление x

    history_x.append(x)
    history_function.append(c_v)



    if i % 10 == 0:
        print(f"Шаг {i}: x = {x:.4f}, значение функции = {c_v:.4f}")




plt.figure(figsize=(12, 4))  # Размер картинки

# График 1: Как менялся x
plt.subplot(1, 2, 1)  # 1 строка, 2 колонки, 1-й график
plt.plot(history_x, color='blue', marker='o', linestyle='--')
plt.axhline(y=5, color='red', linestyle='-', label='Правильный ответ (5)')
plt.title('Как X приближается к 5')
plt.xlabel('Шаг (эпоха)')
plt.ylabel('Значение X')
plt.grid(True)
plt.legend()

# График 2: Как падала ошибка
plt.subplot(1, 2, 2)  # 1 строка, 2 колонки, 2-й график
plt.plot(history_function, color='green', marker='s')
plt.title('Как падает значение функции (ошибка)')
plt.xlabel('Шаг (эпоха)')
plt.ylabel('Значение функции')
plt.grid(True)

plt.tight_layout()
plt.show()'''

'''==========================================================================================='''

'''смое просто обьяснение мы стоим на горе и начинаем спускаться и мы всегда спустимся в низ 
тк на горе не выбоин и ямок но след задача это реальная гора где есть и 
ямы и впадины из за этого будет другая формула и другой шаг что бы не перелететь настоящий низ горы '''





'''                                    ЗАДАЧА 2                                       '''

'''

import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand()
lr = 0.01
iteration = 100

history_x = []
history_function = []

for i in range(iteration):

    c_v = x**4 - 3 * x**2 + 2
    gradient = 4 * (x ** 3) - 6 * x
    x = x - gradient * lr

    history_x.append(x)
    history_function.append(c_v)

    if i % 10 == 0:
        print(f"Шаг {i}: x = {x:.4f}, значение функции = {c_v:.4f}")

# Рисуем графики для второй функции
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_x, color='purple', marker='o')
plt.title('Как меняется X для сложной функции')
plt.xlabel('Шаг')
plt.ylabel('Значение X')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_function, color='orange', marker='s')
plt.title('Как падает сложная функция')
plt.xlabel('Шаг')
plt.ylabel('Значение функции')
plt.grid(True)

plt.tight_layout()
plt.show()

'''

'''==================================================================================================='''



'''                                        ЗАДАЧА 3                                                   '''

"""import numpy as np
import matplotlib.pyplot as plt


# Функция с двумя ямами разной глубины
def f(x):
    return (x + 1) ** 2 * (x - 2) ** 2 + 0.1 * np.sin(3 * x)


# Градиент функции
def gradient(x):
    return 4 * (x + 1) * (x - 2) ** 2 + 4 * (x + 1) ** 2 * (x - 2) + 0.3 * np.cos(3 * x)


print("=" * 70)
print("ЗАДАНИЕ 3: Спасаем застрявший шарик (симуляция импульса)")
print("=" * 70)

# Параметры
lr = 0.01
iterations = 200
start_x = 3.0

# ========================================
# 1. ОБЫЧНЫЙ ГРАДИЕНТНЫЙ СПУСК (БЕЗ ИМПУЛЬСА)
# ========================================
print("\n🔵 ОБЫЧНЫЙ ГРАДИЕНТНЫЙ СПУСК (без импульса)")
print("-" * 50)

x_normal = start_x
history_normal_x = [x_normal]
history_normal_f = [f(x_normal)]

for i in range(iterations):
    grad = gradient(x_normal)
    x_normal = x_normal - lr * grad

    history_normal_x.append(x_normal)
    history_normal_f.append(f(x_normal))

    if i % 20 == 0:
        print(f"Шаг {i:3d}: x = {x_normal:.6f}, f(x) = {f(x_normal):.6f}")

print(f"\n🏁 ФИНИШ: x = {x_normal:.6f}, f(x) = {f(x_normal):.6f}")

# Определяем, в какую яму упал
if x_normal > 0.5:
    print("🕳️ Упал в ПРАВУЮ (мелкую) яму! ❌")
    print("   (застрял в локальном минимуме)")
else:
    print("🕳️ Упал в ЛЕВУЮ (глубокую) яму! ✅")

# ========================================
# 2. ГРАДИЕНТНЫЙ СПУСК С ИМПУЛЬСОМ (MOMENTUM)
# ========================================
print("\n" + "=" * 70)
print("🟣 ГРАДИЕНТНЫЙ СПУСК С ИМПУЛЬСОМ (Momentum)")
print("-" * 50)

x_momentum = start_x
velocity = 0
momentum = 0.9
history_momentum_x = [x_momentum]
history_momentum_f = [f(x_momentum)]

for i in range(iterations):
    grad = gradient(x_momentum)
    velocity = momentum * velocity - lr * grad
    x_momentum = x_momentum + velocity

    history_momentum_x.append(x_momentum)
    history_momentum_f.append(f(x_momentum))

    if i % 20 == 0:
        print(f"Шаг {i:3d}: x = {x_momentum:.6f}, f(x) = {f(x_momentum):.6f}")

print(f"\n🏁 ФИНИШ: x = {x_momentum:.6f}, f(x) = {f(x_momentum):.6f}")

# Определяем, в какую яму упал
if x_momentum > 0.5:
    print("🕳️ Упал в ПРАВУЮ (мелкую) яму! ❌")
else:
    print("🕳️ Упал в ЛЕВУЮ (глубокую) яму! ✅")
    print("   🎉 Импульс помог перелететь через холм!")

# ========================================
# ВИЗУАЛИЗАЦИЯ
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Траектории X (сравнение)
ax1 = axes[0, 0]
ax1.plot(history_normal_x, label='Без импульса', color='blue', linewidth=2)
ax1.plot(history_momentum_x, label='С импульсом', color='purple', linewidth=2)
ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Правая яма (x≈2)')
ax1.axhline(y=-1, color='green', linestyle='--', alpha=0.5, label='Левая яма (x≈-1)')
ax1.set_title('Траектории X: обычный vs импульс', fontsize=12)
ax1.set_xlabel('Шаг')
ax1.set_ylabel('Значение X')
ax1.grid(True, alpha=0.3)
ax1.legend()

# График 2: Значения функции (сравнение)
ax2 = axes[0, 1]
ax2.plot(history_normal_f, label='Без импульса', color='blue', linewidth=2)
ax2.plot(history_momentum_f, label='С импульсом', color='purple', linewidth=2)
ax2.set_title('Значение функции: обычный vs импульс', fontsize=12)
ax2.set_xlabel('Шаг')
ax2.set_ylabel('f(x)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# График 3: Сама функция + траектории на фоне
ax3 = axes[1, 0]
x_plot = np.linspace(-3, 4, 1000)
y_plot = f(x_plot)
ax3.plot(x_plot, y_plot, color='black', linewidth=2, label='f(x)')
ax3.scatter(history_normal_x, history_normal_f, color='blue', s=10, alpha=0.5, label='Без импульса')
ax3.scatter(history_momentum_x, history_momentum_f, color='purple', s=10, alpha=0.5, label='С импульсом')
ax3.scatter([history_normal_x[-1]], [history_normal_f[-1]], color='red', s=100, marker='*', label='Финиш (без)')
ax3.scatter([history_momentum_x[-1]], [history_momentum_f[-1]], color='gold', s=100, marker='*', label='Финиш (с)')
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax3.set_title('Траектории на фоне функции', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('f(x)')
ax3.grid(True, alpha=0.3)
ax3.legend()

# График 4: Анимация импульса (скорость)
ax4 = axes[1, 1]
# Пересчитываем скорость для визуализации
velocities = []
v = 0
for i in range(iterations):
    grad = gradient(history_momentum_x[i])
    v = momentum * v - lr * grad
    velocities.append(v)
ax4.plot(velocities, color='orange', linewidth=2)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax4.set_title('Скорость (velocity) при импульсе', fontsize=12)
ax4.set_xlabel('Шаг')
ax4.set_ylabel('velocity')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================
# ИТОГОВЫЙ ВЫВОД
# ========================================
print("\n" + "=" * 70)
print("ВЫВОДЫ:")
print("=" * 70)

if abs(x_normal - 2) < abs(x_normal + 1):
    print("✅ Обычный градиентный спуск застрял в ПРАВОЙ (мелкой) яме")
    print("   Потому что стартовали справа, и градиент вел вниз по склону")
else:
    print("✅ Обычный градиентный спуск попал в ЛЕВУЮ (глубокую) яму")

if abs(x_momentum + 1) < abs(x_momentum - 2):
    print("✅ Импульсный спуск смог перелететь через холм и попал в ЛЕВУЮ (глубокую) яму")
    print("   Импульс накопил скорость и пролетел мимо мелкой ямы!")
else:
    print("❌ Импульс не помог — все равно упал в правую яму")
    print("   Возможно, нужен больший momentum или lr")

print("\n💡 Ключевой урок: Импульс (Momentum) помогает нейросетям")
print("   преодолевать локальные минимумы и находить более глубокие!")
print("=" * 70)

"""

'''==================================================================================================='''



'''                                        ЗАДАЧА 4                                                   '''


'''import numpy as np
import matplotlib.pyplot as plt

x, y = 3, 4
lr = 0.1
history_x, history_y = [x], [y]

for i in range(50):
    grad_x = 2 * x
    grad_y = 2 * y
    x = x - grad_x * lr
    y = y - grad_y * lr
    
    history_x.append(x)
    history_y.append(y)
    
plt.plot(history_x, history_y, 'bo-', markersize=4)
plt.scatter([0], [0], color='red', s=100, label='Минимум (0,0)')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.legend()
plt.show()'''

'''==================================================================================================='''
'''                                         обучение одного нейрона                                   '''


'''                                        ЗАДАЧА 5                                                   '''

'''import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

w = 0.0
lr = 0.01
history_w = [w]
history_loss = []

for i in range(100):
    pred = w * X
    loss = np.mean((pred - Y) ** 2)
    history_loss.append(loss)
    grad = np.mean(2 * X * (pred - Y))
    w = w - lr * grad
    history_w.append(w)

print(f"Финальный вес: {w:.4f}")  # Должно быть ≈ 2.0

# Графики
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_w, color='blue')
plt.axhline(y=2, color='red', linestyle='--', label='Правильный вес=2')
plt.title('Как вес приближается к 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_loss, color='green')
plt.title('Как падает ошибка')
plt.show()'''

'''==================================================================================================='''


'''from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-5, 5, 30)
y = np.linspace(-5, 5, 30)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter([3], [4], [25], color='red', s=100, label='Старт')
ax.scatter([0], [0], [0], color='green', s=100, label='Финиш')
ax.legend()
plt.show()'''


'''=====================================================================================================================
========================================================================================================================
====================================================================================================================='''

'''                                               ЭТАП 2                                              '''

'''
"=== ЭТАП 2: ЛИНЕЙНАЯ РЕГРЕССИЯ С ДВУМЯ ПАРАМЕТРАМИ ==="

import numpy as np
import matplotlib.pyplot as plt

true_w = 3.0
true_b = 2.0
lr = 0.01
iterations = 200

np.random.seed(42)

X = np.linspace(0, 10, 50)
noise = np.random.randn(50) * 2

w = np.random.randn() * 0.5
b = 0.0

y = true_w * X + true_b + noise

history_w = [w]
history_b = [b]
history_loss = []

for epoch in range(iterations):
    y_pred = w * X + b
    loss = np.mean((y_pred - y) ** 2)

    grad_pred = 2 * (y_pred - y) / len(X)
    grad_w = np.mean(grad_pred * X)
    grad_b = np.mean(grad_pred)

    w = w - lr * grad_w
    b = b - lr * grad_b

    history_w.append(w)
    history_b.append(b)
    history_loss.append(loss)

    if epoch % 20 == 0:
        print(f"Эпоха {epoch:3d}: w = {w:.4f}, b = {b:.4f}, loss = {loss:.4f}")

print("-" * 50)
print(f"Финальный вес w = {w:.4f} (правильный = {true_w:.4f})")
print(f"Финальное смещение b = {b:.4f} (правильное = {true_b:.4f})")
print(f"Финальная ошибка loss = {loss:.4f}")

# ВИЗУАЛИЗАЦИЯ
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X, y, color='blue', alpha=0.6, label='Данные (с шумом)')
plt.plot(X, true_w * X + true_b, color='green', linewidth=3, label='Правильная прямая')
plt.plot(X, w * X + b, color='red', linewidth=2, linestyle='--', label='Обученная прямая')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Данные и прямые')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history_w, color='blue', label='w (вес)')
plt.plot(history_b, color='orange', label='b (смещение)')
plt.axhline(y=true_w, color='blue', linestyle='--', alpha=0.5, label='Правильный w=3')
plt.axhline(y=true_b, color='orange', linestyle='--', alpha=0.5, label='Правильное b=2')
plt.title('Изменение параметров')
plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history_loss, color='green')
plt.title('Падение ошибки (Loss)')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True)

plt.tight_layout()
plt.show()'''


'''                      КЛАСС GradientDescent БЕЗ momentum ПРОСТО ГРАД СПУСК                         '''



'''import numpy as np 
import matplotlib.pyplot as plt
class GradientDescent:

    def __init__(self, lr=0.1, iterations=50, x=None):
        if x is None:
            x = np.random.rand()
        self.x = x
        self.iterations = iterations
        self.lr = lr
        self.c_v = (self.x - 5) ** 2
        self._history_x = [self.x]
        self._history_function = [self.c_v]



    def step(self, x):

        gradient = 2 * (x - 5)
        self.x = x - gradient * self.lr
        self.c_v = (self.x - 5) ** 2
        self._history_x.append(self.x)
        self._history_function.append(self.c_v)

        return self.x

    def return_C_V(self):
        return self.c_v

    def return_x(self):
        return self.x

    def return_history(self):
        return self._history_x, self._history_function

    def run(self):
        for i in range(self.iterations):
            self.step(self.x)

        return self.x, self.c_v

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # График 1: Как менялся x
        ax1.plot(self._history_x, color='blue', marker='o', linestyle='--')
        ax1.axhline(y=5, color='red', linestyle='-', label='Правильный ответ (5)')
        ax1.set_title('Приближение X к 5')
        ax1.set_xlabel('Шаг (эпоха)')
        ax1.set_ylabel('Значение X')
        ax1.grid(True)
        ax1.legend()

        # График 2: Как падала ошибка
        ax2.plot(self._history_function, color='green', marker='s')
        ax2.set_title('Уменьшение функции потерь')
        ax2.set_xlabel('Шаг (эпоха)')
        ax2.set_ylabel('Значение функции')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()



# ========================================
# ТЕСТЫ (без сторонних библиотек)
# ========================================

def test_initialization():
    """Тест 1: Проверка инициализации"""
    print("🧪 Тест 1: Инициализация...")

    gd = GradientDescent(lr=0.1, iterations=50, x=0.0)
    assert gd.x == 0.0, f"Ошибка: x = {gd.x}, ожидалось 0.0"
    assert gd.lr == 0.1, f"Ошибка: lr = {gd.lr}, ожидалось 0.1"
    assert gd.iterations == 50, f"Ошибка: iterations = {gd.iterations}, ожидалось 50"
    assert gd.c_v == 25.0, f"Ошибка: c_v = {gd.c_v}, ожидалось 25.0"
    assert len(gd._history_x) == 1, f"Ошибка: len(history_x) = {len(gd._history_x)}, ожидалось 1"
    assert len(gd._history_function) == 1, f"Ошибка: len(history_function) = {len(gd._history_function)}, ожидалось 1"
    assert gd._history_x[0] == 0.0, f"Ошибка: history_x[0] = {gd._history_x[0]}, ожидалось 0.0"
    assert gd._history_function[0] == 25.0, f"Ошибка: history_function[0] = {gd._history_function[0]}, ожидалось 25.0"

    print("✅ Тест 1 пройден!\n")


def test_step():
    """Тест 2: Проверка одного шага"""
    print("🧪 Тест 2: Один шаг градиентного спуска...")

    gd = GradientDescent(lr=0.1, iterations=50, x=0.0)

    # Расчет вручную:
    # x = 0, gradient = 2*(0-5) = -10
    # new_x = 0 - 0.1*(-10) = 1.0
    # c_v = (1-5)² = 16

    new_x = gd.step(gd.x)
    assert new_x == 1.0, f"Ошибка: step вернул {new_x}, ожидалось 1.0"
    assert gd.x == 1.0, f"Ошибка: x = {gd.x}, ожидалось 1.0"
    assert gd.c_v == 16.0, f"Ошибка: c_v = {gd.c_v}, ожидалось 16.0"
    assert len(gd._history_x) == 2, f"Ошибка: len(history_x) = {len(gd._history_x)}, ожидалось 2"
    assert len(gd._history_function) == 2, f"Ошибка: len(history_function) = {len(gd._history_function)}, ожидалось 2"
    assert gd._history_x[1] == 1.0, f"Ошибка: history_x[1] = {gd._history_x[1]}, ожидалось 1.0"
    assert gd._history_function[1] == 16.0, f"Ошибка: history_function[1] = {gd._history_function[1]}, ожидалось 16.0"

    print("✅ Тест 2 пройден!\n")


def test_run():
    """Тест 3: Проверка полного цикла"""
    print("🧪 Тест 3: Полный цикл обучения...")

    gd = GradientDescent(lr=0.1, iterations=10, x=0.0)
    final_x, final_loss = gd.run()

    # После 10 шагов с lr=0.1, x должен приблизиться к 5
    # Проверяем, что x стал ближе к 5, чем был
    assert abs(final_x - 5.0) < 1.0, f"Ошибка: final_x = {final_x}, слишком далеко от 5"
    assert final_loss < 1.0, f"Ошибка: final_loss = {final_loss}, слишком большая ошибка"
    assert len(
        gd._history_x) == 11, f"Ошибка: len(history_x) = {len(gd._history_x)}, ожидалось 11 (10 шагов + начальное)"
    assert len(gd._history_function) == 11, f"Ошибка: len(history_function) = {len(gd._history_function)}, ожидалось 11"

    print(f"   Финальный x = {final_x:.6f}, ошибка = {final_loss:.6f}")
    print("✅ Тест 3 пройден!\n")


def test_history():
    """Тест 4: Проверка сохранения истории"""
    print("🧪 Тест 4: Проверка истории...")

    gd = GradientDescent(lr=0.1, iterations=5, x=0.0)
    gd.run()

    history_x, history_f = gd.return_history()

    assert len(history_x) == 6, f"Ошибка: len(history_x) = {len(history_x)}, ожидалось 6"
    assert len(history_f) == 6, f"Ошибка: len(history_f) = {len(history_f)}, ожидалось 6"
    assert history_x[0] == 0.0, f"Ошибка: history_x[0] = {history_x[0]}, ожидалось 0.0"
    assert history_f[0] == 25.0, f"Ошибка: history_f[0] = {history_f[0]}, ожидалось 25.0"

    # Проверяем, что история монотонно приближается к 5
    for i in range(1, len(history_x)):
        assert abs(history_x[i] - 5.0) < abs(history_x[i - 1] - 5.0), \
            f"Ошибка: x не приближается к 5 на шаге {i}: {history_x[i - 1]} -> {history_x[i]}"

    print("✅ Тест 4 пройден!\n")


def test_getters():
    """Тест 5: Проверка геттеров"""
    print("🧪 Тест 5: Проверка методов return...")

    gd = GradientDescent(lr=0.1, iterations=50, x=0.0)

    assert gd.return_x() == 0.0, f"Ошибка: return_x() = {gd.return_x()}, ожидалось 0.0"
    assert gd.return_C_V() == 25.0, f"Ошибка: return_C_V() = {gd.return_C_V()}, ожидалось 25.0"

    gd.step(gd.x)
    assert gd.return_x() == 1.0, f"Ошибка: return_x() = {gd.return_x()}, ожидалось 1.0"
    assert gd.return_C_V() == 16.0, f"Ошибка: return_C_V() = {gd.return_C_V()}, ожидалось 16.0"

    history_x, history_f = gd.return_history()
    assert len(history_x) == 2, f"Ошибка: len(history_x) = {len(history_x)}, ожидалось 2"
    assert len(history_f) == 2, f"Ошибка: len(history_f) = {len(history_f)}, ожидалось 2"

    print("✅ Тест 5 пройден!\n")


def test_different_start_points():
    """Тест 6: Проверка с разными стартовыми точками"""
    print("🧪 Тест 6: Разные стартовые точки...")

    start_points = [-5.0, -1.0, 2.0, 4.5, 5.0, 6.0, 10.0]

    for start in start_points:
        gd = GradientDescent(lr=0.1, iterations=100, x=start)
        final_x, final_loss = gd.run()

        # Проверяем, что x приблизился к 5
        assert abs(final_x - 5.0) < 0.01, \
            f"Ошибка: из точки {start} пришел в x = {final_x}, ожидалось ~5.0"
        assert final_loss < 0.001, \
            f"Ошибка: из точки {start} ошибка = {final_loss}, ожидалось ~0"

    print("✅ Тест 6 пройден!\n")


def test_convergence_speed():
    """Тест 7: Проверка скорости сходимости"""
    print("🧪 Тест 7: Скорость сходимости...")

    # Большой lr -> быстрая сходимость
    gd_fast = GradientDescent(lr=0.5, iterations=20, x=0.0)
    gd_fast.run()
    assert abs(gd_fast.x - 5.0) < 0.001, \
        f"Ошибка: быстрый lr не сошелся: x = {gd_fast.x}"

    # Маленький lr -> медленная сходимость
    gd_slow = GradientDescent(lr=0.01, iterations=50, x=0.0)
    gd_slow.run()
    # После 50 шагов с lr=0.01 должно быть ближе к 5
    assert abs(gd_slow.x - 5.0) < 0.5, \
        f"Ошибка: медленный lr не дошел: x = {gd_slow.x}"

    print("✅ Тест 7 пройден!\n")


def test_monotonic_decrease():
    """Тест 8: Проверка монотонного уменьшения ошибки"""
    print("🧪 Тест 8: Монотонное уменьшение ошибки...")

    gd = GradientDescent(lr=0.1, iterations=50, x=0.0)
    gd.run()

    history_x, history_f = gd.return_history()

    # Ошибка должна монотонно уменьшаться
    for i in range(1, len(history_f)):
        assert history_f[i] <= history_f[i - 1], \
            f"Ошибка: ошибка увеличилась на шаге {i}: {history_f[i - 1]} -> {history_f[i]}"

    # x должен монотонно приближаться к 5
    for i in range(1, len(history_x)):
        dist_prev = abs(history_x[i - 1] - 5.0)
        dist_curr = abs(history_x[i] - 5.0)
        assert dist_curr <= dist_prev, \
            f"Ошибка: расстояние до 5 увеличилось на шаге {i}: {dist_prev} -> {dist_curr}"

    print("✅ Тест 8 пройден!\n")


def test_no_negative_values():
    """Тест 9: Проверка отсутствия отрицательных значений"""
    print("🧪 Тест 9: Проверка на отрицательные значения...")

    gd = GradientDescent(lr=0.1, iterations=50, x=0.0)
    gd.run()

    history_x, history_f = gd.return_history()

    # c_v всегда >= 0, так как это квадрат
    assert all(f >= 0 for f in history_f), "Ошибка: найдено отрицательное значение в history_f"

    # x может быть отрицательным (если стартуем с отрицательного),
    # но для старта с 0 он должен быть >= 0
    assert all(x >= 0 for x in history_x), "Ошибка: найдено отрицательное значение в history_x"

    print("✅ Тест 9 пройден!\n")


def test_random_start():
    """Тест 10: Проверка работы со случайным стартом"""
    print("🧪 Тест 10: Случайный старт...")

    # Создаем несколько объектов со случайным стартом
    for _ in range(5):
        gd = GradientDescent(lr=0.1, iterations=100)
        final_x, final_loss = gd.run()

        # Проверяем, что сошлись к 5
        assert abs(final_x - 5.0) < 0.01, \
            f"Ошибка: случайный старт пришел в x = {final_x}, ожидалось ~5.0"
        assert final_loss < 0.001, \
            f"Ошибка: случайный старт ошибка = {final_loss}, ожидалось ~0"

    print("✅ Тест 10 пройден!\n")


def test_plot():
    """Тест 11: Проверка метода plot (просто проверяем, что не падает)"""
    print("🧪 Тест 11: Проверка метода plot...")

    try:
        gd = GradientDescent(lr=0.1, iterations=10, x=0.0)
        gd.run()
        gd.plot()
        plot_success = True
        print("   График отобразился (закройте его для продолжения)")
    except Exception as e:
        plot_success = False
        print(f"   Ошибка при вызове plot: {e}")

    assert plot_success, "Ошибка: метод plot упал с исключением"
    print("✅ Тест 11 пройден!\n")


# ========================================
# ЗАПУСК ВСЕХ ТЕСТОВ
# ========================================

def run_all_tests():
    """Запускает все тесты"""
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ ДЛЯ КЛАССА GradientDescent")
    print("=" * 60)
    print()

    tests = [
        test_initialization,
        test_step,
        test_run,
        test_history,
        test_getters,
        test_different_start_points,
        test_convergence_speed,
        test_monotonic_decrease,
        test_no_negative_values,
        test_random_start,
        test_plot
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {e}\n")
            failed += 1
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"ИТОГО: {passed} тестов пройдено, {failed} тестов провалено")
    print("=" * 60)

    if failed == 0:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Класс работает корректно!")
    else:
        print(f"⚠️ {failed} тестов провалено. Проверьте код!")


# Запускаем тесты
if __name__ == "__main__":
    run_all_tests()'''


'''                             КЛАСС GradientDescent С momentum                                      '''

'''import numpy as np
import matplotlib.pyplot as plt




class GradientDescentWithMomentum:
    def __init__(self, lr=0.1, iterations=50, momentum=0.9, x_momentum=None):
        if x_momentum is None:
            x_momentum = np.random.rand()

        self.x_momentum = x_momentum
        self.lr = lr
        self.c_v = (x_momentum - 5) ** 2
        self.iterations = iterations
        self.history_x = [self.x_momentum]
        self.history_function = [self.c_v]
        self.velocity = 0
        self._history_velocity = [self.velocity]
        self.momentum = momentum

    def step(self, x_momentum):
        gradient = 2 * (x_momentum - 5)
        self.velocity = self.momentum * self.velocity - self.lr * gradient

        self.x_momentum = self.x_momentum + self.velocity
        self.c_v = (self.x_momentum - 5) ** 2

        self.history_x.append(self.x_momentum)
        self.history_function.append(self.c_v)
        self._history_velocity.append(self.velocity)

    def return_C_V(self):
        return self.c_v

    def return_x(self):
        return self.x_momentum

    def return_velocity(self):
        return self.velocity

    def return_history(self):
        return self.history_x, self.history_function, self._history_velocity

    def run(self):
        for i in range(self.iterations):
            self.step(self.x_momentum)
        return self.x_momentum, self.c_v

    def plot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        ax1.plot(self.history_x, color='blue', marker='o', linestyle='--')
        ax1.axhline(y=5, color='red', linestyle='-', label='Цель (5)')
        ax1.set_title('Приближение X к 5')
        ax1.set_xlabel('Шаг')
        ax1.set_ylabel('Значение X')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(self.history_function, color='green', marker='s')
        ax2.set_title('Уменьшение функции потерь')
        ax2.set_xlabel('Шаг')
        ax2.set_ylabel('Значение функции')
        ax2.grid(True)

        ax3.plot(self._history_velocity, color='purple', marker='^')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_title('Изменение скорости')
        ax3.set_xlabel('Шаг')
        ax3.set_ylabel('Скорость')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Создаем объект
    gd = GradientDescentWithMomentum(lr=0.1, iterations=30, momentum=0.9, x_momentum=0.0)

    # Запускаем обучение
    print("=" * 50)
    print("ОБУЧЕНИЕ ГРАДИЕНТНОГО СПУСКА С МОМЕНТОМ")
    print("=" * 50)
    print(f"Начальный x = {gd.x_momentum}")
    print(f"Целевое значение = 5")
    print(f"Learning rate = {gd.lr}")
    print(f"Momentum = {gd.momentum}")
    print(f"Итераций = {gd.iterations}")
    print("-" * 50)

    final_x, final_loss = gd.run()

    print(f"Финальный x = {final_x:.6f}")
    print(f"Финальная ошибка = {final_loss:.6f}")
    print(f"Погрешность = {abs(final_x - 5):.6f}")
    print("=" * 50)
    print()

    # Показываем графики
    print("Открываю окно с графиками...")
    print("(Закройте окно с графиком, чтобы завершить программу)")
    gd.plot()

    print("Программа завершена!")'''




'''                                             ТРЕНИРОВКА ПЕРЕД Adam                                   '''


'''import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(0, 10, 100)

true_w = 3.0
true_b = 0

noise = np.random.randn(100) * 2
y = true_w * X + true_b + noise
'''
'''                                 rmsprop                                                 '''


'''def rmsprop(X, y, lr=0.01, beta2=0.999, epsilon=1e-8, iterations=200):
    w = np.random.randn() * 0.5
    b = 0.0

    v_w = 0.0
    v_b = 0.0

    history_w = [w]
    history_b = [b]
    history_loss = []

    for epoch in range(iterations):
        y_pred = w * X + b
        loss = np.mean((y_pred - y) ** 2)

        grad_pred = 2 * (y_pred - y) / len(X)
        grad_w = np.mean(grad_pred * X)
        grad_b = np.mean(grad_pred)

        v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
        v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

        w = w - lr * grad_w / (np.sqrt(v_w) + epsilon)
        b = b - lr * grad_b / (np.sqrt(v_b) + epsilon)

        history_w.append(w)
        history_b.append(b)
        history_loss.append(loss)

        if epoch % 20 == 0:
            print(f"Эпоха {epoch:3d}: w = {w:.4f}, b = {b:.4f}, loss = {loss:.4f}")

    return w, b, history_w, history_b, history_loss

# Запускаем
w_rms, b_rms, hw_rms, hb_rms, hl_rms = rmsprop(X, y)

print("-" * 50)
print(f"Финальный вес w = {w_rms:.4f} (правильный = {true_w:.4f})")
print(f"Финальное смещение b = {b_rms:.4f} (правильный = {true_b:.4f})")
print(f"Финальная ошибка loss = {hl_rms[-1]:.4f}")'''




'''                     momentum_with_bias_correction                           '''


'''def momentum_with_bias_correction(X, y, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=200):
    w = np.random.randn() * 0.5
    b = 0.0

    m_w = 0.0
    m_b = 0.0
    v_w = 0.0
    v_b = 0.0

    history_w = [w]
    history_b = [b]
    history_loss = []

    for i in range(1 , iterations+1):
        y_pred = w * X + b
        loss = np.mean((y_pred - y) ** 2)

        grad_pred =  2 * (y_pred - y) / len(X)
        grad_w = np.mean(grad_pred * X)
        grad_b = np.mean(grad_pred)

        m_w = m_w * beta1 + (1 - beta1) * grad_w
        m_b = m_b * beta1 + (1 - beta1) * grad_b
        v_w = v_w * beta2 + (1 - beta2) * (grad_w ** 2)
        v_b = v_b * beta2 + (1 - beta2) * (grad_b ** 2)

        m_w_corrected = m_w / (1 - beta1 ** i)
        m_b_corrected = m_b / (1 - beta1 ** i)
        v_w_corrected = v_w / (1 - beta2 ** i)
        v_b_corrected = v_b / (1 - beta2 ** i)

        w = w - lr * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
        b = b - lr * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

        history_w.append(w)
        history_b.append(b)
        history_loss.append(loss)

        if i % 20 == 0:
            print(f"Эпоха {i:3d}: w = {w:.4f}, b = {b:.4f}, loss = {loss:.4f}")

    return w, b, history_w, history_b, history_loss

        # Запускаем
w_bc, b_bc, hw_bc, hb_bc, hl_bc = momentum_with_bias_correction(X, y)

print("-" * 50)
print(f"Финальный вес w = {w_bc:.4f} (правильный = {true_w:.4f})")
print(f"Финальное смещение b = {b_bc:.4f} (правильный = {true_b:.4f})")
print(f"Финальная ошибка loss = {hl_bc[-1]:.4f}")'''


'''                                                 ADAM                                             '''
'''def adam(X, y, lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=200):
    w = np.random.randn() * 0.5
    b = 0.0

    m_w = 0.0
    m_b = 0.0
    v_w = 0.0
    v_b = 0.0

    history_w = [w]
    history_b = [b]
    history_loss = []
    history_m_w = [m_w]
    history_v_w = [v_w]

    for epoch in range(1, iterations + 1):
        y_pred = X * w + b
        loss = np.mean((y_pred - y) ** 2)

        grad_pred = 2 * (y_pred - y) / len(X)
        grad_w = np.mean(grad_pred * X)
        grad_b = np.mean(grad_pred)

        m_w = m_w * beta1 + (1 - beta1) * grad_w
        m_b = m_b * beta1 + (1 - beta1) * grad_b

        v_w = v_w * beta2 + (1 - beta2) * (grad_w ** 2)
        v_b = v_b * beta2 + (1 - beta2) * (grad_b ** 2)

        m_w_corrected = m_w / (1 - beta1 ** epoch)
        m_b_corrected = m_b / (1 - beta1 ** epoch)
        v_w_corrected = v_w / (1 - beta2 ** epoch)
        v_b_corrected = v_b / (1 - beta2 ** epoch)

        w = w - lr * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
        b = b - lr * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

        history_w.append(w)
        history_b.append(b)
        history_loss.append(loss)
        history_m_w.append(m_w)
        history_v_w.append(v_w)

        if epoch % 20 == 0:
            print(f"Эпоха {epoch:3d}: w = {w:.4f}, b = {b:.4f}, loss = {loss:.4f}")
            print(f"         m_w = {m_w:.4f}, v_w = {v_w:.4f}")

    print("-" * 60)
    print(f"Финальный вес w = {w:.4f} (правильный = {true_w:.4f})")
    print(f"Финальное смещение b = {b:.4f} (правильный = {true_b:.4f})")
    print(f"Финальная ошибка loss = {loss:.4f}")

    return w, b, history_w, history_b, history_loss, history_m_w, history_v_w


# Запускаем ADAM
w_adam, b_adam, hw_adam, hb_adam, hl_adam, hm_adam, hv_adam = adam(X, y)

# Визуализация
plt.figure(figsize=(15, 10))

# График 1: Loss
plt.subplot(2, 2, 1)
plt.plot(hl_adam, label='Adam', color='red', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Падение ошибки (Adam)')
plt.legend()
plt.grid(True)

# График 2: w и b
plt.subplot(2, 2, 2)
plt.plot(hw_adam, label='w', color='blue')
plt.plot(hb_adam, label='b', color='orange')
plt.axhline(y=true_w, color='blue', linestyle='--', alpha=0.5, label='Правильный w=3')
plt.axhline(y=true_b, color='orange', linestyle='--', alpha=0.5, label='Правильное b=0')
plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.title('Изменение параметров')
plt.legend()
plt.grid(True)

# График 3: Моменты
plt.subplot(2, 2, 3)
plt.plot(hm_adam, label='m_w (первый момент)', color='green')
plt.plot(hv_adam, label='v_w (второй момент)', color='purple')
plt.xlabel('Эпоха')
plt.ylabel('Значение')
plt.title('Моменты в Adam')
plt.legend()
plt.grid(True)

# График 4: Данные и прямая
plt.subplot(2, 2, 4)
plt.scatter(X, y, alpha=0.6, label='Данные с шумом')
plt.plot(X, true_w * X + true_b, 'g-', linewidth=2, label='Правильная прямая')
plt.plot(X, w_adam * X + b_adam, 'r--', linewidth=2, label='Обученная прямая')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Результат обучения')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()'''





'''


X = np.array([1,2,3,4,5])
Y = np.array([2,4,6,8,10])


iterations = 2000
learning_rate = 0.01


np.random.seed(42)


weight_1 = np.random.randn() * 0.5
bias_1 = np.random.randn() * 0.5
weight_2 = np.random.randn() * 0.5
bias_2 = np.random.randn() * 0.5


for i in range(iterations):
    z1 = weight_1 * X + bias_1

    a1 = np.maximum(0, z1)

    pred = weight_2 * a1 + bias_2

    loss = np.mean((pred - Y) ** 2)

    dL_Lpred = 2 * (pred - Y) / len(X)

    grad_w2 =  np.mean(dL_Lpred * a1)
    grad_b2 = np.mean(dL_Lpred)
    dL_da1 = dL_Lpred * weight_2

    dReLu = (z1 > 0).astype(float)
    dL_dz1 = dL_da1 * dReLu

    grad_w1 = np.mean(dL_dz1 * X)
    grad_b1 = np.mean(dL_dz1)

    weight_1 = weight_1 - learning_rate * grad_w1
    weight_2 = weight_2 - learning_rate * grad_w2

    bias_1 = bias_1 - learning_rate * grad_b1
    bias_2 = bias_2 - learning_rate * grad_b2

    if i % 200 == 0:
        print(f"Эпоха {i:4d} | Loss = {loss:.6f}")



'''

''' простая сеть для распознования 0 или 1 проще говоря XOR  на  numpy'''

'''class Layer:
    def __init__(self, size_input, size_output):

        self.cache = None
        self.weights = np.random.randn(size_input, size_output) * 0.1
        self.biases = np.zeros(size_output)

    def forward(self, X):
        self.cache = X
        return X @ self.weights + self.biases

    def backward(self, grad_output, learning_rate):
        grad_weights = self.cache.T @ grad_output
        grad_biases = np.sum(grad_output, axis=0)

        grad_input = grad_output @ self.weights.T

        self.weights -= grad_weights * learning_rate
        self.biases -= grad_biases * learning_rate

        return grad_input

class MSE:
    @staticmethod
    def loss(pred, y):
        return np.mean((pred - y) ** 2)

    @staticmethod
    def gradient(pred, y):
        return 2 * (pred - y) / len(y)

class ReLU:
    def forward(self, X):
        self.cache = X
        return np.maximum(0, X)

    def backward(self, grad_output, learning_rate):
        return grad_output * (self.cache > 0)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.activations  = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.activations.append(ReLU())
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
        loss = MSE.loss(pred, y)
        grad = MSE.gradient(pred, y)
        self.backward(grad, learning_rate)
        return loss, grad


np.random.seed(42)
X = np.random.randn(1000, 2)
y = (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)

model = NeuralNetwork([2, 8, 8, 1])

epochs = 500
learning_rate = 0.1
batch_size = 32

print("Начинаем обучение...")
loss_history = []

for epoch in range(epochs):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    total_loss = 0

    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        loss, _ = model.train_step(X_batch, y_batch, learning_rate)
        total_loss += loss

    avg_loss = total_loss / (len(X) / batch_size)
    loss_history.append(avg_loss)
    if epoch % 100 == 0:
        print(f"Эпоха {epoch}, средняя ошибка: {avg_loss:.6f}")

# Проверка на тесте (4 угла квадрата)
test_X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
test_y = (test_X[:, 0] * test_X[:, 1] > 0).astype(float).reshape(-1, 1)
preds = model.forward(test_X)
print("\nРезультат на тесте:")
print("Истина:   ", test_y.flatten())
print("Предсказания (округлённые):", np.round(preds).flatten())
print("Вероятности:               ", preds.flatten())

# Рисуем график ошибки
plt.plot(loss_history)
plt.title("График ошибки (MSE) во время обучения")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
'''
