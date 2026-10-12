import torch
import matplotlib.pyplot as plt


# ============================================================
# 1. Gradient Descent с нуля
# ============================================================

class GradientDescent:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ============================================================
# 2. Adam с нуля
# ============================================================

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        # Моменты
        self.m = [torch.zeros_like(p) for p in self.params]  # 1-й момент
        self.v = [torch.zeros_like(p) for p in self.params]  # 2-й момент
        self.t = 0  # шаг

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad

                # Обновление моментов
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Обновление параметров
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ============================================================
# 3. Линейная регрессия с нуля (y = Xw + b)
# ============================================================

def linear_regression_demo():
    torch.manual_seed(42)

    # Данные: y = 3x + 2 + шум
    X = torch.linspace(-3, 3, 100).unsqueeze(1)  # (100, 1)
    true_w, true_b = 17.0, 2.0
    y = true_w * X + true_b + 0.5 * torch.randn_like(X)

    # Параметры
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # Оптимизатор
    optimizer = Adam([w, b], lr=0.1)

    losses = []

    for epoch in range(300):
        # Forward
        y_pred = X @ w + b

        # MSE loss
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | w={w.item():.3f} | b={b.item():.3f}")

    print(f"\nИстинные: w={true_w}, b={true_b}")
    print(f"Найденные: w={w.item():.4f}, b={b.item():.4f}")

    # График
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X.numpy(), y.numpy(), alpha=0.6, label='Данные')
    plt.plot(X.numpy(), (X @ w + b).detach().numpy(), 'r-', linewidth=2, label='Предсказание')
    plt.legend()
    plt.title("Линейная регрессия")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Логистическая регрессия с нуля
# ============================================================

def logistic_regression_demo():
    torch.manual_seed(42)

    # Данные: два класса
    n = 150
    X0 = torch.randn(n, 2) + torch.tensor([-2.0, -1.5])  # класс 0
    X1 = torch.randn(n, 2) + torch.tensor([2.0, 1.5])  # класс 1
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).unsqueeze(1)

    # Параметры
    w = torch.randn(2, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = Adam([w, b], lr=0.1)

    losses = []
    accuracies = []

    for epoch in range(400):
        # Forward
        logits = X @ w + b
        probs = torch.sigmoid(logits)

        # Binary Cross Entropy
        # стабильная версия: - (y*log(p) + (1-y)*log(1-p))
        eps = 1e-7
        loss = -(y * torch.log(probs + eps) + (1 - y) * torch.log(1 - probs + eps)).mean()
        losses.append(loss.item())

        # Accuracy
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean().item()
        accuracies.append(acc)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    print(f"\nФинальная accuracy: {accuracies[-1]:.3f}")

    # Визуализация
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X0[:, 0], X0[:, 1], alpha=0.6, label='Класс 0')
    plt.scatter(X1[:, 0], X1[:, 1], alpha=0.6, label='Класс 1')

    # Разделяющая прямая: w1*x + w2*y + b = 0  →  y = -(w1*x + b)/w2
    x_line = torch.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_line = -(w[0] * x_line + b) / w[1]
    plt.plot(x_line.detach(), y_line.detach(), 'k-', linewidth=2, label='Граница')
    plt.legend()
    plt.title("Логистическая регрессия")

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss')
    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.title("Обучение")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("ЛИНЕЙНАЯ РЕГРЕССИЯ + Adam")
    print("=" * 50)
    linear_regression_demo()

    print("\n" + "=" * 50)
    print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ + Adam")
    print("=" * 50)
    logistic_regression_demo()