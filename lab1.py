import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Параметри варіанта 10
n, m, r = 3, 1, 100  # Параметри адаптивного регулятора
x1 = 0  # x(t)
x2 = 0  # x'(t)
def desired_trajectory(t):
    return np.sin(0.1 * t)

# Передаточна функція: 1 / (p^2 + 3p + 2)
num = [1]
den = [1, 3, 2]
system = signal.TransferFunction(num, den)

# Часовий інтервал
T = np.linspace(0, 50, 500)

# Вхідний сигнал (адаптивне керування)
u = np.zeros_like(T)
x_real = np.zeros_like(T)  # Реальна траєкторія

desired_values = desired_trajectory(T)  # Значення бажаної траєкторії

# Початкові коефіцієнти адаптивного регулятора
b = np.random.rand(m)  # Випадкові початкові значення

plt.ion()  # Увімкнення інтерактивного режиму графіка
fig, ax = plt.subplots(figsize=(10, 5))
line1, = ax.plot([], [], label='Бажана траєкторія y(t)', linestyle='dashed')
line2, = ax.plot([], [], label='Реальна траєкторія x(t)')
ax.set_xlabel('Час')
ax.set_ylabel('Значення')
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()
ax.set_title('Адаптивний регулятор')
ax.grid()

# Симуляція роботи регулятора
for i in range(1, len(T)):
    dt = T[i] - T[i - 1]
    error = desired_values[i] - x_real[i - 1]
    u[i] = b[0] * error  # Простий адаптивний контроль

    # Відгук системи
    dx1 = x2
    dx2 = u[i] - 3 * x2 - 2 * x1
    x1 += dt * dx1
    x2 += dt * dx2
    x_real[i] = x1

    # Оновлення графіка в реальному часі
    line1.set_data(T[:i], desired_values[:i])
    line2.set_data(T[:i], x_real[:i])
    ax.set_xlim(0, T[i])
    ax.set_ylim(min(min(x_real), min(desired_values)) - 0.5,
                max(max(x_real), max(desired_values)) + 0.5)
    plt.pause(0.01)

plt.ioff()
plt.show()
