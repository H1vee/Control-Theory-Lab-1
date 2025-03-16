import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Параметри варіанта 10
n, m, r = 3, 1, 100  # Параметри адаптивного регулятора

def desired_trajectory(t):
    return np.sin(0.1 * t)

# Передаточна функція: p / (p^2 + 2p + 1)
num = [1, 0]
den = [1, 2, 1]
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
extremum_point, = ax.plot([], [], 'ro', label='Точка екстремуму')
desired_extremum, = ax.plot([], [], 'go', label='Екстремум бажаної траєкторії')
ax.set_xlabel('Час')
ax.set_ylabel('Значення')
ax.axhline(0, color='black', linewidth=1)  # Додавання осі ординат
ax.axvline(0, color='black', linewidth=1)  # Додавання осі абсцис
ax.legend()
ax.set_title('Адаптивний регулятор')
ax.grid()

# Симуляція роботи регулятора
for i in range(1, len(T)):
    dt = T[i] - T[i - 1]
    error = desired_values[i] - x_real[i - 1]
    u[i] = b[0] * error  # Простий адаптивний контроль
    
    # Відгук системи
    _, y, _ = signal.lsim(system, U=[u[i-1], u[i]], T=[T[i-1], T[i]])
    x_real[i] = y[-1]
    
    # Пошук екстремуму реальної траєкторії
    if i > 1 and x_real[i-1] > x_real[i-2] and x_real[i-1] > x_real[i]:
        extremum_point.set_data(T[i-1], x_real[i-1])
    if i > 1 and x_real[i-1] < x_real[i-2] and x_real[i-1] < x_real[i]:
        extremum_point.set_data(T[i-1], x_real[i-1])
    
    # Пошук екстремуму бажаної траєкторії
    if i > 1 and desired_values[i-1] > desired_values[i-2] and desired_values[i-1] > desired_values[i]:
        desired_extremum.set_data(T[i-1], desired_values[i-1])
    if i > 1 and desired_values[i-1] < desired_values[i-2] and desired_values[i-1] < desired_values[i]:
        desired_extremum.set_data(T[i-1], desired_values[i-1])
    
    # Оновлення графіка в реальному часі
    line1.set_data(T[:i], desired_values[:i])
    line2.set_data(T[:i], x_real[:i])
    ax.set_xlim(0, T[i])
    ax.set_ylim(min(min(x_real), min(desired_values)) - 0.5, max(max(x_real), max(desired_values)) + 0.5)
    plt.pause(0.01)

plt.ioff()
plt.show()
