import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметри адаптивного регулятора
n, m, r = 3, 1, 100

# Бажана траєкторія
def desired_trajectory(t):
    return np.sin(0.1 * t)

# Модель системи
def system_dynamics(t, y, u):
    return [y[1], -2*y[0] - 3*y[1] + u]

# Часові параметри
t_start, t_end, dt = 0, 50, 0.1
t_eval = np.arange(t_start, t_end, dt)
num_steps = len(t_eval)

# Ініціалізація масивів
x_real = np.zeros((num_steps, 2))
y_desired = desired_trajectory(t_eval)
u_control = np.zeros(num_steps)
x_history = np.zeros(num_steps + n + 1)
u_history = np.zeros(num_steps + m + 1)
a = np.zeros(n)
b = np.zeros(m)
b[0] = 1.0
learning_rate = 0.005

# Симуляція одного кроку
def simulate_step(t_span, x_init, u):
    sol = solve_ivp(
        lambda t, y: system_dynamics(t, y, u),
        t_span,
        x_init,
        method='RK45',
        t_eval=[t_span[1]]
    )
    return sol.y[:, 0]

# Підготовка графіка
plt.figure(figsize=(14, 10))
plt.ion()

print("Початок моделювання системи з адаптивним регулятором...")
x_real[0] = [0, 0]
x_history[n-1] = 0

# Основний цикл
for k in range(1, num_steps):
    current_t = t_eval[k-1]
    next_t = t_eval[k]
    t_span = [current_t, next_t]
    idx = min(k + n - 1, len(u_history) - 2)

    # Адаптація
    for s in range(10):
        J_grad_a = np.zeros(n)
        J_grad_b = np.zeros(m)
        for i in range(1, min(r, k)):
            hist_idx = idx - i
            predicted = 0
            for j in range(n):
                if hist_idx - j >= 0:
                    predicted += a[j] * x_history[hist_idx - j]
            for j in range(m):
                if hist_idx - j >= 0:
                    predicted += b[j] * u_history[hist_idx - j]
            actual = x_history[hist_idx + 1]
            error = actual - predicted
            for j in range(n):
                if hist_idx - j >= 0:
                    J_grad_a[j] += (2.0 / i) * error * x_history[hist_idx - j]
            for j in range(m):
                if hist_idx - j >= 0:
                    J_grad_b[j] += (2.0 / i) * error * u_history[hist_idx - j]
        a += learning_rate * J_grad_a
        b += learning_rate * J_grad_b

    if abs(b[0]) < 0.01:
        b[0] = 1.0

    next_desired = y_desired[k]
    sum_a_terms = sum(a[j] * x_history[idx - j] for j in range(n) if idx - j >= 0)
    sum_b_terms = sum(b[j] * u_history[idx - j] for j in range(1, m) if idx - j >= 0)
    u_control[k] = (next_desired - sum_a_terms - sum_b_terms) / b[0]
    u_control[k] = np.clip(u_control[k], -10.0, 10.0)
    u_history[idx] = u_control[k]
    x_new = simulate_step(t_span, x_real[k-1], u_control[k])
    x_real[k] = x_new
    x_history[idx + 1] = x_real[k, 0]

    # Візуалізація
    if k % 10 == 0 or k == num_steps - 1:
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(t_eval[:k+1], y_desired[:k+1], 'r--', linewidth=2, label='Бажана траєкторія y(t)')
        plt.plot(t_eval[:k+1], x_real[:k+1, 0], 'b-', linewidth=2, label='Реальна траєкторія x(t)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title(f'Адаптивний регулятор (n={n}, m={m}, r={r})')
        plt.ylabel('Амплітуда')

        plt.subplot(2, 1, 2)
        plt.plot(t_eval[:k+1], u_control[:k+1], 'g-', linewidth=2, label='Керування u(t)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('Час')
        plt.ylabel('Керування')

        plt.tight_layout()
        plt.pause(0.001)

plt.ioff()

# Фінальний графік
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_eval, y_desired, 'r--', linewidth=2, label='Бажана траєкторія y(t)')
plt.plot(t_eval, x_real[:, 0], 'b-', linewidth=2, label='Реальна траєкторія x(t)')
plt.grid(True)
plt.legend(loc='best')
plt.title(f'Адаптивний регулятор (n={n}, m={m}, r={r})')
plt.ylabel('Амплітуда')

plt.subplot(2, 1, 2)
plt.plot(t_eval, u_control, 'g-', linewidth=2, label='Керування u(t)')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Час')
plt.ylabel('Керування')

plt.tight_layout()
plt.savefig('adaptive_controller_result.png')
plt.show()

# Обчислення помилки
tracking_error = np.mean(np.abs(y_desired - x_real[:, 0]))
print(f"\nСередня абсолютна помилка відстеження: {tracking_error:.6f}")
