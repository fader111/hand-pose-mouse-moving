import numpy as np

# инициализация начальных значений
dt = 0.1  # время между измерениями
A = np.array([[1, dt], [0, 1]])  # матрица перехода состояний
H = np.array([[1, 0], [0, 1]])  # матрица измерений
Q = np.array([[0.01, 0], [0, 0.01]])  # ковариация шума процесса
R = np.array([[1, 0], [0, 1]])  # ковариация шума измерений
x = np.array([[0], [0]])  # начальное состояние
P = np.array([[1, 0], [0, 1]])  # начальная ковариация состояния

# функция обновления состояния фильтра Калмана
def update_Kalman(x, P, z):
    # предсказание состояния
    x = np.dot(A, x)
    P = np.dot(np.dot(A, P), A.T) + Q

    # коррекция состояния на основе измерений
    K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + R))
    x = x + np.dot(K, (z - np.dot(H, x)))
    P = np.dot((np.eye(2) - np.dot(K, H)), P)

    return x, P

# пример использования фильтра
measured_values = np.array([[0.5, 0.2], [1.2, 0.8], [2.1, 1.7]])  # измеренные координаты
filtered_values = []
for z in measured_values:
    x, P = update_Kalman(x, P, z)
    filtered_values.append(x)

print(filtered_values)
