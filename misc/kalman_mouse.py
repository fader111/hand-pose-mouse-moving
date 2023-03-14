import numpy as np
import pyautogui

# Define system model parameters
dt = 0.1  # Time step
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

B = np.array([[0.5*dt**2, 0],
              [0, 0.5*dt**2],
              [dt, 0],
              [0, dt]])

Q = np.diag([0.1, 0.1, 0.01, 0.01])

# Define measurement model parameters
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

R = np.diag([10, 10])

# Initialize state vector and covariance matrix
x = np.array([0, 0, 0, 0])
P = np.diag([1000, 1000, 1000, 1000])

# Set target position
target_pos = np.array([500, 500])

# Main loop
while True:
    # Get current mouse position
    mouse_pos = np.array(pyautogui.position())

    # Calculate control input
    u = B.T @ np.linalg.inv(B @ B.T) @ (target_pos - H @ x[:2])

    # Prediction step
    x[:2] += dt * x[2:]
    x[2:] += u
    P = F @ P @ F.T + Q

    # Update step
    y = mouse_pos - H @ x[:2]
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x += K @ y
    P = (np.eye(4) - K @ H) @ P

    # Move mouse pointer to filtered position
    pyautogui.moveTo(x[0], x[1], duration=0)

