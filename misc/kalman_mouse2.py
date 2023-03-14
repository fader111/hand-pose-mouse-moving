import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
import pyautogui

pyautogui.FAILSAFE = False # security

dt = 1.0/30  # time step
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([0, 0, 0, 0])  # initial state (x, y, vx, vy)
kf.F = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0 ,1]])  # state transition matrix

kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # measurement function
kf.P *= 1000.0 # initial state covariance
kf.R = np.array([[0.1, 0],[0, 0.1]]) # measurement noise covariance
kf.Q = np.array([[0.01, 0, 0.1, 0], [0, 0.01, 0, 0.1], [0.1, 0, 1, 0], [0, 0.1, 0, 1]]) # process noise covariance

cap = cv2.VideoCapture(2) # capture video
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
ret, frame = cap.read()
print(f"frame dims {frame.shape}")
while True:
    ret, frame = cap.read()
    
    # pyautogui.moveRel(-x, -y, _pause=False)
    # x, y = np.array([frame.shape[1], frame.shape[0]])/2  # initial position
    x, y = pyautogui.position()
    print(f"x y  {x} {y}")
    z = np.array([x, y]) # measurement (x, y)
    kf.predict()
    kf.update(z)
    x, y, _, _ = kf.x
    # pyautogui.moveRel(-x, -y, _pause=False)
    # pyautogui.move(x, y, _pause=False)
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1) # draw circle at predicted position
    cv2.imshow("frame", frame)
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()