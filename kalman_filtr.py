import numpy as np

class KalmanFilter:
    def __init__(self, dt, u, std_acc, x_std_meas, y_std_meas):
        """
        dt: time step
        u: acceleration magnitude
        std_acc: process noise standard deviation
        x_std_meas: measurement noise standard deviation for x
        y_std_meas: measurement noise standard deviation for y
        """
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas
        self.A = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])
        self.B = np.array([dt**2/2, dt, dt**2/2, dt]).reshape(4, 1)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        self.Q = np.array([[dt**4/4, dt**3/2, 0, 0],
                           [dt**3/2, dt**2, 0, 0],
                           [0, 0, dt**4/4, dt**3/2],
                           [0, 0, dt**3/2, dt**2]]) * std_acc**2
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])
        self.P = np.eye(self.A.shape[0])
        self.x = np.zeros((self.A.shape[0], 1))

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

if __name__ == "__main__":
    """ This will update the internal state estimate of the system based on the dynamic model and the process noise.
    To update the state estimate based on a new measurement, you can call the update method and pass in the measurement vector:
    z = np.array([[1.2], [0.8]])
    kf.update(z)
    This will update the internal state estimate based on the measurement and the measurement noise.
    You can repeat the predict and update steps for each time step to estimate the system state over time. 
    You can access the estimated state at any time step using the x attribute of the KalmanFilter instance:
    x_estimate = kf.x
    """
    kf = KalmanFilter(0.1, 0, 0.2, 0.1, 0.1)
    kf.predict()
    z = np.array([[1.2], [0.8]])
    kf.update(z)
    x_estimate = kf.x