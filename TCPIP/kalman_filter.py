import numpy as np

class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-2, measurement_variance=1e-1):
        self.x = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6)
        self.F = np.eye(6)
        self.F[0,3] = self.F[1,4] = self.F[2,5] = 1.0
        self.Q = np.eye(6) * process_variance
        self.H = np.zeros((3,6))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        self.R = np.eye(3) * measurement_variance

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        z = measurement
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_state(self):
        return self.x[:3]
