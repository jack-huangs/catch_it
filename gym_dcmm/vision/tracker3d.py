import numpy as np


class ConstantVelocityKalman3D:
    """
    一个简单的 3D 常速度 Kalman Filter。

    状态:
    [x, y, z, vx, vy, vz]
    观测:
    [x, y, z]
    """

    def __init__(self, process_var=1e-2, measure_var=5e-3):
        self.process_var = process_var
        self.measure_var = measure_var
        self.initialized = False
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32)
        self.last_timestamp = None

    def _transition(self, dt):
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _process_noise(self, dt):
        q = self.process_var
        Q = np.eye(6, dtype=np.float32) * q
        Q[0, 0] *= max(dt, 1e-3)
        Q[1, 1] *= max(dt, 1e-3)
        Q[2, 2] *= max(dt, 1e-3)
        Q[3, 3] *= 10 * max(dt, 1e-3)
        Q[4, 4] *= 10 * max(dt, 1e-3)
        Q[5, 5] *= 10 * max(dt, 1e-3)
        return Q

    def update(self, pos3d, timestamp):
        pos3d = np.asarray(pos3d, dtype=np.float32).reshape(3, 1)

        if not self.initialized:
            self.x[:3] = pos3d
            self.x[3:] = 0.0
            self.P = np.eye(6, dtype=np.float32) * 0.1
            self.last_timestamp = float(timestamp)
            self.initialized = True
            return self.x[:3, 0].copy(), self.x[3:, 0].copy()

        dt = float(timestamp) - float(self.last_timestamp)
        dt = max(dt, 1e-3)
        self.last_timestamp = float(timestamp)

        F = self._transition(dt)
        Q = self._process_noise(dt)
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.eye(3, dtype=np.float32) * self.measure_var

        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # update
        y = pos3d - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        return self.x[:3, 0].copy(), self.x[3:, 0].copy()

