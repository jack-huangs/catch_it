import numpy as np


class GravityKalman3D:
    """
    带重力先验的 3D Kalman Filter。

    状态:
    [x, y, z, vx, vy, vz]
    观测:
    [x, y, z]
    """

    def __init__(self, process_var=8e-2, measure_var=2e-3, gravity=9.81):
        self.process_var = process_var
        self.measure_var = measure_var
        self.gravity = gravity
        self.initialized = False
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32)
        self.last_timestamp = None
        self.last_measured_pos = None

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

    def _gravity_control(self, dt):
        """
        物体在世界坐标系里主要受重力影响：
        z = z + vz*dt - 0.5*g*dt^2
        vz = vz - g*dt
        """
        u = np.zeros((6, 1), dtype=np.float32)
        u[2, 0] = -0.5 * self.gravity * dt * dt
        u[5, 0] = -self.gravity * dt
        return u

    def update(self, pos3d, timestamp):
        pos3d = np.asarray(pos3d, dtype=np.float32).reshape(3, 1)

        if not self.initialized:
            self.x[:3] = pos3d
            self.x[3:] = 0.0
            self.P = np.eye(6, dtype=np.float32) * 0.05
            self.last_timestamp = float(timestamp)
            self.last_measured_pos = pos3d.copy()
            self.initialized = True
            return self.x[:3, 0].copy(), self.x[3:, 0].copy()

        dt = float(timestamp) - float(self.last_timestamp)
        dt = max(dt, 1e-3)
        self.last_timestamp = float(timestamp)

        # 直接差分测量能更快捕捉抛物线速度变化；
        # Kalman 结果再和差分速度融合，避免纯差分太抖。
        measured_vel = (pos3d - self.last_measured_pos) / dt
        self.last_measured_pos = pos3d.copy()

        F = self._transition(dt)
        Q = self._process_noise(dt)
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.eye(3, dtype=np.float32) * self.measure_var

        # predict
        self.x = F @ self.x + self._gravity_control(dt)
        self.P = F @ self.P @ F.T + Q

        # update
        y = pos3d - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        # 速度是位置差分最敏感的部分。这里用“Kalman速度 + 差分速度”融合：
        # - 差分速度响应快
        # - Kalman 速度更平滑
        # 这样比原来的纯常速度模型更适合抛物线来球。
        vel_blend = 0.65
        self.x[3:] = (1.0 - vel_blend) * self.x[3:] + vel_blend * measured_vel

        return self.x[:3, 0].copy(), self.x[3:, 0].copy()


# 保留旧类名，避免其他文件导入时需要大改。
ConstantVelocityKalman3D = GravityKalman3D
