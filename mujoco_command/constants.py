import numpy as np

################################################################################
# Mobile base
ARM_X_OFFSET = 0.1199
BASE_HEIGHT = 0.3948

# Vehicle center to steer axis (m)
h_x, h_y = 0.190150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.170150 * np.array([-1.0, 1.0, 1.0, -1.0])  # Kinova / Franka
# h_x, h_y = 0.140150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.120150 * np.array([-1.0, 1.0, 1.0, -1.0])  # ARX5

# Encoder magnet offsets
ENCODER_MAGNET_OFFSETS = [1465.0 / 4096, 175.0 / 4096, 1295.0 / 4096, 1595.0 / 4096]  # Base #1 (IPRL Kinova)

################################################################################
# Teleop and imitation learning

# Base and arm RPC servers
BASE_RPC_HOST = '192.168.1.11' # ip for NUC on ethernet interface
BASE_RPC_PORT = 50000
ARM_RPC_HOST = '192.168.1.11' # ip for NUC on ethernet interface
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b'secret password'
TELEOP_HOST = '0.0.0.0' # or 'localhost'
#TELEOP_HOST = '192.168.1.100' # if using USB-C + Ethernet cable for iPhone-laptop connection (recommended)

# Policy constants
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
POLICY_SERVER_HOST = 'localhost'
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
POLICY_IMAGE_WIDTH = 320
POLICY_IMAGE_HEIGHT = 240