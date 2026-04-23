"""
Author: Yuanhang Zhang
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the basic class of the DexCatch with Mobile Manipulation (DCMM) in the MuJoCo simulation environment.
"""
import os, sys
sys.path.append(os.path.abspath('../'))
import copy
import configs.env.DcmmCfg as DcmmCfg
import mujoco
from utils.util import calculate_arm_Te
from utils.pid import PID
import numpy as np
from utils.ik_pkg.ik_arm import IKArm
from utils.ik_pkg.ik_base import IKBase
from scipy.spatial.transform import Rotation as R
from collections import deque
import xml.etree.ElementTree as ET

# Function to convert XML file to string
def xml_to_string(file_path):
    try:
        # 读取 MuJoCo XML，并转成字符串；后面会用这个字符串重新构建模型
        tree = ET.parse(file_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is not None and compiler.get("meshdir") is not None:
            # from_xml_string 重新载入 XML 时，原始相对 meshdir 会失效；
            # 这里提前转成绝对路径，避免 tidybot 的网格文件找不到。
            meshdir = compiler.get("meshdir")
            compiler.set("meshdir", os.path.abspath(os.path.join(os.path.dirname(file_path), meshdir)))

        # 把 XML 树重新序列化成字符串
        xml_str = ET.tostring(root, encoding='unicode')
        
        return xml_str
    except Exception as e:
        print(f"Error: {e}")
        return None

def ensure_tidybot_tracking_xml(xml_str, object_name="object"):
    """
    给 tidybot 场景补上 Tracking 任务仍然需要的元素：
    1. floor：用于判断物体落地 / 接触地面
    2. object：继续沿用旧任务里的被抛物体
    """
    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Invalid MuJoCo XML: missing <worldbody>")

    # tidybot 原始 XML 没有地面，这里补一个简单平面
    floor = worldbody.find(".//geom[@name='floor']")
    if floor is None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "0 0 0.05",
                "rgba": "0.85 0.85 0.85 1",
                "friction": "1.0 0.3 0.1",
            },
        )

    # 继续沿用旧 Tracking 任务里的自由抛掷物体
    object_body = worldbody.find(f".//body[@name='{object_name}']")
    if object_body is None:
        object_body = ET.SubElement(
            worldbody,
            "body",
            {"name": object_name, "pos": "0.4 1.25 1.2"},
        )
        ET.SubElement(
            object_body,
            "inertial",
            {
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "mass": "0.25",
                "diaginertia": "7.33516e-05 7.33516e-05 7.33516e-05",
            },
        )
        ET.SubElement(
            object_body,
            "joint",
            {"name": object_name, "type": "free", "damping": "0.1"},
        )
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": object_name,
                "type": "box",
                "size": "0.02 0.025 0.02",
                "solimp": "0.998 0.998 0.001",
                "solref": "0.001 1",
                "density": "100",
                "friction": "0.95 0.3 0.1",
                "rgba": "1 0 0 1",
                "group": "0",
                "condim": "4",
            },
        )

    # 补入 free object 后，原 tidybot 的 keyframe qpos 长度会失配；Tracking 当前不依赖 keyframe，直接移除即可
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    return ET.tostring(root, encoding="unicode")

DEBUG_ARM = False
DEBUG_BASE = False


class MJ_DCMM(object):
    """
    Class of the DexCatch with Mobile Manipulation (DCMM)
    in the MuJoCo simulation environment.

    Args:
    - model: the MuJoCo model of the Dcmm
    - model_arm: the MuJoCo model of the arm
    - viewer: whether to show the viewer of the simulation
    - object_name: the name of the object in the MuJoCo model
    - timestep: the simulation timestep
    - open_viewer: whether to open the viewer initially

    """
    def __init__(self, 
                 model=None, 
                 model_arm=None, 
                 viewer=True, 
                 object_name='object',
                 object_eval=False, 
                 timestep=0.002):
        self.viewer = None
        self.open_viewer = viewer
        self.use_tidybot = getattr(DcmmCfg, "ROBOT_MODEL", "") == "tidybot"
        # 这里同时维护两套模型：
        # 1. self.model：完整机器人 + 物体，用于真正仿真
        # 2. self.model_arm：只包含机械臂，用于单独做 IK 求解
        if model is None:
            if not object_eval: model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_OBJECT_PATH)
            else: model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_UNSEEN_OBJECT_PATH)
            self.model_xml_string = xml_to_string(model_path)
            if self.use_tidybot:
                # tidybot 原始场景没有任务物体；这里提前补上 object / floor
                self.model_xml_string = ensure_tidybot_tracking_xml(self.model_xml_string, object_name=object_name)
        else:
            self.model = model
        if model_arm is None:
            if self.use_tidybot:
                # 新模型的 Tracking 直接做关节空间控制，不再依赖旧的 xArm6 IK 模型
                self.model_arm = None
            else:
                model_arm_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_ARM_PATH)
                self.model_arm = mujoco.MjModel.from_xml_path(model_arm_path)
        else:
            self.model_arm = model_arm
        self.model = mujoco.MjModel.from_xml_string(self.model_xml_string)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.data_arm = mujoco.MjData(self.model_arm) if self.model_arm is not None else None

        # 关节 / body / geom 名称在 tidybot 和旧模型里完全不同，这里统一整理成属性
        if self.use_tidybot:
            self.arm_joint_names = [f"joint_{i}" for i in range(1, 8)]
            self.hand_joint_names = ["right_driver_joint"]
            self.arm_qpos_indices = np.array([self.model.joint(name).qposadr[0] for name in self.arm_joint_names], dtype=int)
            self.hand_qpos_indices = np.array([self.model.joint(name).qposadr[0] for name in self.hand_joint_names], dtype=int)
            self.ee_body_name = DcmmCfg.ee_body_name
            self.ee_site_name = DcmmCfg.ee_site_name
            self.base_body_name = DcmmCfg.base_body_name
            self.arm_base_body_name = DcmmCfg.arm_base_body_name
            self.pad_geom_ids = np.array(
                [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in DcmmCfg.pad_geom_names],
                dtype=int,
            )
            base_body = self.model.body(self.base_body_name)
            start = int(base_body.geomadr[0])
            count = int(base_body.geomnum[0])
            self.base_geom_ids = np.arange(start, start + count, dtype=int)
            self.target_base_pose = np.zeros(3)
        else:
            self.arm_joint_names = []
            self.hand_joint_names = []
            self.arm_qpos_indices = np.arange(15, 21, dtype=int)
            self.hand_qpos_indices = np.arange(21, 37, dtype=int)
            self.ee_body_name = "link6"
            self.ee_site_name = None
            self.base_body_name = "base_link"
            self.arm_base_body_name = "arm_base"
            self.pad_geom_ids = np.array([], dtype=int)
            self.base_geom_ids = np.array([], dtype=int)
            self.target_base_pose = np.zeros(3)

        # 先把机械臂和夹爪放到默认初始位置
        self.data.qpos[self.arm_qpos_indices] = DcmmCfg.arm_joints[:]
        self.data.qpos[self.hand_qpos_indices] = DcmmCfg.hand_joints[:]
        if self.data_arm is not None and not self.use_tidybot:
            self.model_arm.opt.timestep = timestep
            self.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
        mujoco.mj_forward(self.model, self.data)
        if self.data_arm is not None:
            mujoco.mj_forward(self.model_arm, self.data_arm)
        self.arm_base_pos = self.data.body(self.arm_base_body_name).xpos
        if self.use_tidybot:
            self.current_ee_pos = copy.deepcopy(self.data.site(self.ee_site_name).xpos)
            self.current_ee_quat = copy.deepcopy(self.data.body(self.ee_body_name).xquat)
            self.target_base_pose[:] = self.data.qpos[0:3]
        else:
            self.current_ee_pos = copy.deepcopy(self.data_arm.body("link6").xpos)
            self.current_ee_quat = copy.deepcopy(self.data_arm.body("link6").xquat)

        # 检查 XML 里是否真的有目标物体；后面接触检测、抛掷都依赖它
        try:
            _ = self.data.body(object_name)
        except:
            print("The object name is not found in the model!\
                  \nPlease check the object name in the .xml file.")
            raise ValueError
        self.object_name = object_name
        # Get the geom id of the hand, the floor and the object
        if self.use_tidybot:
            self.hand_start_id = -1
        else:
            self.hand_start_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'mcp_joint') - 1
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)

        # ------------------------------
        # 底盘 / 机械臂 / 手部的低层控制器
        # PPO 不直接输出电机力矩，而是先给“目标”
        # 再由 PID 控制器把目标转成可执行的控制量
        # ------------------------------
        self.rp_base = np.zeros(3)
        self.rp_ref_base = np.zeros(3)
        self.drive_pid = PID("drive", DcmmCfg.Kp_drive, DcmmCfg.Ki_drive, DcmmCfg.Kd_drive, dim=4, llim=DcmmCfg.llim_drive, ulim=DcmmCfg.ulim_drive, debug=False)
        self.steer_pid = PID("steer", DcmmCfg.Kp_steer, DcmmCfg.Ki_steer, DcmmCfg.Kd_steer, dim=4, llim=DcmmCfg.llim_steer, ulim=DcmmCfg.ulim_steer, debug=False)
        self.arm_pid = PID("arm", DcmmCfg.Kp_arm, DcmmCfg.Ki_arm, DcmmCfg.Kd_arm, dim=len(DcmmCfg.arm_joints), llim=DcmmCfg.llim_arm, ulim=DcmmCfg.ulim_arm, debug=False)
        self.hand_pid = PID("hand", DcmmCfg.Kp_hand, DcmmCfg.Ki_hand, DcmmCfg.Kd_hand, dim=len(DcmmCfg.hand_joints), llim=DcmmCfg.llim_hand, ulim=DcmmCfg.ulim_hand, debug=False)
        self.cmd_lin_y = 0.0
        self.cmd_lin_x = 0.0
        self.arm_act = False
        self.steer_ang = np.array([0.0, 0.0, 0.0, 0.0])
        self.drive_vel = np.array([0.0, 0.0, 0.0, 0.0])

        # 机械臂 IK 求解器：
        # 输入“末端想去的位置/姿态”，输出“6 个关节应该到哪里”
        self.ik_arm = None
        if not self.use_tidybot:
            self.ik_arm = IKArm(solver_type=DcmmCfg.ik_config["solver_type"], ilimit=DcmmCfg.ik_config["ilimit"], 
                                ps=DcmmCfg.ik_config["ps"], λΣ=DcmmCfg.ik_config["λΣ"], tol=DcmmCfg.ik_config["ee_tol"])

        # 离屏渲染相机参数，后面环境会拿这个相机做 RGB / depth 观测
        self.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], DcmmCfg.cam_config["name"])

        # 目标量：每一步控制前，环境会先更新这些 target，
        # 再由 PID 去追踪这些 target
        self.target_base_vel = np.zeros(3)
        self.target_arm_qpos = np.zeros(len(DcmmCfg.arm_joints))
        self.target_hand_qpos = np.zeros(len(DcmmCfg.hand_joints))
        self.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        self.target_hand_qpos[:] = DcmmCfg.hand_joints[:]

        self.ik_solution = np.zeros(len(DcmmCfg.arm_joints))

        self.vel_history = deque(maxlen=4)  # store the last 2 velocities
        self.vel_init = False

        if self.use_tidybot:
            self.drive_ctrlrange = np.array([-1.0, 1.0])
            self.steer_ctrlrange = np.array([-1.0, 1.0])
        else:
            self.drive_ctrlrange = self.model.actuator(4).ctrlrange
            self.steer_ctrlrange = self.model.actuator(0).ctrlrange

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body(i).name))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}, Damping: {}".format(
                    i, self.model.joint(i).name, self.model.jnt_range[i], self.model.dof_damping[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.data.ctrl)))
        for i in range(len(self.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}".format(
                    i,
                    self.model.actuator(i).name,
                    self.model.joint(self.model.actuator(i).trnid[0]).name,
                    self.model.actuator(i).ctrlrange,
                )
            )
        print("\nMobile Base PID Info: \n")
        print(
            "Drive, P: {}, I: {}, D: {}".format(
                self.drive_pid.Kp,
                self.drive_pid.Ki,
                self.drive_pid.Kd,
            )
        )
        print(
            "Steer, P: {}, I: {}, D: {}".format(
                self.steer_pid.Kp,
                self.steer_pid.Ki,
                self.steer_pid.Kd,
            )
        )
        print("\nArm PID Info: \n")
        print(
            "P: {}, I: {}, D: {}".format(
                self.arm_pid.Kp,
                self.arm_pid.Ki,
                self.arm_pid.Kd,
            )
        )
        print("\nHand PID Info: \n")
        print(
            "P: {}, I: {}, D: {}".format(
                self.hand_pid.Kp,
                self.hand_pid.Ki,
                self.hand_pid.Kd,
            )
        )

        print("\nCamera Info: \n")
        for i in range(self.model.ncam):
            print(
                "Camera ID: {}, Camera Name: {}, Camera Mode: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}, \n Intrinsic Matrix: \n{}".format(
                    i,
                    self.model.camera(i).name,
                    self.model.cam_mode[i],
                    self.model.cam_fovy[i],
                    self.model.cam_pos[i],
                    self.model.cam_quat[i],
                    # self.model.cam_pos0[i],
                    # self.model.cam_mat0[i].reshape((3, 3)),
                    self.cam_matrix,
                )
            )
        print("\nSimulation Timestep: ", self.model.opt.timestep)
    
    def move_base_vel(self, target_base_vel):
        if self.use_tidybot:
            # tidybot 的底盘是 x/y/theta 三个位置执行器；
            # Tracking 里我们把策略输出当作平面速度，再积分成下一时刻的位置目标。
            dt = self.model.opt.timestep
            self.target_base_pose[0] += target_base_vel[0] * dt
            self.target_base_pose[1] += target_base_vel[1] * dt
            return np.zeros(0), np.zeros(0)
        # 先把“底盘目标速度”通过 IKBase 转成：
        # 1. 四个转向轮应该转到的角度
        # 2. 四个驱动轮应该达到的速度
        self.steer_ang, self.drive_vel = IKBase(target_base_vel[0], target_base_vel[1], target_base_vel[2])
        # 读取当前底盘状态，后面交给 PID 做闭环控制
        # TODO: angular velocity is not correct when the robot is self-rotating.
        current_steer_pos = np.array([self.data.joint("steer_fl").qpos[0],
                                      self.data.joint("steer_fr").qpos[0], 
                                      self.data.joint("steer_rl").qpos[0],
                                      self.data.joint("steer_rr").qpos[0]])
        current_drive_vel = np.array([self.data.joint("drive_fl").qvel[0],
                                      self.data.joint("drive_fr").qvel[0], 
                                      self.data.joint("drive_rl").qvel[0],
                                      self.data.joint("drive_rr").qvel[0]])
        # PID 根据“目标 - 当前”的误差输出控制量
        mv_steer = self.steer_pid.update(self.steer_ang, current_steer_pos, self.data.time)
        mv_drive = self.drive_pid.update(self.drive_vel, current_drive_vel, self.data.time)
        # 如果车轮正在向目标方向加速，这里人为限制一下 drive 控制量，避免加速太猛
        if np.all(current_drive_vel > 0.0) and np.all(current_drive_vel < self.drive_vel):
            mv_drive = np.clip(mv_drive, 0, self.drive_ctrlrange[1] / 10.0)
        if np.all(current_drive_vel < 0.0) and np.all(current_drive_vel > self.drive_vel):
            mv_drive = np.clip(mv_drive, self.drive_ctrlrange[0] / 10.0, 0)
        
        # 转向控制量必须落在 MuJoCo actuator 的允许范围内
        mv_steer = np.clip(mv_steer, self.steer_ctrlrange[0], self.steer_ctrlrange[1])
        
        return mv_steer, mv_drive
    
    def move_ee_pose(self, delta_pose):
        """
        Move the end-effector to the target pose.
        delta_pose[0:3]: delta x,y,z
        delta_pose[3:6]: delta euler angles roll, pitch, yaw

        Return:
        - The target joint positions of the arm
        """
        if self.use_tidybot:
            raise NotImplementedError("tidybot Tracking 已切到关节空间控制，不再走旧的末端 IK。")
        # 先读取当前末端位姿
        self.current_ee_pos[:] = self.data_arm.body("link6").xpos[:]
        self.current_ee_quat[:] = self.data_arm.body("link6").xquat[:]
        # PPO 给的是末端位姿增量，这里把增量叠加到当前末端上
        target_pos = self.current_ee_pos + delta_pose[0:3]
        r_delta = R.from_euler('zxy', delta_pose[3:6])
        r_current = R.from_quat(self.current_ee_quat)
        target_quat = (r_delta * r_current).as_quat()
        # 用 IK 把“目标末端位姿”转换成“目标关节角”
        result_QP = self.ik_arm_solve(target_pos, target_quat)
        if DEBUG_ARM: print("result_QP: ", result_QP)
        # 把 IK 的结果写到独立的机械臂模型里，便于后面继续算新的末端状态
        self.data_arm.qpos[0:6] = result_QP[0]
        mujoco.mj_fwdPosition(self.model_arm, self.data_arm)
        
        # 末端离机械臂基座的长度，用于做工作空间/越界相关判断
        relative_ee_pos = target_pos - self.data_arm.body("arm_base").xpos
        ee_length = np.linalg.norm(relative_ee_pos)

        return result_QP, ee_length
    
    def ik_arm_solve(self, target_pose, target_quate):
        """
        Solve the IK problem for the arm.
        """
        if self.ik_arm is None:
            raise NotImplementedError("tidybot Tracking 不使用旧的 IKArm 求解器。")
        # 目标位姿先转成 IK 求解器需要的齐次变换矩阵
        Tep = calculate_arm_Te(target_pose, target_quate)
        if DEBUG_ARM: print("Tep: ", Tep)
        # 给定目标位姿和当前关节角，求一个新的 6 维关节解
        result_QP = self.ik_arm.solve(self.model_arm, self.data_arm, Tep, self.data_arm.qpos[0:6])
        return result_QP

    def set_throw_pos_vel(self, 
                          pose = np.array([0, 0, 0, 1, 0, 0, 0]), 
                          velocity = np.array([0, 0, 0, 0, 0, 0])):
        # 直接设置物体的位姿和速度；环境 reset / 抛掷阶段会调用
        object_joint = self.model.joint(self.object_name)
        qpos_adr = int(object_joint.qposadr[0])
        qvel_adr = int(object_joint.dofadr[0])
        self.data.qpos[qpos_adr:qpos_adr + 7] = pose
        self.data.qvel[qvel_adr:qvel_adr + 6] = velocity

    def action_hand2qpos(self, action_hand):
        """
        Convert the action of the hand to the joint positions.
        """
        if self.use_tidybot:
            # tidybot 夹爪只有 1 个控制量；Tracking 默认保持张开，只允许小范围微调
            self.target_hand_qpos[0] = np.clip(
                self.target_hand_qpos[0] + float(action_hand[0]),
                0.0,
                255.0,
            )
            return
        # PPO 输出的是 12 维手部动作增量；
        # 这里把它映射到 16 维 Leap Hand 目标关节角上
        # Thumb
        self.target_hand_qpos[13] += action_hand[9]
        self.target_hand_qpos[14] += action_hand[10]
        self.target_hand_qpos[15] += action_hand[11]
        # Other Three Fingers
        self.target_hand_qpos[0] += action_hand[0]
        self.target_hand_qpos[2] += action_hand[1]
        self.target_hand_qpos[3] += action_hand[2]
        self.target_hand_qpos[4] += action_hand[3]
        self.target_hand_qpos[6] += action_hand[4]
        self.target_hand_qpos[7] += action_hand[5]
        self.target_hand_qpos[8] += action_hand[6]
        self.target_hand_qpos[10] += action_hand[7]
        self.target_hand_qpos[11] += action_hand[8]

    def pixel_2_world(self, pixel_x, pixel_y, depth, camera="top"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            camera: Name of camera used to obtain the image.
        """

        # 之前这里只在第一次调用时初始化相机参数。
        # 现在项目里会同时用 wrist / base 等不同相机，
        # 所以只要 camera 变了，就必须重新计算对应相机的内外参。
        if (not self.cam_init) or (getattr(self, "cam_name", None) != camera):
            self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], camera)

        # 对于 targetbody / tracking 等动态相机，外参不是 XML 里的静态 cam_mat0，
        # 而应该使用 MuJoCo 在当前时刻算出来的真实世界位姿。
        # 否则只要相机朝向发生变化，深度反投影出来的 3D 坐标就会整体偏掉。
        cam_id = self.model.camera(camera).id
        self.cam_rot_mat = np.reshape(self.data.cam_xmat[cam_id], (3, 3)) @ np.array(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        )
        self.cam_pos = self.data.cam_xpos[cam_id]

        # 先把像素点从图像坐标还原到相机坐标系
        pixel_coord = np.array([pixel_x, 
                                pixel_y, 
                                1]) * (depth)
        
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # MuJoCo 相机坐标系和世界坐标系轴定义不同，这里要做轴变换
        pos_c[1] *= -1
        pos_c[1], pos_c[2] = pos_c[2], pos_c[1]
        # 最后再变换到世界坐标系
        pos_w = self.cam_rot_mat @ (pos_c) + self.cam_pos

        return pos_c, pos_w

    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """

        # MuJoCo depth buffer 不是直接的米，需要根据 near / far 平面反算
        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend

        return near / (1 - depth * (1 - near / far))

    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """
        # 这里只计算一次相机内参和外参，后面像素转世界坐标会重复用到
        cam_id = self.model.camera(camera).id
        fovy = self.model.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # 相机内参矩阵 K
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # 外参（cam_rot_mat / cam_pos）对动态相机需要每次实时刷新，
        # 所以这里只先做初始化占位，真正值在 pixel_2_world() 里按当前 data 更新。
        self.cam_rot_mat = np.eye(3)
        if self.use_tidybot:
            self.cam_pos = self.data.cam_xpos[cam_id]
        else:
            self.cam_pos = self.model.cam_pos0[cam_id] + self.data.body("base_link").xpos - self.data.body("arm_base").xpos
        self.cam_init = True
        self.cam_name = camera

    def set_arm_target_qpos(self, delta_qpos):
        """
        tidybot Tracking 使用关节空间控制：
        PPO 直接输出 7 个关节增量，这里把它累加到目标关节角上并裁剪到关节范围内。
        """
        if not self.use_tidybot:
            raise NotImplementedError("旧模型仍使用末端 IK 控制。")
        current_target = self.target_arm_qpos.copy()
        target_qpos = current_target + delta_qpos
        arm_low = []
        arm_high = []
        for name in self.arm_joint_names:
            joint_id = self.model.joint(name).id
            # tidybot 里并不是每个关节都显式设置了 joint range。
            # 对这些“无限位”关节，MuJoCo 的 jnt_range 默认会是 [0, 0]；
            # 如果这里仍然机械地按 jnt_range 裁剪，就会把目标关节角错误地强行裁到 0。
            # 之前 joint_3 / joint_7 一直自己回零，根因就在这里。
            if bool(self.model.jnt_limited[joint_id]):
                arm_low.append(self.model.jnt_range[joint_id][0])
                arm_high.append(self.model.jnt_range[joint_id][1])
            else:
                arm_low.append(-np.inf)
                arm_high.append(np.inf)
        arm_low = np.array(arm_low)
        arm_high = np.array(arm_high)
        clipped_qpos = np.clip(target_qpos, arm_low, arm_high)
        success = np.allclose(target_qpos, clipped_qpos)
        self.target_arm_qpos[:] = clipped_qpos
        return clipped_qpos, success
