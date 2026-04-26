# Author: Jimmy Wu
# Date: October 2024

import threading
import time
import cv2 as cv
import numpy as np
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2
from envs.utils.kinova import DeviceConnection
import pyrealsense2 as rs
import multiprocessing as mp

class Camera:
    def __init__(self):
        self.image = None
        self.last_read_time = time.time()
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        # Note: We read frames at 30 fps but not every frame is necessarily
        # saved during teleop or used during policy inference
        while True:
            # Reading new frames too quickly causes latency spikes
            while time.time() - self.last_read_time < 0.0333:  # 30 fps
                time.sleep(0.0001)
            _, bgr_image = self.cap.read()
            self.last_read_time = time.time()
            if bgr_image is not None:
                self.image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def close(self):
        self.cap.release()

class LogitechCamera(Camera):
    def __init__(self, serial, frame_width=640, frame_height=360, focus=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = focus  # Note: Set this to 100 when using fisheye lens attachment
        self.cap = self.get_cap(serial)
        super().__init__()

    def get_cap(self, serial):
        cap = cv.VideoCapture(f'/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0')
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Important - results in much better latency

        # Disable autofocus
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

        # Read several frames to let settings (especially gain/exposure) stabilize
        for _ in range(30):
            cap.read()
            cap.set(cv.CAP_PROP_FOCUS, self.focus)  # Fixed focus

        # Check all settings match expected
        assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == self.frame_width
        assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == self.frame_height
        assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
        assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
        assert cap.get(cv.CAP_PROP_FOCUS) == self.focus

        return cap

def find_fisheye_center(image):
    # Find contours
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Fit a minimum enclosing circle around all contours
    return cv.minEnclosingCircle(np.vstack(contours))

def check_fisheye_centered(image):
    height, width, _ = image.shape
    center, _ = find_fisheye_center(image)
    if center is None:
        return True
    return abs(width / 2 - center[0]) < 0.05 * width and abs(height / 2 - center[1]) < 0.05 * height

class KinovaCamera(Camera):
    def __init__(self):
        # GStreamer video capture (see https://github.com/Kinovarobotics/kortex/issues/88)
        # Note: max-buffers=1 and drop=true are added to reduce latency spikes
        self.cap = cv.VideoCapture('rtspsrc location=rtsp://192.168.1.10/color latency=0 ! decodebin ! videoconvert ! appsink sync=false max-buffers=1 drop=true', cv.CAP_GSTREAMER)
        # self.cap = cv.VideoCapture('rtsp://192.168.1.10/color', cv.CAP_FFMPEG)  # This stream is high latency but works with pip-installed OpenCV
        assert self.cap.isOpened(), 'Unable to open stream. Please make sure OpenCV was built from source with GStreamer support.'

        # Apply camera settings
        threading.Thread(target=self.apply_camera_settings, daemon=True).start()
        super().__init__()

        # Wait for camera to warm up
        image = None
        while image is None:
            image = self.get_image()

        # Make sure fisheye lens did not accidentally get bumped
        # TODO: FIXME
        #if not check_fisheye_centered(image):
        #    raise Exception('The fisheye lens on the Kinova wrist camera appears to be off-center')

    def apply_camera_settings(self):
        # Note: This function adds significant camera latency when it is called
        # directly in __init__, so we call it in a separate thread instead

        # Use Kortex API to set camera settings
        with DeviceConnection.createTcpConnection() as router:
            device_manager = DeviceManagerClient(router)
            vision_config = VisionConfigClient(router)

            # Get vision device ID
            device_handles = device_manager.ReadAllDevices()
            vision_device_ids = [
                handle.device_identifier for handle in device_handles.device_handle
                if handle.device_type == DeviceConfig_pb2.VISION
            ]
            assert len(vision_device_ids) == 1
            vision_device_id = vision_device_ids[0]

            # Check that resolution, frame rate, and bit rate are correct
            sensor_id = VisionConfig_pb2.SensorIdentifier()
            sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_settings = vision_config.GetSensorSettings(sensor_id, vision_device_id)
            try:
                assert sensor_settings.resolution == VisionConfig_pb2.RESOLUTION_640x480  # FOV 65 ± 3° (diagonal)
                assert sensor_settings.frame_rate == VisionConfig_pb2.FRAMERATE_30_FPS
                assert sensor_settings.bit_rate == VisionConfig_pb2.BITRATE_10_MBPS
            except:
                sensor_settings.sensor = VisionConfig_pb2.SENSOR_COLOR
                sensor_settings.resolution = VisionConfig_pb2.RESOLUTION_640x480
                sensor_settings.frame_rate = VisionConfig_pb2.FRAMERATE_30_FPS
                sensor_settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS
                vision_config.SetSensorSettings(sensor_settings, vision_device_id)
                assert False, 'Incorrect Kinova camera sensor settings detected, please restart the camera to apply new settings'

            # Disable autofocus and set manual focus to infinity
            # Note: This must be called after the OpenCV stream is created,
            # otherwise the camera will still have autofocus enabled
            sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
            sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
            sensor_focus_action.manual_focus.value = 0
            vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)

class RealSenseCamera(Camera):
    def __init__(self, serial_number, frame_width=640, frame_height=480, fps=30, use_depth=True):
        self.serial_number = serial_number
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.use_depth = use_depth

        self.image = None
        self.depth = None
        self.last_read_time = time.time()

        #rs.log_to_console(rs.log_severity.debug)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)
        if use_depth:
            self.config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)

        self.profile = self.pipeline.start(self.config)

        # Align depth to color
        if use_depth:
            self.align = rs.align(rs.stream.color)

        #self.rs_stream = rs_stream

        self.device = self.profile.get_device()
        for sensor in self.device.query_sensors():
            # Exposure (disable auto, set manual)
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
            # White balance (disable auto, set manual)
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 1)
            #if sensor.supports(rs.option.visual_preset):
            #    sensor.set_option(rs.option.visual_preset, 5) # Close range, 0 = default
            # Queue size
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)

        # Warmup frames
        for _ in range(10):
            self.pipeline.wait_for_frames()

        self.depth_filters = [
            rs.spatial_filter().process,
            rs.hole_filling_filter().process,
            rs.temporal_filter().process,
        ]

        # Start worker thread
        threading.Thread(target=self.camera_worker, daemon=True).start()


    def camera_worker(self):
        while True:
            while time.time() - self.last_read_time < 1.0 / self.fps:
                time.sleep(0.0001)
    
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception as e:
                #print(f"[RealSense] Failed to get frames: {e}")
                continue
    
            if not frames:
                continue
    
            # If using depth, try alignment and verify success
            if self.use_depth:
                try:
                    frames = self.align.process(frames)
                except Exception as e:
                    print(f"[RealSense] Align failed: {e}")
                    continue
    
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.use_depth else None
            for rs_filter in self.depth_filters:
                depth_frame = rs_filter(depth_frame)
    
            # Ensure both are valid
            if not color_frame or (self.use_depth and not depth_frame):
                continue
    
            try:
                bgr_image = np.asanyarray(color_frame.get_data())
                color = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    
                if self.use_depth:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    if len(depth_image.shape) == 2:
                        depth_image = depth_image[..., None]
            except Exception as e:
                print(f"[RealSense] Frame decoding failed: {e}")
                continue
    
            # Only update if both are valid
            self.image = color
            if self.use_depth:
                self.depth = depth_image.copy()
    
            self.last_read_time = time.time()

    def get_image(self):
        return self.image

    def get_depth(self):
        return self.depth if self.use_depth else None

    def get_rgbd(self):
        return self.image, self.depth

    def get_intrinsics(self):
        cprofile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        cintrinsics = cprofile.get_intrinsics()
        depth_scale = self.device.first_depth_sensor().get_depth_scale()
        dist_coeffs = np.array(cintrinsics.coeffs)
        return dict(
            matrix=np.array([
                [cintrinsics.fx, 0, cintrinsics.ppx],
                [0, cintrinsics.fy, cintrinsics.ppy],
                [0, 0, 1.0],
            ]),
            width=cintrinsics.width,
            height=cintrinsics.height,
            depth_scale=depth_scale,
            distortion=dist_coeffs,
        )

    def close(self):
        self.pipeline.stop()

if __name__ == '__main__':
#    import open3d as o3d
#    import numpy as np
#    import cv2 as cv
#    from common_utils import Stopwatch
#    from envs.utils.cameras import RealSenseCamera, KinovaCamera
#    
#    base_camera = RealSenseCamera("247122072471", use_depth=True)
#    wrist_camera = KinovaCamera()
#    
#    vis = o3d.visualization.Visualizer()
#    vis.create_window(window_name="Colorized PointCloud", width=960, height=540)
#    pcd = o3d.geometry.PointCloud()
#    vis.add_geometry(pcd)
#    
#    stopwatch = Stopwatch()
#    try:
#        while True:
#            with stopwatch.time('base_image'):
#                rgb, depth, points = base_camera.get_rgbd()
#            with stopwatch.time('wrist_image'):
#                wrist_image = wrist_camera.get_image()
#    
#            if rgb is None or depth is None or points is None:
#                continue
#    
#            # Show RGB feeds
#            cv.imshow('base_image', cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
#            cv.imshow('wrist_image', cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
#            key = cv.waitKey(1)
#            if key == 27:  # ESC
#                break
#    
#            # Flatten the (H, W, 3) → (N, 3)
#            flat_points = points.reshape(-1, 3)
#    
#            # Filter out invalid depth points (e.g., zero or NaN)
#            #valid_mask = np.isfinite(flat_points).all(axis=1) & (flat_points[:, 2] > 0)
#            #valid_points = flat_points[valid_mask]
#    
#            # Corresponding RGB values
#            flat_rgb = rgb.reshape(-1, 3) / 255.0  # normalize to [0, 1]
#            #valid_colors = flat_rgb[valid_mask]
#
#
#            # Update point cloud
#            pcd.points = o3d.utility.Vector3dVector(flat_points)
#            pcd.colors = o3d.utility.Vector3dVector(flat_rgb)
#    
#            vis.add_geometry(pcd)
#            vis.poll_events()
#            vis.update_renderer()
#            vis.remove_geometry(pcd)
#    
#    finally:
#        stopwatch.summary()
#        base_camera.close()
#        wrist_camera.close()
#        cv.destroyAllWindows()
#        vis.destroy_window()


    from envs.utils.camera_utils import pcl_from_obs
    from common_utils import Stopwatch
    import cv2 as cv
    base1_camera = RealSenseCamera("247122072471", use_depth=1)
    base2_camera = RealSenseCamera("247122073666", use_depth=1)
    wrist_camera = KinovaCamera()

    stopwatch = Stopwatch()
    try:
        while True:
            with stopwatch.time('base1_image'):
                base1_image, base1_depth = base1_camera.get_rgbd()
            with stopwatch.time('base2_image'):
                base2_image, base2_depth = base2_camera.get_rgbd()
            with stopwatch.time('wrist_image'):
                wrist_image = wrist_camera.get_image()
            cv.imshow('base1_image', cv.cvtColor(base1_image, cv.COLOR_RGB2BGR))
            cv.imshow('base2_image', cv.cvtColor(base2_image, cv.COLOR_RGB2BGR))
            cv.imshow('wrist_image', cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
            key = cv.waitKey(1)

    finally:
        stopwatch.summary()
        base1_camera.close()
        base2_camera.close()
        wrist_camera.close()
        cv.destroyAllWindows()
