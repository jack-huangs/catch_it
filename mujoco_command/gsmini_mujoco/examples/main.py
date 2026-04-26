import sys
import os
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sensors.gelsight_mini.gelsight_mini import GelSightMini
from sensors.gelsight_mini.tactile_contour import TactileContourRecognizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
os.chdir(project_root)

class GelSightSimNode:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.xml_path = os.path.join(project_root, "assets", "gelsight_mini", "grasp.xml")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        # sensor init
        self.args = SimpleNamespace(cam_width=320, cam_height=240, save_dir="./saved_images/")
        self.sensor = GelSightMini(self.args, self.model, self.data, cam_name="gelsight_mini")
        # create windows
        cv2.namedWindow("Control Panel")
        cv2.createTrackbar("X", "Control Panel", 50, 100, lambda x: None)
        cv2.createTrackbar("Y", "Control Panel", 50, 100, lambda x: None)
        cv2.createTrackbar("Z", "Control Panel", 50, 100, lambda x: None)

        self.marker_base_img = None 
        bg_path = os.path.join(project_root, "sensors", "gelsight_mini", "assets", "background.png")
        self.tactile_recognizer = TactileContourRecognizer(bg_image_path=bg_path)

    def generate_background(self, width, height, rows=7, cols=9, radius=3):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        bg_path = os.path.join(project_root, "sensors", "gelsight_mini", "assets", "background.png")

        # reload image
        if os.path.exists(bg_path):
            bg = cv2.imread(bg_path)

            if bg is None:
                bg = np.ones((height, width, 3), dtype=np.uint8) * 255
            else:
                bg = cv2.resize(bg, (width, height))
            
        else:
            bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # print mark
        xs = np.linspace(0, width, cols + 2)[1:-1].astype(int)
        ys = np.linspace(0, height, rows + 2)[1:-1].astype(int)

        # add black pot
        for x in xs:
            for y in ys:
                cv2.circle(bg, (x, y), radius, (30, 30, 30), -1)

        return bg
    
    def _update_control(self):
        val_x = cv2.getTrackbarPos("X", "Control Panel")
        val_y = cv2.getTrackbarPos("Y", "Control Panel")
        val_z = cv2.getTrackbarPos("Z", "Control Panel")

        self.data.ctrl[0] = (val_x - 50) / 500.0 
        self.data.ctrl[1] = (val_y - 50) / 500.0
        self.data.ctrl[2] = (val_z - 50) / 500.0

    def _update_vision(self):
        # ready for sensor
        if self.sensor is None or self.sensor.depth_image is None:
            return

        rgb_img, depth_img = self.sensor._generate_gelsight_img(self.sensor.depth_image, return_depth=True)

        # marker distortion + tactile contour recognition
        if depth_img is not None:
            result = self.tactile_recognizer.process(depth_img)
            if result is not None:
                cv2.imshow("GelSight Markers (Deformed)", result["distorted_markers"])
                cv2.imshow("GelSight Contact Contours", result["contour_overlay"])

        # RGB imshow
        if rgb_img is not None:
            show_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("GelSight Simulation", show_rgb)

    def run(self):
        # open mujoco
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                self._update_control()
                self._update_vision()
                viewer.sync()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sim = GelSightSimNode()
    sim.run()
