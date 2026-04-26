import importlib.util
import os

import cv2
import mujoco.viewer


def _load_wrapper_class():
    """Load GelSightSimWrapper from 1111.py (non-identifier module name)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(current_dir, "1111.py")
    spec = importlib.util.spec_from_file_location("gelsight_demo_1111", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {source_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "GelSightSimWrapper"):
        raise AttributeError("1111.py 中未找到 GelSightSimWrapper")
    return module.GelSightSimWrapper


def _nothing(_):
    pass


def main():
    GelSightSimWrapper = _load_wrapper_class()

    xml_path = "/home/hency/rmm-sim/mj_assets/stanford_tidybot2/tidybot_cube.xml"
    print(f">>> 正在加载最终模型: {xml_path}")

    sim = GelSightSimWrapper(xml_path=xml_path)
    print(f">>> GelSight 传感器加载成功！cam_name={sim.sensor_cam_name}")

    cv2.namedWindow("Control Panel")
    cv2.createTrackbar("X", "Control Panel", 50, 100, _nothing)
    cv2.createTrackbar("Y", "Control Panel", 50, 100, _nothing)
    cv2.createTrackbar("Z", "Control Panel", 50, 100, _nothing)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            val_x = cv2.getTrackbarPos("X", "Control Panel")
            val_y = cv2.getTrackbarPos("Y", "Control Panel")
            val_z = cv2.getTrackbarPos("Z", "Control Panel")

            ctrl_xyz = [
                (val_x - 50) / 500.0,
                (val_y - 50) / 500.0,
                (val_z - 50) / 500.0,
            ]

            raw_img, processed_img = sim.step(ctrl_xyz=ctrl_xyz)

            if processed_img is not None:
                cv2.imshow("GelSight Markers (Deformed)", processed_img)
            if raw_img is not None:
                cv2.imshow("GelSight Simulation", cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            viewer.sync()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
