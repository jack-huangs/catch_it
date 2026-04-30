This directory stores calibration data for the real robot camera setup.

Current assumption:
- The real `base2` camera is mounted to match the model camera `base` in
  [assets/urdf/stanford_tidybot2/tidybot.xml](/home/jack/桌面/catch_it/assets/urdf/stanford_tidybot2/tidybot.xml:137).
- The transform convention matches
  [camera_utils.py](/home/jack/桌面/catch_it/mujoco_command/envs/utils/camera_utils.py:10):
  `T_camera_to_robot`, meaning points are first expressed in the camera frame
  and then transformed into the robot frame.

Files:
- `base2_extrinsics_model.json`
  Model-based camera-to-robot extrinsics derived from `tidybot.xml`.
- `base2_intrinsics_template.json`
  Empty template for the real D455 intrinsics that should be filled from the
  device.
- `base2_intrinsics_sim_fallback.json`
  Temporary intrinsics approximated from the MuJoCo camera `fovy=65`,
  `resolution=640x360`. This is only a fallback for early testing.

Notes:
- The model camera position is `(-0.089, -0.214, 1.600)` meters in `base_link`.
- The model camera is aimed at `(1.2, 0.0, 1.0)` meters using `targetbody`.
- The extrinsics file here converts that look-at definition into a numeric
  rotation matrix.
- For real deployment, keep this extrinsics file if you physically mount the
  camera to match the model. Replace the intrinsics with values read from the
  real D455.
