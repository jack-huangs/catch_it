from datetime import datetime
from typing import Tuple, Union

import cv2
import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter

from sensors import Camera

import os

class GelSightMini(Camera):
    """
    GelSightMini class for handling GelSight tactile mini sensor data.
    """

    def __init__(self, args, model: mj.MjModel, data: mj.MjData, cam_name: str) -> None:
        """
        Constructor method for GelSightMini class.

        Parameters
        ------------
        - args: Command-line arguments.
        - model (mj.MjModel): MuJoCo model instance.
        - data (mj.MjData): MuJoCo data instance.
        - cam_name (str): Name of the camera.

        Returns
        ------------
        - None.
        """
        self._args = args
        self._cam_name = cam_name
        self._model = model
        self._data = data
        self.cam = Camera(
            args=args, model=self._model, data=self._data, cam_name=self._cam_name
        )
        self._save_dir = self.cam._save_dir
        self._renderer = self.cam._renderer

        bg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "background.png",
        )
        self._background_img = cv2.imread(bg_path)
        if self._background_img is None:
            # Fallback so downstream rendering does not crash if asset is missing.
            self._background_img = np.ones(
                (self.cam.height, self.cam.width, 3), dtype=np.uint8
            ) * 127

        # constants from: https://github.com/danfergo/gelsight_simulation/tree/master
        self._min_depth = (
            0.026  # distance from the image sensor to the rigid glass outer surface
        )
        self._ELASTOMER_THICKNESS = 0.004  # m
        self._kernel_1_sigma = 7
        self._kernel_1_kernel_size = 21
        self._kernel_2_sigma = 9
        self._kernel_2_kernel_size = 52
        self._Ka = 0.8
        self._Ks = None
        self._Kd = None
        self._t = 3
        self._texture_sigma = 0.00001
        self._px2m_ratio = 5.4347826087e-05

        self._default_ks = 0.15
        self._default_kd = 0.5
        self._default_alpha = 5

        self._max_depth = self._min_depth + self._ELASTOMER_THICKNESS
        self._light_sources = [
            {
                "position": [0, 1, 0.25],
                "color": (255, 255, 255),
                "kd": 0.6,
                "ks": 0.5,
            },  # white, top
            {
                "position": [-1, 0, 0.25],
                "color": (255, 130, 115),
                "kd": 0.5,
                "ks": 0.3,
            },  # blue, right
            {
                "position": [0, -1, 0.25],
                "color": (108, 82, 255),
                "kd": 0.6,
                "ks": 0.4,
            },  # red, bottom
            {
                "position": [1, 0, 0.25],
                "color": (120, 255, 153),
                "kd": 0.1,
                "ks": 0.1,
            },  # green, left
        ]

    @property
    def name(self) -> str:
        """
        Property method to get the name of the GelSight sensor.

        Returns
        ------------
        - str: Name of the GelSight sensor.
        """
        return self._cam_name

    @property
    def tactile_image(self) -> np.ndarray:
        """
        Property method to get the GelSight image data.

        Returns
        ------------
        - numpy.ndarray: GelSight image data.
        """
        return self._generate_gelsight_img(self.depth_image, return_depth=False)

    @property
    def tactile_depth_image(self) -> np.ndarray:
        _, depth = self._generate_gelsight_img(self.depth_image, return_depth=True)
        return depth

    def _show_normalized_img(self, name: str, img: np.ndarray) -> np.ndarray:
        """
        Display a normalized image using OpenCV.

        Parameters
        ------------
        - name (str): The name of the window.
        - img (numpy.ndarray): The input image.

        Returns
        ------------
        - numpy.ndarray: The normalized image.
        """
        draw = img.copy()
        draw -= np.min(draw)
        draw = draw / np.max(draw)
        cv2.imshow(name, draw)
        return draw

    def _gkern2D(self, kernlen: int = 21, nsig: int = 3) -> np.ndarray:
        """
        Returns a 2D Gaussian kernel array.

        Parameters
        ------------
        - kernlen (int): Size of the kernel (default is 21).
        - nsig (float): Standard deviation of the Gaussian distribution (default is 3).

        Returns
        ------------
        - numpy.ndarray: 2D Gaussian kernel.
        """
        inp = np.zeros((kernlen, kernlen))
        inp[kernlen // 2, kernlen // 2] = 1
        return gaussian_filter(inp, nsig)

    def _gauss_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Add Gaussian noise to an image.

        Parameters
        ------------
        - image (numpy.ndarray): Input image.
        - sigma (float): Standard deviation of the Gaussian distribution.

        Returns
        ------------
        - numpy.ndarray: Noisy image.
        """
        row, col = image.shape
        mean = 0
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy

    def _derivative(self, mat: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of an image along the specified direction.

        Parameters
        ------------
        - mat (numpy.ndarray): Input image.
        - direction (str): Derivative direction ('x' or 'y').

        Returns
        ------------
        - numpy.ndarray: Image derivative.
        """
        assert (
            direction == "x" or direction == "y"
        ), "The derivative direction must be 'x' or 'y'"
        kernel = None
        if direction == "x":
            kernel = [[-1.0, 0.0, 1.0]]
        elif direction == "y":
            kernel = [[-1.0], [0.0], [1.0]]
        kernel = np.array(kernel, dtype=np.float64)
        return cv2.filter2D(mat, -1, kernel) / 2.0

    def _tangent(self, mat: np.ndarray) -> np.ndarray:
        """
        Compute the tangent vector of an image.

        Parameters
        ------------
        - mat (numpy.ndarray): Input image.

        Returns
        ------------
        - numpy.ndarray: Tangent vector of the image.
        """
        dx = self._derivative(mat, "x")
        dy = self._derivative(mat, "y")
        img_shape = np.shape(mat)
        _1 = (
            np.repeat([1.0], img_shape[0] * img_shape[1])
            .reshape(img_shape)
            .astype(dx.dtype)
        )
        unnormalized = cv2.merge((-dx, -dy, _1))
        norms = np.linalg.norm(unnormalized, axis=2)
        return unnormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2)

    def _solid_color_img(self, color: Tuple, size: Tuple) -> np.ndarray:
        """
        Create a solid color image.

        Parameters
        ------------
        - color (tuple): RGB color tuple.
        - size (tuple): Size of the image (height, width).

        Returns
        ------------
        - numpy.ndarray: Solid color image.
        """
        image = np.zeros(size + (3,), np.float64)
        image[:] = color
        return image

    def _add_overlay(
        self, rgb: np.ndarray, alpha: np.ndarray, color: Tuple
    ) -> np.ndarray:
        """
        Add an overlay to an RGB image.

        Parameters
        ------------
        - rgb (numpy.ndarray): RGB image.
        - alpha (numpy.ndarray): Alpha channel for the overlay.
        - color (tuple): RGB color tuple for the overlay.

        Returns
        ------------
        - numpy.ndarray: Image with overlay.
        """
        s = np.shape(alpha)
        opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))
        overlay = self._solid_color_img(color, s)
        foreground = opacity3 * overlay
        background = (1.0 - opacity3) * rgb.astype(np.float64)
        res = background + foreground
        res[res > 255.0] = 255.0
        res[res < 0.0] = 0.0
        res = res.astype(np.uint8)
        return res

    def _generate_gelsight_img(
        self, obj_depth: np.ndarray, return_depth: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate GelSight image based on the provided depth information.

        Parameters
        ------------
        - obj_depth: Depth information of the object.
        - return_depth (bool): Whether to return the elastomer depth or not.

        Returns
        ------------
        - GelSight image or tuple (GelSight image, elastomer depth) if return_depth is True.
        """
         # --- 建议插入这段自动缩放代码 ---
        h, w = obj_depth.shape[:2]
        if self._background_img.shape[:2] != (h, w):
            self._background_img = cv2.resize(self._background_img, (w, h))

        not_in_touch, in_touch = self._segments(obj_depth)
        protrusion_depth = self._protrusion_map(obj_depth, not_in_touch)
        elastomer_depth = self._apply_elastic_deformation(
            protrusion_depth, not_in_touch, in_touch
        )

        textured_elastomer_depth = self._gauss_noise(
            elastomer_depth, self._texture_sigma
        )

        out = self._Ka * self._background_img
        out = self._add_overlay(
            out, self._internal_shadow(protrusion_depth), (0.0, 0.0, 0.0)
        )

        T = self._tangent(textured_elastomer_depth / self._px2m_ratio)
        # show_normalized_img('tangent', T)
        for light in self._light_sources:
            ks = light["ks"] if "ks" in light else self._default_ks
            kd = light["kd"] if "kd" in light else self._default_kd
            alpha = light["alpha"] if "alpha" in light else self._default_alpha
            out = self._add_overlay(
                out,
                self._phong_illumination(T, light["position"], kd, ks, alpha),
                light["color"],
            )

        kernel = self._gkern2D(3, 1)
        out = cv2.filter2D(out, -1, kernel)

        if return_depth:
            return out, elastomer_depth
        return out

    def _phong_illumination(
        self, T: np.ndarray, source_dir: np.ndarray, kd: float, ks: float, alpha: float
    ) -> np.ndarray:
        """
        Apply Phong illumination model to calculate the reflected light intensity.

        The Phong reflection model combines diffuse and specular reflection components.

        Parameters
        ------------
        - T (numpy.ndarray): Surface normals of the object.
        - source_dir (numpy.ndarray): Direction vector of the light source.
        - kd (float): Diffuse reflection coefficient.
        - ks (float): Specular reflection coefficient.
        - alpha (float): Shininess or specular exponent.

        Returns
        ------------
        - numpy.ndarray: Resultant illumination intensity for each pixel.
        """
        # Calculate the dot product between surface normals and light source direction
        dot = np.dot(T, np.array(source_dir)).astype(np.float64)

        # Compute diffuse reflection component
        difuse_l = dot * kd
        difuse_l[difuse_l < 0] = 0.0  # Ensure non-negative values

        # Compute reflection vector R and the view vector V
        dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)
        R = 2.0 * dot3 * T - source_dir
        V = [0.0, 0.0, 1.0]

        # Compute specular reflection component
        spec_l = np.power(np.dot(R, V), alpha) * ks

        # Combine diffuse and specular components to get the final illumination
        return difuse_l + spec_l

    def _apply_elastic_deformation_gauss(
        self,
        protrusion_depth: np.ndarray,
        not_in_touch: np.ndarray,
        in_touch: np.ndarray,
    ) -> np.ndarray:
        """
        Apply elastic deformation to an input depth map.

        Parameters
        ------------
        - protrusion_depth (numpy.ndarray): Input depth map.
        - not_in_touch (numpy.ndarray): Mask indicating areas not in touch.
        - in_touch (numpy.ndarray): Mask indicating areas in touch.

        Returns
        ------------
        - numpy.ndarray: Deformed depth map.
        """
        kernel = self._gkern2D(15, 7)
        deformation = self._max_depth - protrusion_depth

        for i in range(5):
            deformation = cv2.filter2D(deformation, -1, kernel)

        return 30 * -deformation * not_in_touch + (protrusion_depth * in_touch)

    def _apply_elastic_deformation(
        self,
        protrusion_depth: np.ndarray,
        not_in_touch: np.ndarray,
        in_touch: np.ndarray,
    ) -> np.ndarray:
        """
        Apply a more complex version of elastic deformation to an input depth map.

        Parameters
        ------------
        - protrusion_depth (numpy.ndarray): Input depth map.
        - not_in_touch (numpy.ndarray): Mask indicating areas not in touch.
        - in_touch (numpy.ndarray): Mask indicating areas in touch.

        Returns
        ------------
        - numpy.ndarray: Deformed depth map.
        """

        protrusion_depth = -(protrusion_depth - self._max_depth)
        kernel = self._gkern2D(self._kernel_1_kernel_size, self._kernel_1_sigma)
        deformation = protrusion_depth

        deformation2 = protrusion_depth
        kernel2 = self._gkern2D(self._kernel_2_kernel_size, self._kernel_2_sigma)

        for i in range(self._t):
            deformation_ = cv2.filter2D(deformation, -1, kernel)
            r = (
                np.max(protrusion_depth) / np.max(deformation_)
                if np.max(deformation_) > 0
                else 1
            )
            deformation = np.maximum(r * deformation_, protrusion_depth)

            deformation2_ = cv2.filter2D(deformation2, -1, kernel2)
            r = (
                np.max(protrusion_depth) / np.max(deformation2_)
                if np.max(deformation2_) > 0
                else 1
            )
            deformation2 = np.maximum(r * deformation2_, protrusion_depth)

        deformation_v1 = self._apply_elastic_deformation_gauss(
            protrusion_depth, not_in_touch, in_touch
        )

        for i in range(self._t):
            deformation_ = cv2.filter2D(deformation2, -1, kernel)
            r = (
                np.max(protrusion_depth) / np.max(deformation_)
                if np.max(deformation_) > 0
                else 1
            )
            deformation2 = np.maximum(r * deformation_, protrusion_depth)

        deformation_x = 2 * deformation - deformation2

        return self._max_depth - deformation_x

    def _protrusion_map(
        self, original: np.ndarray, not_in_touch: np.ndarray
    ) -> np.ndarray:
        """
        Generate a protrusion map based on the original depth map and regions that are not in touch.

        Parameters
        ------------
        - original (numpy.ndarray): Original depth map.
        - not_in_touch (numpy.ndarray): Binary map indicating regions not in touch.

        Returns
        ------------
        - numpy.ndarray: Protrusion map where values exceeding `max_depth` are clamped to `max_depth`.
        """
        protrusion_map = np.copy(original)
        # OBS! This line caused the non-register issue
        # protrusion_map[not_in_touch >= self._max_depth] = self._max_depth
        return protrusion_map

    def _segments(self, depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment the depth map into regions in touch and regions not in touch based on the max depth.

        Parameters
        ------------
        - depth_map (numpy.ndarray): Depth map.

        Returns
        ------------
        - Tuple[numpy.ndarray, numpy.ndarray]: Two binary maps representing regions not in touch and in touch, respectively.
        """
        # 0.03 vs 0.025791787
        not_in_touch = np.copy(depth_map)
        not_in_touch[not_in_touch < self._max_depth] = 0.0
        not_in_touch[not_in_touch >= self._max_depth] = 1.0

        in_touch = 1 - not_in_touch

        return not_in_touch, in_touch

    def _internal_shadow(self, elastomer_depth: np.ndarray) -> np.ndarray:
        """
        Generate an internal shadow map based on the elastomer depth.

        Parameters
        ------------
        - elastomer_depth (numpy.ndarray): Elastomer depth map.

        Returns
        ------------
        - numpy.ndarray: Internal shadow map indicating regions of the elastomer that are shadowed.
        """
        elastomer_depth_inv = self._max_depth - elastomer_depth
        elastomer_depth_inv = np.interp(
            elastomer_depth_inv, (0, self._ELASTOMER_THICKNESS), (0.0, 1.0)
        )
        return elastomer_depth_inv

    def shoot(self) -> None:
        """
        Captures a new rgb image, depth image and point cloud from the camera.

        Returns
        ------------
        - None.
        """

        self.cam.shoot()
        self.save()

    def save(self, img_name: str = "") -> None:
        """Saves the captured image and depth information.

        Args
        ------------
        - img_name: Name for the saved image file.
        """
        print(f"saving rgb image, depth image and point cloud to {self.save_dir}")
        if img_name == "":
            timestamp = datetime.now()
            print(self._save_dir + f"{timestamp}_tac_rgb.png")
            cv2.imwrite(
                self._save_dir + f"{timestamp}_tac_rgb.png",
                cv2.cvtColor(self.tactile_image, cv2.COLOR_RGB2BGR),
            )
            np.save(
                self._save_dir + f"{timestamp}_tac_depth.npy", self.tactile_depth_image
            )

        else:
            cv2.imwrite(
                self._save_dir + f"{img_name}_tac_rgb.png",
                cv2.cvtColor(self.tactile_image, cv2.COLOR_RGB2BGR),
            )
            np.save(
                self._save_dir + f"{img_name}_tac_depth.npy", self.tactile_depth_image
            )
