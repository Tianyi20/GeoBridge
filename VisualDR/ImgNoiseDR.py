import cv2
import numpy as np


class ImgNoiseDR(object):
    """Image noise / post-processing domain randomization."""

    def __init__(self, bullet_client, seed=42):
        self.bullet_client = bullet_client
        self.ImgNoise_rng = np.random.default_rng(seed)
        self.image_noise_cfg = None

    def reset(self):
        self.image_noise_cfg = None

    def sample_image_noise_randomization(self):
        rng = self.ImgNoise_rng
        self.image_noise_cfg = {
            "brightness": float(rng.uniform(-12.0, 12.0)),
            "contrast": float(rng.uniform(0.85, 1.15)),
            "gaussian_std": float(rng.uniform(0.0, 6.0)),
            "salt_pepper_prob": float(rng.uniform(0.0, 0.004)),
            "blur_prob": float(rng.uniform(0.0, 0.25)),
        }
        return self.image_noise_cfg

    def apply_image_noise(self, rgb):
        if self.image_noise_cfg is None:
            return rgb

        cfg = self.image_noise_cfg
        rng = self.ImgNoise_rng
        img = rgb.astype(np.float32)

        img = img * cfg["contrast"] + cfg["brightness"]

        gaussian_std = cfg["gaussian_std"]
        if gaussian_std > 0:
            img += rng.normal(0.0, gaussian_std, size=img.shape)

        img = np.clip(img, 0, 255).astype(np.uint8)

        if rng.random() < cfg["blur_prob"]:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        sp_prob = cfg["salt_pepper_prob"]
        if sp_prob > 0:
            mask = rng.random(img.shape[:2])
            img[mask < sp_prob / 2.0] = 0
            img[(mask >= sp_prob / 2.0) & (mask < sp_prob)] = 255

        return img
