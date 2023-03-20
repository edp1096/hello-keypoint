from config import *

import torch
from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

import cv2
import numpy as np
import math
from typing import List, Iterable, Tuple
from PIL import Image


class CLAHE(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)

        return img

    def __repr__(self):
        return self.__class__.__name__ + "(clipLimit={0}, tileGridSize={1})".format(self.clipLimit, self.tileGridSize)


class PilToNumpy(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return {"image": np.array(data["image"]) / 255.0, "keypoints": data["keypoints"]}

    def __repr__(self):
        return self.__class__.__name__ + "()"


class KeypointImageToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return {"image": F.to_tensor(data["image"].copy()), "keypoints": data["keypoints"]}

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ResizeKeypointAndImage(object):
    def __init__(self, size: Tuple[int, int]):
        super().__init__()

        self.size = size

    def __call__(self, data):
        return_dict = {"image": cv2.resize(data["image"], self.size), "keypoints": data["keypoints"].copy()}

        width_img, height_img = np.shape(data["image"])[0], np.shape(data["image"])[1]
        width_new, height_new = self.size[0], self.size[1]
        return_dict["keypoints"][0::2] = return_dict["keypoints"][0::2] * width_new / width_img
        return_dict["keypoints"][1::2] = return_dict["keypoints"][1::2] * height_new / height_img

        return return_dict

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        super().__init__()

        _log_api_usage_once(self)
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError(f"probability should be float between 0 and 1 (got {p})")
        self.p = p

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            new_image = np.flip(img_with_keypoints["image"], axis=1)

            width_img = np.shape(img_with_keypoints["image"])[0]
            old_keypoints = np.copy(img_with_keypoints["keypoints"])
            new_keypoints = np.zeros(POINT_NUM)

            new_keypoints[0], new_keypoints[1] = width_img - old_keypoints[2], old_keypoints[3]  # left eye
            new_keypoints[2], new_keypoints[3] = width_img - old_keypoints[0], old_keypoints[1]  # right eye
            if POINT_NUM == 6:
                new_keypoints[4], new_keypoints[5] = width_img - old_keypoints[4], old_keypoints[5]  # nose

            return {"image": new_image, "keypoints": new_keypoints}
        else:
            return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation(object):
    def __init__(self, angle: int, p=0.5):
        super().__init__()

        _log_api_usage_once(self)
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        self.p = p
        self.angle = angle

    def rotate_point(self, origin, point, angle):
        xo, yo = origin
        xp, yp = point

        x_final = xo + math.cos(math.radians(angle)) * (xp - xo) - math.sin(math.radians(angle)) * (yp - yo)
        y_final = yo + math.sin(math.radians(angle)) * (xp - xo) + math.cos(math.radians(angle)) * (yp - yo)

        return x_final, y_final

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return result

    def translate_image(self, image, translate):
        y, x = image.shape[0], image.shape[1]

        xp = x * translate[0]
        yp = y * translate[1]

        M = np.float32([[1, 0, xp], [0, 1, yp]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        return shifted

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            angle = np.random.randint(-self.angle, self.angle)
            image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"]

            # Image rotation
            new_image = self.rotate_image(image, -angle)

            # Keypoints rotation
            width, height = image.shape[0], image.shape[1]
            origin = (width / 2, height / 2)
            new_keypoints = np.copy(keypoints)
            point_num = int(POINT_NUM / 2)

            if POINT_NUM == 4:  # bounding box
                # Append lower left and upper right corner from new_keypoints
                new_keypoints = np.append(new_keypoints, [new_keypoints[0], new_keypoints[3], new_keypoints[2], new_keypoints[1]])
                point_num = POINT_NUM

            for i, point in enumerate(new_keypoints.reshape(point_num, 2)):
                new_point = self.rotate_point(origin, point, angle)
                new_keypoints[i * 2] = new_point[0]
                new_keypoints[i * 2 + 1] = new_point[1]

            if POINT_NUM == 4:  # bounding box
                if angle > 0:
                    new_keypoints[0] = new_keypoints[4]  # Reassign upper left x
                    new_keypoints[2] = new_keypoints[6]  # Reassign lower right x
                if angle < 0:
                    new_keypoints[1] = new_keypoints[7]  # Reassign upper left y
                    new_keypoints[3] = new_keypoints[5]  # Reassign lower right y

                if new_keypoints[0] < 0:
                    new_keypoints[0] = 0
                if new_keypoints[1] < 0:
                    new_keypoints[1] = 0
                if new_keypoints[2] > width:
                    new_keypoints[2] = width
                if new_keypoints[3] > height:
                    new_keypoints[3] = height

                new_keypoints = new_keypoints[0:4]  # Remove lower left and upper right

            return {"image": new_image, "keypoints": new_keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomTranslation(object):
    def __init__(self, translate: Tuple[float, float], p=0.5):
        """

        :type translate: (float, float) x, y translate in percent - from 0 to 1
        """
        super().__init__()
        _log_api_usage_once(self)
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        if not (len(translate) == 2 and (0.0 <= translate[0] <= 1.0) and (0.0 <= translate[1] <= 1.0)):
            raise ValueError("there should be 2 numbers in translate, both between 0 and 1")
        self.p = p
        self.translate = translate

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"].copy()

            height, width = image.shape[0], image.shape[1]

            x_translate_pixel = width * np.random.uniform(low=-self.translate[0], high=self.translate[0])
            y_translate_pixel = height * np.random.uniform(low=-self.translate[1], high=self.translate[1])

            M = np.float32([[1, 0, x_translate_pixel], [0, 1, y_translate_pixel]])
            shifted = cv2.warpAffine(image.copy(), M, (image.shape[1], image.shape[0]))

            new_keypoints = keypoints.copy()
            new_keypoints[0::2] = keypoints[0::2] + x_translate_pixel
            new_keypoints[1::2] = keypoints[1::2] + y_translate_pixel

            if POINT_NUM == 4:  # bounding box
                if new_keypoints[0] < 0:
                    new_keypoints[0] = 0
                if new_keypoints[1] < 0:
                    new_keypoints[1] = 0
                if new_keypoints[2] > width:
                    new_keypoints[2] = width
                if new_keypoints[3] > height:
                    new_keypoints[3] = height

            return {"image": shifted, "keypoints": new_keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomBrightnessAdjust(object):
    def __init__(self, brightness: float, p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        if not (0.0 <= brightness <= 1.0):
            raise ValueError("brightness should be float between 0 and 1")
        self.p = p
        self.brightness = brightness

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"]
            random_brightness = np.random.uniform(low=-self.brightness, high=self.brightness)

            # Image rotation
            hsv = cv2.cvtColor(image.astype("float32"), cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            lim = 255 - random_brightness
            v[v > lim] = 255
            v[v <= lim] += random_brightness

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            img = np.clip(img, 0.0, 1.0)

            # Keypoints rotation

            return {"image": img, "keypoints": keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomJitter(object):
    def __init__(self, jitter: float, p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        if not (0.0 <= jitter <= 1.0):
            raise ValueError("jitter should be float between 0 and 1")
        self.p = p
        self.jitter = jitter

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"]
            random_jitter = np.random.uniform(low=-self.jitter, high=self.jitter)

            hsv = cv2.cvtColor(image.astype("float32"), cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            lim = 255 - random_jitter
            v[v > lim] = 255
            v[v <= lim] += random_jitter

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            img = np.clip(img, 0.0, 1.0)

            return {"image": img, "keypoints": keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
