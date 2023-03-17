import torch
from torchvision.utils import _log_api_usage_once

import cv2
import math
import numpy as np
from typing import List, Iterable, Tuple


def rotate_point(origin, point, angle):
    xo, yo = origin
    xp, yp = point

    x_final = xo + math.cos(math.radians(angle)) * (xp - xo) - math.sin(math.radians(angle)) * (yp - yo)
    y_final = yo + math.sin(math.radians(angle)) * (xp - xo) + math.cos(math.radians(angle)) * (yp - yo)

    return x_final, y_final


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def translate_image(image, translate):
    y, x = image.shape[0], image.shape[1]

    xp = x * translate[0]
    yp = y * translate[1]

    M = np.float32([[1, 0, xp], [0, 1, yp]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted


class MyRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        super().__init__()

        _log_api_usage_once(self)
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        self.p = p

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"]

            # Image
            # new_image = F.vflip(image)
            new_image = np.fliplr(image)

            # Keypoints
            width_img = image.shape[0]
            old_keypoints = keypoints
            new_keypoints = np.copy(keypoints)

            # left eye center x, y
            new_keypoints[0] = width_img - old_keypoints[2]
            new_keypoints[1] = old_keypoints[3]
            # right eye center x, y
            new_keypoints[2] = width_img - old_keypoints[0]
            new_keypoints[3] = old_keypoints[1]
            # left eye inner corner x, y
            new_keypoints[4] = width_img - old_keypoints[8]
            new_keypoints[5] = old_keypoints[9]
            # left eye outer corner x, y
            new_keypoints[6] = width_img - old_keypoints[10]
            new_keypoints[7] = old_keypoints[11]
            # right eye inner corner x, y
            new_keypoints[8] = width_img - old_keypoints[4]
            new_keypoints[9] = old_keypoints[5]
            # right eye outer corner x, y
            new_keypoints[10] = width_img - old_keypoints[6]
            new_keypoints[11] = old_keypoints[7]
            # left eyebrow inner end x, y
            new_keypoints[12] = width_img - old_keypoints[16]
            new_keypoints[13] = old_keypoints[17]
            # left eyebrow outer end x, y
            new_keypoints[14] = width_img - old_keypoints[18]
            new_keypoints[15] = old_keypoints[19]
            # right eyebrow inner end x, y
            new_keypoints[16] = width_img - old_keypoints[12]
            new_keypoints[17] = old_keypoints[13]
            # right eyebrow outer end x, y
            new_keypoints[18] = width_img - old_keypoints[14]
            new_keypoints[19] = old_keypoints[15]
            # nose tip x, y
            new_keypoints[20] = width_img - old_keypoints[20]
            # new_keypoints[21] = old_keypoints[21]
            # mouth left corner x, y
            new_keypoints[22] = width_img - old_keypoints[24]
            new_keypoints[23] = old_keypoints[25]
            # mouth right corner x, y
            new_keypoints[24] = width_img - old_keypoints[22]
            new_keypoints[25] = old_keypoints[23]
            # mouth center top lip x, y
            new_keypoints[26] = width_img - old_keypoints[26]
            # new_keypoints[27] = old_keypoints[27]
            # mouth center bottom lip x, y
            new_keypoints[28] = width_img - old_keypoints[28]
            # new_keypoints[29] = old_keypoints[29]

            return {"image": new_image, "keypoints": new_keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class MyRandomRotation(object):
    def __init__(self, angle: int, p=0.5):
        super().__init__()

        _log_api_usage_once(self)
        if not (isinstance(p, float) and (0.0 <= p <= 1.0)):
            raise ValueError("probability should be float between 0 and 1")
        self.p = p
        self.angle = angle

    def __call__(self, img_with_keypoints):
        if torch.rand(1) < self.p:
            angle = np.random.randint(-self.angle, self.angle)
            # angle = np.random.triangular(-self.angles, 0, self.angles)
            image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"]

            # Image rotation
            # new_image = F.vflip(image)
            new_image = rotate_image(image, -angle)

            # Keypoints rotation
            width, height = image.shape[0], image.shape[1]
            origin = (width / 2, height / 2)
            new_keypoints = np.copy(keypoints)
            for i, point in enumerate(keypoints.reshape(15, 2)):
                new_point = rotate_point(origin, point, angle)
                new_keypoints[i * 2] = new_point[0]
                new_keypoints[i * 2 + 1] = new_point[1]

            return {"image": new_image, "keypoints": new_keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class MyRandomTranslation(object):
    # def __init__(self, translate: (float, float), p=0.5):
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
            image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"]

            height, width = image.shape[0], image.shape[1]

            # x_translate_rate = np.random.triangular(-self.translate[0], 0, self.translate[0])
            # y_translate_rate = np.random.triangular(-self.translate[1], 0, self.translate[1])
            x_translate_rate = np.random.uniform(low=-self.translate[0], high=self.translate[0])
            y_translate_rate = np.random.uniform(low=-self.translate[1], high=self.translate[1])
            x_translate_pixel = width * x_translate_rate
            y_translate_pixel = height * y_translate_rate

            # Image rotation
            M = np.float32([[1, 0, x_translate_pixel], [0, 1, y_translate_pixel]])
            shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # Keypoints rotation
            new_keypoints = np.copy(keypoints)
            for i in range(len(keypoints) // 2):
                new_keypoints[2 * i] = keypoints[2 * i] + x_translate_pixel
                new_keypoints[2 * i + 1] = keypoints[2 * i + 1] + y_translate_pixel

            return {"image": shifted, "keypoints": new_keypoints}

        return img_with_keypoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class MyRandomBrightnessAdjust(object):
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


class MyToTensor(object):
    def __call__(self, img_with_keypoints):
        image, keypoints = img_with_keypoints["image"], img_with_keypoints["keypoints"]
        image = np.transpose(image, (2, 0, 1)).copy()

        image = torch.from_numpy(image).type(torch.FloatTensor)
        keypoints = torch.from_numpy(keypoints).type(torch.FloatTensor)

        return {"image": image, "keypoints": keypoints}
