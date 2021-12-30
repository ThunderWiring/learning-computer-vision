
import abc
import cv2 as cv
import torch
import numpy as np
from features.image_features import get_numpy_image


class ImageTransform:
    def __call__(self, image_tensor):
        return self.transform(image_tensor)

    def to_tensor(self, numpy_img):
        img = torch.transpose(numpy_img, 0, 1)
        return torch.transpose(img, 1, 2)

    @abc.abstractclassmethod
    def transform(self, image_tensor):
        pass


class WarpTransform(ImageTransform):
    def transform(self, img_tensor):
        img_tensor = get_numpy_image(img_tensor)
        h, w = img_tensor.shape[0], img_tensor.shape[1]
        input_pts = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        output_pts = np.float32([[100, 1050], [0, h], [w, 100], [w, h]])
        M = cv.getPerspectiveTransform(input_pts, output_pts)
        return self.to_tensor(
            cv.warpPerspective(img_tensor, M, (w, h), flags=cv.INTER_LINEAR))


class BlurTransform(ImageTransform):
    def transform(self, img_tensor):
        img_tensor = get_numpy_image(img_tensor)
        # the kernel size can be passed to the __init__
        return cv.blur(img_tensor, (3, 3))


class NoiseTransform(ImageTransform):
    def noise(img, snr=0.9):
        # snr = signal to noise ratio is a number in [0, 1]
        h = img.shape[0]
        w = img.shape[1]
        img1 = img.copy()
        noise_points = int(h*w*(1-snr))
        for i in range(noise_points):
            randx = np.random.randint(1, h-1)
            randy = np.random.randint(1, w-1)
            if np.random.random() <= 0.5:
                img1[randx, randy] = 0
            else:
                img1[randx, randy] = 255
        return img1

    def transform(self, img_tensor):
        # could support multiple nosise types and pass identifier to c'tor
        img_tensor = get_numpy_image(img_tensor)
        return self.to_tensor(self.noise(img_tensor))
