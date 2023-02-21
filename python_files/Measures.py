#import libs
from math import log10, sqrt
import numpy as np
from tensorflow.image import psnr, ssim

class InvalidImage(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class NoImageExcepiton(Exception):
    def __init__(self, message):
        super().__init__(message)

class ImageShapeException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class Measures():
    def __init__(self):
        self.PSNR_val = None
        self.SSIM_val  = None
        self.NAE_val = None
        self.SC_val  = None
    def __error_handler(self, original_image,reconstructed_image):
        if original_image is None or reconstructed_image is None:
            raise NoImageExcepiton("Input image is None")
        try:
            original_image_shape = original_image.shape
            reconstructed_image_shape = reconstructed_image.shape
        except Exception:
            raise InvalidImage("Invalid input data")
        if original_image_shape != reconstructed_image_shape:
            raise ImageShapeException("Images shapes are not equal")

    def __metrics(self, original_image, reconstructed_image):
        self.__error_handler(original_image, reconstructed_image)
        self.PSNR_val = self.PSNR(original_image, reconstructed_image)
        self.SSIM_val = self.SSIM(original_image, reconstructed_image)
        self.NAE_val  = self.NAE(original_image, reconstructed_image)
        self.SC_val   = self.SC(original_image, reconstructed_image)

    def PSNR(self, original_image, reconstructed_image):
        return float(psnr(original_image, reconstructed_image, max_val=255.))

    def SSIM(self, original_image, reconstructed_image):
        return float(ssim(original_image, reconstructed_image, max_val=255., filter_size=11,\
                                        filter_sigma=1.5, k1=0.01, k2=0.03))

    def NAE(self, original_image, reconstructed_image):
        return np.sum(np.abs(original_image-reconstructed_image))/np.sum(original_image)

    def SC(self, original_image, reconstructed_image):
        return np.sum(original_image**2)/np.sum(reconstructed_image**2)

    def make_metrics(self, *args):
        self.__metrics(*args)
        return (self.PSNR_val, self.SSIM_val, self.NAE_val, self.SC_val)
