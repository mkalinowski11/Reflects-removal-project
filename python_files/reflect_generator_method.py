import random
from copy import deepcopy
import cv2
import numpy as np

class NoImageExcepiton(Exception):
    def __init__(self, message):
        super().__init__(message)

class ImageShapeException(Exception):
    def __init__(self, msg):
        super().__init__(msg)
class InvalidImage(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class Rainbow_Dash_Algorithm():
    @staticmethod
    def __apply_mean_filter(image, mask):
        if image.shape != mask.shape:
            raise ImageShapeException("Image shape not equal to mask")
        if image is None or mask is None:
            raise NoImageExcepiton("Input is None")
        for i in range(2, len(image) - 2):
            for j in range(2, len(image[i]) - 2):
                if mask[i, j, 0] != 0 and mask[i, j, 1] != 0 and mask[i, j, 2] != 0:
                    for k in range(3):
                        image[i, j, k] = (
                                1 / 25. * image[i - 2, j - 2, k] + 1 / 25. * image[i - 2, j - 1, k] + 1 / 25. * image[
                            i - 2, j, k] + 1 / 25. * image[i - 2, j + 1, k] + 1 / 25. * image[i - 2, j + 2, k] +
                                1 / 25. * image[i - 1, j - 2, k] + 1 / 25. * image[i - 1, j - 1, k] + 1 / 25. * image[
                                    i - 1, j, k] + 1 / 25. * image[i - 1, j + 1, k] + 1 / 25. * image[i - 1, j + 2, k] +
                                1 / 25. * image[i, j - 2, k] + 1 / 25. * image[i, j - 1, k] + 1 / 25. * image[
                                    i, j, k] + 1 / 25. * image[i, j + 1, k] + 1 / 25. * image[i, j + 2, k] +
                                1 / 25. * image[i + 1, j - 2, k] + 1 / 25. * image[i + 1, j - 1, k] + 1 / 25. * image[
                                    i + 1, j, k] + 1 / 25. * image[i + 1, j + 1, k] + 1 / 25. * image[i + 1, j + 2, k] +
                                1 / 25. * image[i + 2, j - 2, k] + 1 / 25. * image[i + 2, j - 1, k] + 1 / 25. * image[
                                    i + 2, j, k] + 1 / 25. * image[i + 2, j + 1, k] + 1 / 25. * image[i + 2, j + 2, k]
                        )
    @staticmethod
    def apply_filter_kernel(image):
        tmp_image = image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cv2.filter2D(src=tmp_image, ddepth=-1, kernel=kernel)
        return tmp_image

    @staticmethod
    def __prepare_mask(image, max_reflects = 60, max_radius = 10):
        if image is None:
            raise NoImageExcepiton("No input image")
        image_copy = deepcopy(image)
        min_reflects = max_reflects - 20
        if min_reflects<0:
            min_reflects = 0
        number_of_reflects = random.randint(min_reflects, max_reflects)
        smaller_reflects_radius_range = 4
        if max_radius>10 and max_radius<=18:
            smaller_reflects_radius_range = 8
        elif max_radius>18:
            smaller_reflects_radius_range = 12
        for _ in range(number_of_reflects):
            random_point = (random.randint(0, image.shape[0]-1), random.randint(0, image.shape[0]-1))
            random_radius = random.randint(3, max_radius)
            random_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
            cv2.circle(image_copy, random_point, random_radius, color=random_color, thickness=-1)
            next_direction = [*list(range(-4, 8))]
            tmp_point = [random_point[0], random_point[1]]
            how_many_smaller_reflects_range = 12
            if max_radius > 18:
                how_many_smaller_reflects_range = 20
            for _ in range(random.randint(0, how_many_smaller_reflects_range)):
                tmp_point[0] += random.choice(next_direction)
                tmp_point[1] += random.choice(next_direction)
                random_color = (random.randint(220, 255), random.randint(220, 255), random.randint(200, 255))
                cv2.circle(image_copy, tuple(tmp_point), random.randint(2, smaller_reflects_radius_range),
                                                                            color=random_color, thickness=-1)
        return image_copy

    @staticmethod
    def __merge_images(original_image, reflect_mask):
        tmp_image = original_image
        for i in range(len(original_image)):
            for j in range(len(original_image[i])):
                if reflect_mask[i, j, 0]:
                    tmp_image[i, j] = reflect_mask[i, j]
        return tmp_image
    @staticmethod
    def create_border(mask):
        tmp_image = mask
        iterations = random.randint(1, 3)
        filter_size = random.choice([3, 5])
        dilated_mask = cv2.dilate(tmp_image,
                                  kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (filter_size, filter_size)),
                                  iterations=iterations)
        reflect_borders = dilated_mask - mask*255
        Rainbow_Dash_Algorithm.image_relu(reflect_borders)
        random_color = (random.randint(10, 30), random.randint(30, 80), random.randint(30, 150))
        for i in range(len(reflect_borders)):
            for j in range(len(reflect_borders[i])):
                if reflect_borders[i, j, 0] != 0 and reflect_borders[i, j, 1] != 0 and reflect_borders[i, j, 2] != 0:
                    reflect_borders[i, j] = (random_color[0] + random.randint(-10, 10),
                                             random_color[1] + random.randint(-10, 10),
                                             random_color[2] + random.randint(-10, 10))
        reflect_borders = tmp_image + reflect_borders
        return reflect_borders
    @staticmethod
    def image_relu(image):
        for rows in range(len(image)):
            for cols in range(len(image[rows])):
                for width in range(3):
                    if image[rows, cols, width] < 0:
                        image[rows, cols, width] = 0

    @staticmethod
    def prepare_image_with_reflects(image, use_mean_filter = False, create_borders=False , **kwargs):
        if image is None:
            raise NoImageExcepiton("Input image is None")
        try:
            image_shape = image.shape
        except Exception:
            raise InvalidImage("Invalid input data")
        tmp_image = deepcopy(image)
        mask = np.zeros(shape=image.shape, dtype='float32')
        mask = Rainbow_Dash_Algorithm.__prepare_mask(mask, **kwargs)
        tmp_image = Rainbow_Dash_Algorithm.__merge_images(tmp_image, mask)
        if use_mean_filter:
            Rainbow_Dash_Algorithm.__apply_mean_filter(tmp_image, mask)
        if create_borders:
            mask = Rainbow_Dash_Algorithm.create_border(mask)
            tmp_image = Rainbow_Dash_Algorithm.__merge_images(tmp_image, mask)
        return tmp_image

    @staticmethod
    def image_subtraction(original_image, reconstructed_image):
        if original_image is None or reconstructed_image is None:
            raise NoImageExcepiton("Input image is None")
        try:
            original_image_shape = original_image.shape
            reconstructed_image_shape = reconstructed_image.shape
        except Exception:
            raise InvalidImage("Invalid input data")
        if original_image_shape != reconstructed_image_shape:
            raise ImageShapeException("Input shapes not equal")
        image_subtracted = original_image - reconstructed_image
        return image_subtracted
