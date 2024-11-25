import cv2 as cv
import os
from .logging_config import logger


def load_images_dir(image_dir: str):
    '''Load images from directory, returns array of images(numpy array)'''
    logger.debug(f"Loading image from directory {image_dir}")
    filenames = os.listdir(image_dir)
    images = []
    for filename in filenames:
        image = load_images_dir(filename)
        if image:
            images.append(image)
    return images


def load_image(image_path: str):
    '''Load image from image path, returns image numpy array'''
    logger.debug(f"Loading image from directory {image_path}")
    if os.path.isfile(image_path):
        try:
            image = cv.imread(image_path)
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            return image_rgb
        except IOError:
            logger.warning(f"Cannot open image file {image}")
    else:
        logger.warning(f"Cannot recognize image file inside {image_path}")
