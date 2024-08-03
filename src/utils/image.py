import cv2 as cv
import os
from .env_config import environment_variables
from .logging_config import logger


def load_image_dir(image_dir: str):
    filenames = os.listdir(image_dir)
    images = []
    for filename in filenames:
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path):
            try:
                image = cv.imread(file_path, cv.cvtColor(cv.COLOR_BGR2RGB))
                images.append(image)
            except IOError:
                print(f"Cannot open image file {file_path}")
    return images
