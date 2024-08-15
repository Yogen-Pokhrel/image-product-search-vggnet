import tensorflow as tf
import cv2 as cv
import numpy as np
from utils.logging_config import logger
from utils.image_loader import load_image

class ImageSegmentor:
    _model = None
    _model_path = None

    def __init__(self, model_path:str) -> None:
        self._model_path = model_path
        self._load_model()

    
    def remove_background(image, input_path: str, output_path: str):
        image = load_image(input_path)
        height, width = image.shape[:2]
        image_resized = cv.resize(image, (256,256))

        image_array = image_resized.astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        mask = ImageSegmentor._model.predict(image_array)[0]
        mask = (mask > 0.5).astype(np.uint8)
        
        
        # Resize mask to original image size
        mask_resized = cv.resize(mask, (width, height))
        
        # Apply mask
        result = cv.bitwise_and(image, image, mask=mask_resized.astype(np.uint8))
        
        # Save result
        cv.imwrite(output_path, result)
        print(f"Background removed and saved to {output_path}.")
        

    def _load_model(self):
        if ImageSegmentor._model is None and ImageSegmentor._model_path is not None:
            logger.debug(f"Loading model from {self._model_path}")
            ImageSegmentor._model = tf.keras.models.load_model(self._model_path)
        else:
            logger.info("Model is already loaded and cached")