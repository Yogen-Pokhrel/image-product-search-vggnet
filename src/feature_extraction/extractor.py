import cv2
import numpy as np
from keras.applications import VGG16
from keras.models import Model



def load_model():
  '''Load the vgg16 model from local directory or download it'''
  base_model = VGG16(weights='imagenet', include_top=False)
  model = Model(inputs=base_model.input, outputs = base_model.get_layer("block5_pool").output)
  return model

def embedding_extraction(image):
  '''Extract embedding of the image from vgg16'''
  if image is None:
    raise "image not loaded"
  model = load_model()
  image  = cv2.resize(image, (224,224))
  image_expand = np.expand_dims(image, axis=0)
  embedding = model.predict(image_expand)
  return embedding
