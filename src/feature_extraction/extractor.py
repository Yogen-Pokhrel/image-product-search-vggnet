import cv2
import numpy as np
from keras.models import Model, load_model
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D



def load_custom_model():
  '''Load the vgg16 model from local directory or download it'''
  # base_model = VGG16(weights='imagenet', include_top=False)
  # model = Model(inputs=base_model.input, outputs = base_model.get_layer("block5_pool").output)
  classification_model  = load_model("/Users/yallahproperty/Downloads/classification_deepfasthion_v2.keras")
  embedding_model = Model(inputs=classification_model.input, outputs = classification_model.get_layer("block5_pool").output)
  # maxpool_layer = MaxPooling2D(pool_size=(2, 2), padding='valid')(embedding_model.output)
  # embedding_model_maxpool = Model(inputs=embedding_model.input, outputs=maxpool_layer)
  gap_layer = GlobalAveragePooling2D()(embedding_model.output)
  model_with_gap = Model(inputs=embedding_model.input, outputs=gap_layer)
  return model_with_gap, classification_model

def embedding_extraction(image):
  '''Extract embedding of the image from vgg16'''
  if image is None:
    raise "image not loaded"
  embedding_model, classification_model = load_custom_model()
  if not (image.shape[0] == 224 and image.shape[1] == 224): #dont resize if input is already in target shape
    image  = cv2.resize(image, (224,224))
  image_expand = np.expand_dims(image, axis=0)
  cv2.imwrite("test.jpg", image)
  print("here")
  embedding = embedding_model.predict(image_expand)
  classification = np.argmax(classification_model.predict(image_expand))
  return embedding, classification
