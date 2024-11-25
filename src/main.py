import numpy as np
from utils import image
from feature_extraction import extractor
from retrieval import index_image, query_image

image = image.load_image("/Users/gauravneupane/Documents/ml/projects/image_retrieval/output/new_classification/validation/long sleeve outwear/0ac312ba-b828-4701-87fa-0099d5c359d8.jpg")
embedding, classification = extractor.embedding_extraction(image)
print(embedding.shape, classification)
# # add image to the vector database
# index_image(embedding,"hello.txt")

# print(query_image(embedding))
