import os
from utils import image_loader
import numpy as np
from utils import image
from feature_extraction import extractor
from retrieval import index_image, query_image, clearCollection

relative_image_dir = "sample"
script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, relative_image_dir)

def generate_embeddings():
    for file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file)
        image = image_loader.load_image(image_path)
        embedding = extractor.embedding_extraction(image)
        # add image to the vector database
        index_image(embedding,file)


# clearCollection()
# generate_embeddings()
image_path = image_dir + "/000001.jpg"
image = image_loader.load_image(image_path)
embedding = extractor.embedding_extraction(image)
# index_image(embedding,"000001.jpg")

results = query_image(embedding)
print(results)
print("Length: " + str(len(results)))
image = image.load_image("/Users/gauravneupane/Documents/ml/projects/image_retrieval/output/new_classification/validation/long sleeve outwear/0ac312ba-b828-4701-87fa-0099d5c359d8.jpg")
embedding, classification = extractor.embedding_extraction(image)
print(embedding.shape, classification)
# # add image to the vector database
# index_image(embedding,"hello.txt")

# print(query_image(embedding))
