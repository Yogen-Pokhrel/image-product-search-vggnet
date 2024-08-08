from utils import image
from feature_extraction import extractor
from retrieval import index_image, query_image

image = image.load_image("/Users/gauravneupane/Documents/ml/projects/image_retrieval/src/sample/000001.jpg")
embedding = extractor.embedding_extraction(image)

# add image to the vector database
index_image(embedding,"hello.txt")

print(query_image(embedding))
