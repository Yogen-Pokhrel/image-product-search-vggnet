import os
from utils import image_loader
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
