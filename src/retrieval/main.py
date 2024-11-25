import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings())
collection = client.create_collection("image_embeddings")


def index_image(embedding, image_path):
  '''Add embedding of the image to the vector database with its metadata and id'''
  collection.add(
      embeddings=embedding.flatten().tolist(),
      ids=[image_path.split("/")[-1]],
      metadatas=[{"path": image_path}]
  )

def query_image(embedding, num_output: str = 5):
  '''Query similar images from the image embedding'''
  return collection.query(
    query_embeddings = embedding.flatten().tolist(),
    n_results = 5
)