# import chromadb
# from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import numpy as np


# client = chromadb.Client(Settings()) # Enable persistent storage

# client = chromadb.PersistentClient(
#     path="../../chroma",
#     settings=Settings(),
#     tenant=DEFAULT_TENANT,
#     database=DEFAULT_DATABASE,
# )

client =  chromadb.HttpClient(
    host="localhost",
    port=8000,
    ssl=False,
    headers=None,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Define the collection name
collection_name = "image_embeddings"
collections = client.list_collections()
found = False
for col in collections:
  if(col.name == collection_name):
    found = True
    break


# Check if the collection exists and create it if not
collection = None
if not found:
    collection = client.create_collection(collection_name)
else:
    collection = client.get_collection(collection_name)

def normalize_L2(vector):
    """Normalizes a vector to unit length using L2 norm."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def clearCollection():
  '''Clear the existing collection'''
  client.delete_collection(collection_name)
  collection = client.create_collection(collection_name)


def index_image(embedding, image_path):
  '''Add embedding of the image to the vector database with its metadata and id'''
  embedding = normalize_L2(embedding)
  collection.add(
      embeddings=embedding.flatten().tolist(),
      ids=[image_path.split("/")[-1]],
      metadatas=[{"path": image_path}]
  )

def query_image(embedding, num_output: str = 5):
  '''Query similar images from the image embedding'''
  embedding = normalize_L2(embedding)
  return collection.query(
    query_embeddings = embedding.flatten().tolist(),
    n_results = 5
)