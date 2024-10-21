from llama_index.core import SimpleDirectoryReader
documents=SimpleDirectoryReader(input_files=['FILES TO TRAIN HERE']).load_data()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model=HuggingFaceEmbedding()

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

db=chromadb.PersistentClient(path="./paul_chromadb")
chroma_coll=db.get_or_create_collection("paul_collection")
vector_store=ChromaVectorStore(chroma_collection=chroma_coll)
storage_context=StorageContext.from_defaults(vector_store=vector_store)
index=VectorStoreIndex.from_documents(documents,
                                      storage_context=storage_context,
                                      embed_model=embed_model)



