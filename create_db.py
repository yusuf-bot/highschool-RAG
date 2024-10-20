from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


embed_model = HuggingFaceEmbedding()
documents = SimpleDirectoryReader(
    input_files=[
        "books/Cambridge International AS  A Level Business Coursebook  4th edition (Peter Stimpson, Alastair Farquharson) (z-lib.org) (1).pdf"
    ]
).load_data()


db = chromadb.PersistentClient(path="./paul_chromadb")
chroma_coll = db.get_or_create_collection("paul_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_coll)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)


db2 = chromadb.PersistentClient(path="./paul_chromadb")
chroma_coll = db2.get_or_create_collection("paul_collection")

vector_store = ChromaVectorStore(chroma_collection=chroma_coll)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


llm = Ollama(model="llama3.2", request_timeout=420.0)

query_engine = index.as_query_engine(llm=llm)

resp = query_engine.query(
    "wht is the difference between a public sector and private sector"
)

print(resp)

