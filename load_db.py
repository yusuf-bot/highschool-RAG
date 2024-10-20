import time
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

embed_model = HuggingFaceEmbedding()
db2 = chromadb.PersistentClient(path="./paul_chromadb")
chroma_coll = db2.get_or_create_collection("paul_collection")

vector_store = ChromaVectorStore(chroma_collection=chroma_coll)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = Ollama(model="llama3.2", request_timeout=420.0)
query_engine = index.as_query_engine(llm=llm)


def repl():
    while True:
        inp = input("Enter prompt here: ")
        if inp == "quit":
            break
        start_time = time.time()
        resp = query_engine.query(inp)
        end_time = time.time()
        resp.print_response_stream()
        print(f"\n\nElapsed time: {'{0:.3g}'.format(end_time - start_time)}s")


if __name__ == "__main__":
    repl()

