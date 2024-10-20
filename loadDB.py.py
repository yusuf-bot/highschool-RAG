
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model=HuggingFaceEmbedding()


import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


db2=chromadb.PersistentClient(path='./paul_chromadb')
chroma_coll=db2.get_or_create_collection('paul_collection')

vector_store=ChromaVectorStore(chroma_collection=chroma_coll)

index=VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model
)

from llama_index.llms.ollama import Ollama
llm = Ollama(model="llama3.2", request_timeout=420.0)

query_engine=index.as_query_engine(llm=llm,streaming=True, similarity_top_k=1)
que=input('Enter prompt here: ')
print('')
while que!='quit()':
    resp=query_engine.query(que)
    resp.print_response_stream()
    print('')
    que=input('Enter prompt here: ')
    print('')