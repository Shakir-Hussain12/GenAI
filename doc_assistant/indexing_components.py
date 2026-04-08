# from langchain_community.document_loaders import TextLoader 

# loader = TextLoader("sample.txt")

# documents = loader.load()

# print(documents)



# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader 

# loader = DirectoryLoader(".\smart_products_chat\docs", glob="*.pdf", loader_cls=PyPDFLoader, recursive=True)

# documents = loader.load()

# print(len(documents))





# from langchain_text_splitters import (
#     Language,
#     RecursiveCharacterTextSplitter,
# )

# PYTHON_CODE = """
# def hello_world():
#     print("Hello, World!")

# # Call the function
# hello_world()
# """

# # Default Seperators
# # ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

# python_splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.PYTHON, chunk_size=50, chunk_overlap=0
# )
# python_docs = python_splitter.create_documents([PYTHON_CODE])
# print(python_docs)








# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()
# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=5)
# embeddings = embeddings_model.embed_query("Hello I am Osama")

# print(embeddings)






# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()
# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=5)

# documents = {
#     "I am Osama",
#     "I am am=n AI Engineer",
#     "Conducting class"
# }
# embeddings = embeddings_model.embed_documents(documents)

# print(embeddings)






# from langchain_huggingface import HuggingFaceEmbeddings

# embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# result = embedding.embed_query("When Pakistan became nuclear power?")

# print(result)





# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     # Where to save data locally, remove if not necessary
#     persist_directory="./chroma_langchain_db",
# )




# from uuid import uuid4
# from langchain_core.documents import Document
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     # Where to save data locally, remove if not necessary
#     persist_directory="./chroma_langchain_db",
# )

# document_1 = Document(
#     page_content=("I had chocolate chip pancakes and " 
#     "scrambled eggs for breakfast this morning."),
#     metadata={"source": "tweet"}
# )
# document_2 = Document(
#     page_content=("The weather forecast for tomorrow "
#     "is cloudy and overcast, with a high of 62 degrees."),
#     metadata={"source": "news"}
# )
# documents = [
#     document_1,
#     document_2
# ]
# uuids = [str(uuid4()) for _ in range(len(documents))]
# vector_store.add_documents(documents=documents, ids=uuids)








from uuid import uuid4
from langchain_core.documents import Document
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

document_1 = Document(
    page_content=("I had chocolate chip pancakes and " 
    "scrambled eggs for breakfast this morning."),
    metadata={"source": "tweet"}
)
document_2 = Document(
    page_content=("The weather forecast for tomorrow "
    "is cloudy and overcast, with a high of 62 degrees."),
    metadata={"source": "news"}
)
documents = [
    document_1,
    document_2
]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

# Check collection has data first
print("Total docs in collection:", vector_store._collection.count())

results = vector_store.similarity_search("tell me about weather", k=1)
print(results)



# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     # Where to save data locally, remove if not necessary
#     persist_directory="./chroma_langchain_db",
# )
# results = vector_store.similarity_search(
#     "tell me about weather",
#     k=1,
#     # filter={"source": "tweet"},
# )

# print(results)
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")



# updated_document_1 = Document(
#     page_content=("I had chocolate chip pancakes "
#     "and fried eggs for breakfast this morning."),
#     metadata={"source": "tweet"},
#     id=1,
# )
# updated_document_2 = Document(
#     page_content=("The weather forecast for tomorrow "
#     "is sunny and warm, with a high of 82 degrees."),
#     metadata={"source": "news"},
#     id=2,
# )