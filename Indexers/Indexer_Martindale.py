# Import necessary libraries
import os
import pickle
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

# Import Azure Search libraries
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswParameters,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)

# Load environment variables from .env file
load_dotenv("azure_api.env")

# Azure Search config
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") 
admin_key = os.getenv("AZURE_SEARCH_KEY")
index_name = "pyartes-martindale-index"
max_chunk_size = 15000

# Azure OpenAI config
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_API_BASE")
)
embedding_model = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# Function to create batches of data
def batch(iterable, size=100):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Function to create the index in Azure Search
def create_index():
    index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(admin_key))

    # Define the index schema
    fields = [
        SimpleField(name="id", type="Edm.String", key=True),
        SearchableField(name="content", type="Edm.String", analyzer_name="fr.lucene"),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,  
            vector_search_profile_name="pyrates-martindale-vector-profile"
        )

    ]
    
    # Define the vector search configuration
    name = "pyrates-martindale"
    vector_config = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=f"{name}-vector-algorithm",
                kind="hnsw",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric="cosine"
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=f"{name}-vector-profile",
                algorithm_configuration_name=f"{name}-vector-algorithm"
            ),
        ]
    )

    # Create the index with the defined fields and vector search configuration
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_config
    )

    # Create the index in Azure Search
    try:
        index_client.create_index(index)
        print("Index créé")
    except Exception as e:
        print(f"Index déjà existant ou erreur : {e}")

# Function to get the embedding for a given text
def get_embedding(text: str):
    preview = text[:60].replace("\n", " ") + "..."  
    print(f"\n Embedding pour : {preview}")
    try:
        response = client.embeddings.create(
            input=text,
            model=embedding_model
        )
        embedding = response.data[0].embedding
        print("Embedding généré (taille:", len(embedding), ")")
        return embedding
    except Exception as e:
        print(f"Erreur embedding : {e}")
        return None

# Function to index chunks of text into Azure Search
def index_chunks(chunks):
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(admin_key))
    docs = []

    print(f"\n Début indexation : {len(chunks)} chunks à traiter\n")
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk = chunk[:max_chunk_size]
        vector = get_embedding(chunk)
        if vector:
            # Create a document with the chunk and its vector
            doc = {
                "id": f"chunk-{i}",
                "content": chunk,
                "vector": vector
            }
            docs.append(doc)

            # Print progress every 10 chunks
            if len(docs) >= 10:
                try:
                    result = search_client.upload_documents(documents=docs)
                    print(f"📤 Batch indexé (10 docs)")
                    docs = []  
                except Exception as e:
                    print(f"Erreur indexation batch : {e}")
    
    # upload any remaining documents
    if docs:
        try:
            result = search_client.upload_documents(documents=docs)
            print(f"Dernier batch indexé ({len(docs)} docs)")
        except Exception as e:
            print(f"Erreur indexation finale : {e}")

# Entry point of the script
if __name__ == "__main__":
    with open("vectordb/texts.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Create the index and index the chunks
    create_index()
    index_chunks(chunks)
