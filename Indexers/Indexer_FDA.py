from __future__ import annotations
import asyncio, json, os, random, time, nest_asyncio
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
from tqdm.auto import tqdm

# ───────────────────────────── Configuration ─────────────────────────────

# Azure Blob Storage configuration
BLOB_CONNECTION_STRING = ""  # TODO: Add your connection string
CONTAINER_NAME = ""          # TODO: Add your container name

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = ""   # TODO: Add your endpoint
OPENAI_API_KEY = ""          # TODO: Add your key
DEPLOYMENT_NAME = ""         # TODO: Add your deployment name
API_VERSION = ""             # TODO: Add API version (e.g. "2024-04-01-preview")

# Azure AI Search configuration
SEARCH_ENDPOINT = ""         # TODO: Add your search endpoint
SEARCH_ADMIN_KEY = ""        # TODO: Add your admin key
SEARCH_INDEX = ""            # TODO: Add your index name

# Indexing parameters
KEY_FIELD   = "product_ndc"
VECTOR_DIM  = 1536           # Dimensionality of embedding vectors
CHUNK_SIZE  = 3_000          # Max characters per chunk
OVERLAP     = 500            # Overlap between chunks
BATCH_SIZE  = 500            # Upload to Azure Search in batches of N documents
CHECKPOINT  = Path("checkpoint.txt")  # File to resume indexing if interrupted

# ───────────────────────────── Azure Clients ─────────────────────────────

# Initialize Azure clients
blob_client   = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container     = blob_client.get_container_client(CONTAINER_NAME)

oai_client    = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION
)

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_ADMIN_KEY)
)

# ───────────────────────────── Utility Functions ─────────────────────────────

# Split long text into overlapping chunks
def split_text(txt: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    out, start = [], 0
    while start < len(txt):
        out.append(txt[start:start+size])
        start += size - overlap
    return out

# Create embedding for a text chunk with retry logic on rate limits
def embed(chunk: str, retries: int = 3):
    for attempt in range(retries):
        try:
            r = oai_client.embeddings.create(input=[chunk], model=DEPLOYMENT_NAME)
            vec = r.data[0].embedding
            if len(vec) != VECTOR_DIM:
                raise ValueError(f"Embedding size mismatch: {len(vec)} ≠ {VECTOR_DIM}")
            return vec
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                backoff = 2 ** attempt + random.random()
                time.sleep(backoff)
            else:
                print("⚠️  Embedding failed:", e)
                return None

# Compute average vector from multiple chunk embeddings
def average_vector(chunks: list[str]):
    vecs = [v for c in chunks if (v := embed(c))]
    return [sum(v) / len(v) for v in zip(*vecs)] if vecs else None

# Build the document to upload to Azure Search
def make_doc(src: dict[str, Any], vec: list[float]):
    return {
        KEY_FIELD            : src.get("product_ndc"),
        "notice"             : src.get("notice", ""),
        "notice_vector"      : vec,
        "active_ingredients" : src.get("active_ingredients", ""),
        "brand_name"         : src.get("brand_name", ""),
        "labeler_name"       : src.get("labeler_name", ""),
        "descriptions"       : src.get("descriptions", ""),
        "active_ingredient"  : src.get("active_ingredient", ""),
        "inactive_ingredient": src.get("inactive_ingredient", ""),
        "purpose"            : src.get("purpose", ""),
    }

# ───────────────────────────── Main Pipeline ─────────────────────────────

async def main():
    # Step 1: List all blob files (filtering for JSON)
    all_blobs = [b.name for b in container.list_blobs() if b.name.endswith(".json")]

    # Step 2: Read checkpoint (resume from last processed blob if any)
    start_idx = int(CHECKPOINT.read_text()) if CHECKPOINT.exists() else 0

    # Step 3: Setup progress bar
    progress = tqdm(
        all_blobs[start_idx:], total=len(all_blobs),
        initial=start_idx, desc="Indexing", unit="doc"
    )

    batch = []

    for idx, blob_name in enumerate(progress, start=start_idx):
        try:
            raw = container.download_blob(blob_name).readall()
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Skipping blob {blob_name}: {e}")
            continue

        notice_text = data.get("notice", "")
        if not notice_text:
            continue  # Skip if no text to embed

        vec = average_vector(split_text(notice_text))
        if vec is None:
            continue  # Skip if embedding failed

        batch.append(make_doc(data, vec))

        if len(batch) >= BATCH_SIZE:
            search_client.upload_documents(batch)
            batch.clear()
            CHECKPOINT.write_text(str(idx + 1))
            progress.set_postfix(sent=idx + 1)

    # Final batch flush
    if batch:
        search_client.upload_documents(batch)
        CHECKPOINT.write_text(str(len(all_blobs)))

    progress.close()
    print("🎉 Indexing complete!")

# ───────────────────────────── Script Entry Point ─────────────────────────────

if __name__ == "__main__":
    nest_asyncio.apply()  
    asyncio.run(main())
