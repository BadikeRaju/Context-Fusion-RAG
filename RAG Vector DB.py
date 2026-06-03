import time
import fitz  # PyMuPDF
import numpy as np
import pickle
from mistralai import Mistral
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Set your API key via environment variable (never commit keys to git)
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise EnvironmentError("Set MISTRAL_API_KEY before building the vector database.")
embedding_model = "mistral-embed"  # Replace with the correct model ID

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# Function to get embeddings from Mistral AI with rate limiting and exponential backoff
def get_text_embedding_with_rate_limit(text_list, initial_delay=2, max_retries=10):
    embeddings = []
    for text in text_list:
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                token_count = len(text.split())
                if token_count > 16384:
                    print("Warning: Text chunk exceeds the token limit. Truncating the text.")
                    text = " ".join(text.split()[:16384])

                response = client.embeddings.create(
                    model=embedding_model,
                    inputs=[text]
                )
                embeddings.extend([embedding.embedding for embedding in response.data])
                time.sleep(delay)  # Add delay to avoid rate limiting
                break
            except Exception as e:
                retries += 1
                print(f"Rate limit exceeded, retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                if retries == max_retries:
                    print("Max retries reached. Skipping this chunk.")
                    break
    return embeddings

# Function to store embeddings and text in a "vector database" using pickle with progress tracking
def store_embeddings_in_vector_db(pdf_path: str, vector_db_path: str, annoy_index_path: str, chunk_size=2048, overlap=200, num_trees: int = 10):
    # Open the PDF document
    doc = fitz.open(pdf_path)

    all_embeddings = []
    all_texts = []

    total_pages = doc.page_count
    total_chunks = 0

    # Calculate the total number of chunks for progress tracking
    for start_page in range(0, total_pages, 20):
        end_page = min(start_page + 20, total_pages)
        chunks = extract_and_chunk_text_from_pages(doc, start_page, end_page, chunk_size, overlap)
        total_chunks += len(chunks)

    processed_chunks = 0

    # Process in chunks based on pages
    for start_page in range(0, total_pages, 20):
        end_page = min(start_page + 20, total_pages)
        chunks = extract_and_chunk_text_from_pages(doc, start_page, end_page, chunk_size, overlap)
        
        if chunks:
            # Generate embeddings for the text chunks
            embeddings = get_text_embedding_with_rate_limit(chunks)

            # Append to the list of all embeddings and texts
            all_embeddings.extend(embeddings)
            all_texts.extend(chunks)

            processed_chunks += len(chunks)
            percent_done = (processed_chunks / total_chunks) * 100
            percent_left = 100 - percent_done

            print(f"Processed pages {start_page + 1}-{end_page}/{total_pages}")
            print(f"Progress: {percent_done:.2f}% done, {percent_left:.2f}% left")

    # Convert embeddings to a numpy array
    embeddings_np = np.array(all_embeddings).astype('float32')

    # Store the embeddings and corresponding texts in a pickle file
    with open(vector_db_path, "wb") as f:
        pickle.dump({'embeddings': embeddings_np, 'texts': all_texts}, f)

    print(f"Stored embeddings from {total_pages} pages to the vector database.")

    # Annoy implementation: create and build the index
    if os.path.exists(annoy_index_path):
        os.remove(annoy_index_path)  # Remove old index if it exists

    embedding_dim = embeddings_np.shape[1]
    annoy_index = AnnoyIndex(embedding_dim, 'angular')  # Using angular distance

    for i, embedding in enumerate(embeddings_np):
        annoy_index.add_item(i, embedding)

    # Build the Annoy index with the specified number of trees
    annoy_index.build(num_trees)
    annoy_index.save(annoy_index_path)

    print(f"Annoy index built with {len(all_embeddings)} items and stored at '{annoy_index_path}'.")

# Function to extract and chunk text from a range of PDF pages
def extract_and_chunk_text_from_pages(doc, start_page: int, end_page: int, chunk_size=2048, overlap=200):
    all_chunks = []
    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text:
            chunks = split_text_into_chunks(text, chunk_size, overlap)
            all_chunks.extend(chunks)
    return all_chunks

# Function to split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=2048, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Function to re-rank the retrieved chunks
def rerank_chunks(query_embedding, retrieved_embeddings, top_chunks):
    similarities = cosine_similarity([query_embedding], retrieved_embeddings)[0]
    ranked_indices = similarities.argsort()[::-1]
    reranked_chunks = [top_chunks[i] for i in ranked_indices]
    reranked_similarities = [similarities[i] for i in ranked_indices]
    return reranked_chunks, reranked_similarities

# Function to load Annoy index and perform nearest neighbor search
def load_and_query_annoy_index(annoy_index_path: str, query_embedding, vector_db_path: str, num_neighbors: int = 10):
    # Load the Annoy index
    annoy_index = AnnoyIndex(len(query_embedding), 'angular')
    annoy_index.load(annoy_index_path)
    
    # Retrieve nearest neighbors
    indices, distances = annoy_index.get_nns_by_vector(query_embedding, num_neighbors, include_distances=True)
    
    # Load the embeddings and texts from the vector database
    with open(vector_db_path, "rb") as f:
        vector_db = pickle.load(f)
        retrieved_embeddings = [vector_db['embeddings'][i] for i in indices]
        retrieved_texts = [vector_db['texts'][i] for i in indices]
    
    # Re-rank the retrieved chunks
    reranked_chunks, reranked_scores = rerank_chunks(query_embedding, retrieved_embeddings, retrieved_texts)
    
    return reranked_chunks, reranked_scores

# Usage example
if __name__ == "__main__":
    pdf_path = "study.pdf"  # Replace with your PDF file path
    vector_db_path = "vector_db.pkl"
    annoy_index_path = "vector_index.ann"
    
    # Step 1: Store embeddings in the vector database
    store_embeddings_in_vector_db(pdf_path, vector_db_path, annoy_index_path)
    
    # Step 2: Example query and re-ranking
    with open(vector_db_path, "rb") as f:
        vector_db = pickle.load(f)
        example_embedding = vector_db['embeddings'][0]  # Replace with any query embedding
    
    reranked_results, scores = load_and_query_annoy_index(annoy_index_path, example_embedding, vector_db_path)
    for i, result in enumerate(reranked_results):
        print(f"Result {i+1}: {result[:300]}...")  # Print the first 300 characters of each chunk
