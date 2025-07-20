import numpy as np
import pickle
import streamlit as st
from mistralai import Mistral
import asyncio
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import dill

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Initialize the Mistral client with your API key
api_key = "pF8jEjjcaJmvAX1nP4jhtLodp3LQxjjV"
client = Mistral(api_key=api_key)

class MistralRAGChatbot:
    def __init__(self, vector_db_path, annoy_index_path, num_trees=10):
        self.embeddings, self.texts = self.load_vector_db(vector_db_path)
        self.annoy_index = self.load_annoy_index(annoy_index_path, self.embeddings.shape[1])
        self.tfidf_matrix, self.tfidf_vectorizer = self.calculate_tfidf(self.texts)

    async def get_text_embedding(self, text, model):
        embeddings_batch_response = await client.embeddings.create_async(
            model=model,
            inputs=[text]
        )
        return embeddings_batch_response.data[0].embedding

    def load_vector_db(self, vector_db_path: str):
        try:
            with open(vector_db_path, "rb") as f:
                data = dill.load(f)
            embeddings = np.array(data['embeddings'], dtype='float32')
            texts = data['texts']
            return embeddings, texts
        except FileNotFoundError:
            logging.error(f"Error loading vector database from {vector_db_path}")
            raise
        except Exception as e:
            logging.error(f"Error unpickling vector database: {e}")
            raise

    def load_annoy_index(self, annoy_index_path: str, embedding_dim: int):
        try:
            t = AnnoyIndex(embedding_dim, 'angular')
            t.load(annoy_index_path)
            return t
        except FileNotFoundError:
            logging.error(f"Error loading Annoy index from {annoy_index_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading Annoy index: {e}")
            raise

    def calculate_tfidf(self, texts):
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All elements in 'texts' must be strings.")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return tfidf_matrix, vectorizer

    async def generate_response_with_rag(self, user_query, model="mistral-small-latest", top_k=3, response_style="Detailed"):
        try:
            query_embedding = await self.get_text_embedding(user_query, "mistral-embed")
            retrieved_docs, similarities, source_info = self.retrieve_documents(query_embedding, user_query, top_k)

            if len(retrieved_docs) == 0:
                return "No relevant documents found in the database. Please try a different query.", [], [], []

            context = " ".join([str(doc) for doc in retrieved_docs])

            if response_style == "Detailed":
                full_prompt = f"Context: {context}\n\nUser question: {user_query}\n\nProvide a detailed and insightful response."
            elif response_style == "Concise":
                full_prompt = f"Context: {context}\n\nUser question: {user_query}\n\nProvide a brief and to-the-point response."
            elif response_style == "Creative":
                full_prompt = f"Context: {context}\n\nUser question: {user_query}\n\nProvide a creative and imaginative response."
            elif response_style == "Technical":
                full_prompt = f"Context: {context}\n\nUser question: {user_query}\n\nProvide a technical and detailed response."

            async_response = await client.chat.stream_async(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    },
                ]
            )

            response = ""
            async for chunk in async_response:
                response += chunk.data.choices[0].delta.content

            return response, retrieved_docs, similarities, source_info

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {e}", [], [], []

    def retrieve_documents(self, query_embedding, user_query, top_k=3):
        # Use Annoy for initial retrieval
        top_indices_annoy, annoy_distances = self.annoy_index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
        
        # Use TF-IDF for additional retrieval
        query_tfidf = self.tfidf_vectorizer.transform([user_query])
        cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        top_indices_tfidf = np.argsort(-cosine_similarities)[:top_k]
        
        # Combine results from Annoy and TF-IDF
        combined_indices = list(set(top_indices_annoy).union(set(top_indices_tfidf)))
        retrieved_texts = [self.texts[i] for i in combined_indices]
        similarities = [(cosine_similarities[i], annoy_distances[top_indices_annoy.index(i)] if i in top_indices_annoy else None) for i in combined_indices]
        source_info = [{"text": self.texts[i], "similarity_tfidf": cosine_similarities[i], "similarity_annoy": annoy_distances[top_indices_annoy.index(i)] if i in top_indices_annoy else None} for i in combined_indices]
        
        return retrieved_texts, similarities, source_info

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon=":speech_balloon:", layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        """
        <style>
        .chat-container {
            height: 60vh;
            overflow-y: auto;
            background-color: #1E1E1E;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            flex-direction: column;
        }
        .message-user {
            background-color: #0057D9;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
            max-width: 60%;
            align-self: flex-end;
            margin-left:500px;
            text-align: left;
        }
        .message-assistant {
            background-color: #0b3127;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px;
            max-width: 60%;
            align-self: flex-start;
            text-align: left;
        }
        .fixed-header {
            position: sticky;
            top: 0;
            background-color: #222540;
            padding: 10px;
            z-index: 100;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 10px;
        }
        .stTextInput > div > input {
            color: white;
            background-color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Fixed header
    st.markdown('<div class="fixed-header"><h2>RAG Chatbot</h2></div>', unsafe_allow_html=True)

    # Sidebar for vector database and Annoy index loading
    with st.sidebar:
        st.header("Load Vector Database and Annoy Index")
        vector_db_file = st.file_uploader("Upload Vector Database (PKL)", type="pkl")
        annoy_index_file = st.file_uploader("Upload Annoy Index", type="ann")

        if vector_db_file is not None and annoy_index_file is not None:
            vector_db_path = vector_db_file.name
            annoy_index_path = annoy_index_file.name
            with st.spinner("Loading vector database and Annoy index..."):
                with open(vector_db_path, "wb") as f:
                    f.write(vector_db_file.getvalue())
                with open(annoy_index_path, "wb") as f:
                    f.write(annoy_index_file.getvalue())
                chatbot = MistralRAGChatbot(vector_db_path, annoy_index_path)
            st.success("Vector database and Annoy index loaded successfully!")

        st.header("Chatbot Settings")
        model = st.selectbox("Select Model", ["mistral-small-latest", "mistral-large-latest"])
        top_k = st.slider("Number of Documents to Retrieve (Top K)", min_value=1, max_value=10, value=3)
        response_style = st.selectbox("Response Style", ["Detailed", "Concise", "Creative", "Technical"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, (role, message, source_info) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f'<div class="message-user">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-assistant">{message}</div>', unsafe_allow_html=True)
                if source_info:
                    with st.expander(f"Show details for response {i+1}"):
                        st.write("Retrieved Documents and Similarities:")
                        tabs = st.tabs([f"Document {j+1}" for j in range(len(source_info))])
                        for j, doc in enumerate(source_info):
                            with tabs[j]:
                                st.write(f"Document: {doc['text']}")
                                st.write(f"TF-IDF Similarity: {doc['similarity_tfidf']:.4f}")
                                st.write(f"Annoy Similarity: {doc['similarity_annoy']:.4f}" if doc['similarity_annoy'] is not None else "Annoy Similarity: N/A")

    # Handle user input
    user_message = st.chat_input("Type your message here...")

    if user_message:
        st.session_state.chat_history.append(("user", user_message, []))
        chat_container.empty()
        with st.spinner("Generating response..."):
            response, retrieved_docs, similarities, source_info = asyncio.run(
                chatbot.generate_response_with_rag(user_message, model=model, top_k=top_k, response_style=response_style)
            )

            if isinstance(response, str) and "Error" in response:
                st.error(response)
            else:
                st.session_state.chat_history.append(("assistant", response, source_info))
                chat_container.empty()
                with chat_container:
                    for i, (role, message, source_info) in enumerate(st.session_state.chat_history):
                        if role == "user":
                            st.markdown(f'<div class="message-user">{message}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="message-assistant">{message}</div>', unsafe_allow_html=True)
                            if source_info:
                                with st.expander(f"Show details for response {i+1}"):
                                    st.write("Retrieved Documents and Similarities:")
                                    tabs = st.tabs([f"Document {j+1}" for j in range(len(source_info))])
                                    for j, doc in enumerate(source_info):
                                        with tabs[j]:
                                            st.write(f"Document: {doc['text']}")
                                            st.write(f"TF-IDF Similarity: {doc['similarity_tfidf']:.4f}")
                                            st.write(f"Annoy Similarity: {doc['similarity_annoy']:.4f}" if doc['similarity_annoy'] is not None else "Annoy Similarity: N/A")

if __name__ == "__main__":
    main()
