# README for MistralRAG Chatbot with Streamlit Interface

This repository provides the code for a **MistralRAG Chatbot** that combines **Retrieval-Augmented Generation (RAG)** with the **Mistral API** to retrieve relevant documents and generate context-aware responses based on the user's query. The chatbot utilizes advanced retrieval techniques such as **Annoy**, **BM25**, **TF-IDF**, **Word2Vec**, and **semantic similarity**. It integrates **Streamlit**, making it easy for users to upload documents, configure settings, and interact with the chatbot through a web interface.

The chatbot can be used to query a custom set of documents and generate detailed, concise, creative, or technical responses. It allows users to explore the documents used for generating the responses, showing detailed information about the retrieved documents' similarities.

## Features
- **Document Retrieval**: The chatbot supports multiple document retrieval methods:
  - **Annoy**: Approximate Nearest Neighbor search with Annoy for fast retrieval.
  - **BM25**: A probabilistic model based on term frequency and inverse document frequency.
  - **TF-IDF**: Term Frequency-Inverse Document Frequency to evaluate the relevance of words.
  - **Word2Vec**: Uses semantic embeddings for word similarity and context.
  
- **Response Generation**: The chatbot uses the **Mistral API** to generate responses based on the retrieved documents. It supports four response styles:
  - **Detailed**
  - **Concise**
  - **Creative**
  - **Technical**

- **Reranking**: The system reranks retrieved documents based on similarity measures such as **TF-IDF** and **Annoy** to provide the most relevant context for response generation.

- **Streamlit Interface**: 
  - Upload vector databases and Annoy indices.
  - Set configuration options like model selection, number of documents to retrieve (`Top K`), and response style.
  - Interact with the chatbot in a conversational format.
  - View detailed information about the retrieved documents and their similarities.

## Prerequisites

Before you run the chatbot, ensure that you have the following:

- **Python 3.7+**: Ensure you're using a compatible version of Python.
- **Required Libraries**: Install the necessary Python libraries using:

```bash
pip install numpy pickle streamlit mistralai annoy scikit-learn dill
```

- **Mistral API Key**: Obtain an API key for **Mistral** to generate embeddings and interact with the chatbot's language model.

## Setup and Usage

### 1. **Preparing Vector Database and Annoy Index**
The chatbot uses a **Vector Database** and an **Annoy Index** for efficient document retrieval. You need to upload these files through the Streamlit interface.

- **Vector Database (PKL)**: A file that contains document embeddings and corresponding texts.
- **Annoy Index (ANN)**: A file containing the Annoy index built from the document embeddings.

### 2. **Running the Streamlit Application**
After installing the required dependencies and preparing the vector database and Annoy index, you can launch the chatbot interface by running the following command:

```bash
streamlit run app.py
```

This will start the Streamlit application, where you can:

- Upload the vector database and Annoy index files.
- Select the model, response style, and number of documents to retrieve (`Top K`).
- Type your query and interact with the chatbot.

### 3. **Using the Chatbot**
Once the files are loaded, you can start interacting with the chatbot by typing a query in the input box and selecting your desired **response style** and **retrieval settings**:

- **Response Style**: Choose between **Detailed**, **Concise**, **Creative**, or **Technical** responses.
- **Top K Documents**: Define how many documents you want to retrieve based on their relevance.

After submitting the query, the chatbot will retrieve relevant documents, generate a response, and display the results along with the document details (similarity scores and retrieved text).

### 4. **Displaying Results**
The chatbot provides the following after processing the query:

- **Generated Response**: The context-aware response based on the retrieved documents.
- **Retrieved Documents**: A preview of the documents used to generate the response.
- **Similarity Scores**: Displays both **TF-IDF** and **Annoy** similarity scores for each retrieved document.

You can expand and explore the details of each document used for generating the response.

### 5. **Chat History**
The Streamlit interface maintains a **chat history** that displays previous interactions. User queries and assistant responses are shown in a conversation-like format.

## Code Structure

### Key Components
- **MistralRAGChatbot Class**:
  - Handles loading the vector database and Annoy index.
  - Retrieves documents using **Annoy** and **TF-IDF**.
  - Generates responses using the Mistral model via the **Mistral API**.
  - Supports various response styles (Detailed, Concise, Creative, Technical).

- **Streamlit Interface**:
  - **File Upload**: Upload vector database and Annoy index files.
  - **Settings**: Select model, response style, and document retrieval options.
  - **Chat Interface**: Users can type queries, view responses, and explore document details.

### Core Functions:
1. **Loading Vector Database and Annoy Index**:
   - `load_vector_db()`: Loads the document embeddings and texts from the vector database.
   - `load_annoy_index()`: Loads the Annoy index from the provided file.

2. **Document Retrieval**:
   - `retrieve_documents()`: Retrieves documents using both **Annoy** and **TF-IDF**, combining results from both methods.
   
3. **Response Generation**:
   - `generate_response_with_rag()`: Generates the chatbot's response based on the query and retrieved documents.
   
4. **Embedding Generation**:
   - `get_text_embedding()`: Uses the **Mistral API** to generate embeddings for the user's query.

## Example Interaction

Here is an example interaction with the chatbot:

1. **User**: "What are the side effects of this medication?"
2. **Chatbot**: Returns a detailed response, such as "The medication includes possible side effects such as dizziness, headaches, and nausea."
   - Shows the retrieved documents with **TF-IDF** and **Annoy** similarities.
3. **User**: "Can you tell me more about the dosage?"
4. **Chatbot**: Returns a concise response based on relevant documents retrieved from the database.

## Streamlit Customizations

### Styling
The chatbot interface is designed using custom CSS to ensure a clean and modern layout:
- **Chat Container**: Displays user and assistant messages with distinct styles for clarity.
- **Fixed Header**: A sticky header at the top to display the title.
- **User and Assistant Messages**: Messages are visually distinguished, with the user's messages on the right and assistant's on the left.

### File Upload Section
In the **sidebar**, users can upload:
- **Vector Database** (PKL): A file containing document embeddings and texts.
- **Annoy Index** (ANN): A file containing the Annoy index.

### Response Style Options
You can select one of the following response styles:
- **Detailed**: A thorough and comprehensive response.
- **Concise**: A short and to-the-point response.
- **Creative**: An imaginative response.
- **Technical**: A detailed and technical response, often with more in-depth explanations.

## Example Output

Here’s how the system will output after querying the chatbot:

```text
User: "What are the side effects of the medication?"

Assistant (Detailed): "The medication includes possible side effects such as dizziness, headaches, and gastrointestinal discomfort. It is recommended to monitor for any adverse reactions."

Document 1: "Side effects of medication: dizziness, headache, nausea..."
TF-IDF Similarity: 0.89
Annoy Similarity: 0.75

Document 2: "Patient has reported headaches after starting the treatment..."
TF-IDF Similarity: 0.82
Annoy Similarity: 0.79
```

## Error Handling

- **File Not Found**: If the vector database or Annoy index files are not found, an error message will be displayed.
- **Embedding Generation Errors**: If there’s an issue generating embeddings (e.g., network issues with the Mistral API), the chatbot will handle the error gracefully and notify the user.
  
## License

This project is licensed under the **MIT License**. See the LICENSE file for more details.

## Conclusion

The **MistralRAG Chatbot** provides an advanced solution for querying documents and generating context-aware responses. By combining **Retrieval-Augmented Generation (RAG)** techniques with **Mistral’s language model**, it allows users to interact with a knowledge base and obtain insightful responses. The Streamlit interface makes the system user-friendly and interactive, providing an easy way to upload documents, configure settings, and explore the results.
