#!/usr/bin/env python
"""
Industry-Grade RAG Application with Audio Input
------------------------------------------------

This application implements a Retrieval-Augmented Generation (RAG) pipeline that:
  1. Accepts an audio file input.
  2. Transcribes the audio using the Assembly AI API.
  3. Generates embeddings for the transcript using a Sentence Transformer.
  4. Stores the embeddings in a Chroma vector database.
  5. Sets up a retrieval-based QA system using LangChain and a local Llama 3.1 model (via Ollama).
  6. Provides a Streamlit-based UI for file upload, querying, and chat history display.

Before running, ensure you have:
  - A valid Assembly AI API key (set it in the ASSEMBLYAI_API_KEY variable or via environment variable).
  - The necessary packages installed:
      pip install streamlit requests sentence-transformers langchain chromadb Pillow

Author: harshad warokar
Date: 2025-03-06
"""

import os
import re
import time
import json
import logging
import requests
import threading
import base64
from io import BytesIO

import streamlit as st
from PIL import Image

# Import for embeddings and language models
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama

# ------------------------------------------------------------------------------
# Global Configuration & Logging Setup
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Assembly AI API configuration (set your API key here or via environment variable)
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "13ef50d21e274d5cabe56f1d189ec96d")
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# ------------------------------------------------------------------------------
# Helper Classes and Functions
# ------------------------------------------------------------------------------

def image_to_base64(image_path):
    """
    Converts an image to a Base64 encoded string.
    """
    try:
        image = Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return ""
    except Exception as e:
        st.error(f"Error converting image to Base64: {e}")
        return ""

class AudioProcessor:
    """
    A helper class for handling audio files.
    This class can be extended to include additional processing (e.g., voice activity detection).
    """
    def __init__(self):
        pass

    def save_audio_file(self, uploaded_file, prefix="temp_audio_"):
        """
        Saves the uploaded audio file locally.
        """
        try:
            file_path = f"{prefix}{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info("Audio file saved: %s", file_path)
            return file_path
        except Exception as e:
            logger.error("Error saving audio file: %s", e)
            raise e

# ------------------------------------------------------------------------------
# Core RAG Application Class
# ------------------------------------------------------------------------------

class AudioRAGApplication:
    """
    The main class that encapsulates the RAG workflow:
      - Transcription via Assembly AI API.
      - Embedding generation via Sentence Transformer.
      - Indexing and retrieval using Chroma DB.
      - Question answering using a local Llama 3.1 model (via ChatOllama).
    """
    def __init__(self):
        self.llm = None
        self.embedding_model = None
        self.vectorstore = None
        self.transcript_text = ""
        self.chat_history = []
        self.audio_processor = AudioProcessor()
        self.setup_models()
        self.setup_vectorstore()

    def setup_models(self):
        """
        Loads the Sentence Transformer model for embeddings and initializes the local Llama 3.1 model via ChatOllama.
        """
        try:
            logger.info("Loading Sentence Transformer model for embeddings...")
            # Load a popular Sentence Transformer; adjust the model name as needed.
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence Transformer model loaded.")
        except Exception as e:
            logger.error("Error loading Sentence Transformer: %s", e)
            raise e

        try:
            logger.info("Initializing ChatOllama (local Llama 3.1) model...")
            # Initialize the local Llama model via Ollama.
            self.llm = ChatOllama(model="llama3.1")
            logger.info("ChatOllama model initialized.")
        except Exception as e:
            logger.error("Error initializing ChatOllama model: %s", e)
            raise e

    def setup_vectorstore(self):
        """
        Sets up the Chroma vector database using a HuggingFaceEmbeddings wrapper (which uses Sentence Transformers under the hood).
        """
        try:
            logger.info("Setting up vectorstore (Chroma DB)...")
            hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            # Persist directory for vector store (will be created if it does not exist)
            persist_dir = "chromadb"
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)
            # Pass the entire embeddings object instead of its embed_documents method.
            self.vectorstore = Chroma(persist_directory=persist_dir, embedding_function=hf_embeddings)
            logger.info("Vectorstore set up successfully.")
        except Exception as e:
            logger.error("Error setting up vectorstore: %s", e)
            raise e

    def transcribe_audio(self, audio_file_path):
        """
        Transcribes the given audio file using the Assembly AI API.
        The method performs:
          1. Uploading the file to Assembly AI.
          2. Requesting a transcription.
          3. Polling for the completed transcription.
        Returns:
            The transcribed text.
        """
        try:
            headers = {"authorization": ASSEMBLYAI_API_KEY}
            logger.info("Uploading audio file to Assembly AI: %s", audio_file_path)
            with open(audio_file_path, "rb") as f:
                upload_response = requests.post(ASSEMBLYAI_UPLOAD_URL, headers=headers, data=f)
            if upload_response.status_code != 200:
                raise Exception(f"Upload failed: {upload_response.text}")
            upload_url = upload_response.json()["upload_url"]
            logger.info("Audio file uploaded. Received URL: %s", upload_url)

            # Initiate transcription request
            transcript_request = {"audio_url": upload_url}
            logger.info("Requesting transcription from Assembly AI...")
            transcript_response = requests.post(ASSEMBLYAI_TRANSCRIPT_URL, json=transcript_request, headers=headers)
            if transcript_response.status_code != 200:
                raise Exception(f"Transcription request failed: {transcript_response.text}")
            transcript_id = transcript_response.json()["id"]
            logger.info("Transcription requested. Transcript ID: %s", transcript_id)

            # Polling for transcription status
            polling_url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"
            transcription = None
            max_retries = 60
            retries = 0
            while retries < max_retries:
                polling_response = requests.get(polling_url, headers=headers)
                status = polling_response.json().get("status")
                if status == "completed":
                    transcription = polling_response.json().get("text", "")
                    logger.info("Transcription completed.")
                    break
                elif status == "error":
                    error_msg = polling_response.json().get("error", "Unknown error")
                    raise Exception(f"Transcription error: {error_msg}")
                logger.info("Transcription status: %s. Waiting...", status)
                time.sleep(5)
                retries += 1

            if transcription is None:
                raise Exception("Transcription timed out.")
            return transcription
        except Exception as e:
            logger.error("Error during transcription: %s", e)
            raise e

    def embed_text(self, text):
        """
        Generates an embedding vector for the provided text using the Sentence Transformer.
        """
        try:
            logger.info("Generating embedding for provided text...")
            embedding = self.embedding_model.encode(text)
            logger.info("Embedding generated (vector length: %d)", len(embedding))
            return embedding
        except Exception as e:
            logger.error("Error generating embedding: %s", e)
            raise e

    def index_document(self, text, metadata=None):
        """
        Indexes the given text as a document in the vectorstore.
        """
        try:
            logger.info("Indexing document into vectorstore...")
            doc = Document(page_content=text, metadata=metadata or {})
            self.vectorstore.add_documents([doc])
            logger.info("Document indexed successfully.")
        except Exception as e:
            logger.error("Error indexing document: %s", e)
            raise e

    def create_retrieval_chain(self):
        """
        Creates a retrieval-based question answering chain using LangChain.
        The chain retrieves relevant parts of the transcript from the vectorstore and then
        uses the local Llama 3.1 model (via ChatOllama) to generate an answer.
        """
        try:
            logger.info("Creating retrieval QA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            logger.info("Retrieval QA chain created.")
            return qa_chain
        except Exception as e:
            logger.error("Error creating retrieval chain: %s", e)
            raise e

    def ask_question(self, question):
        """
        Uses the retrieval QA chain to answer the user's question based on the indexed transcript.
        """
        try:
            logger.info("Received question: %s", question)
            qa_chain = self.create_retrieval_chain()
            answer = qa_chain.run(question)
            # Append question-answer pair to chat history
            self.chat_history.append({"question": question, "answer": answer})
            logger.info("Answer generated for question.")
            return answer
        except Exception as e:
            logger.error("Error answering question: %s", e)
            raise e

    def process_audio_file(self, audio_file_path):
        """
        Orchestrates the entire process:
          - Transcribes the audio.
          - Indexes the transcript into the vectorstore.
        """
        try:
            logger.info("Processing audio file: %s", audio_file_path)
            self.transcript_text = self.transcribe_audio(audio_file_path)
            if not self.transcript_text:
                raise Exception("Empty transcript received.")
            logger.info("Transcript (first 100 chars): %s", self.transcript_text[:100])
            self.index_document(self.transcript_text, metadata={"source": audio_file_path})
        except Exception as e:
            logger.error("Error processing audio file: %s", e)
            raise e

# ------------------------------------------------------------------------------
# Streamlit UI Functions
# ------------------------------------------------------------------------------

def display_chat_history(chat_history):
    """
    Displays the chat history in the Streamlit sidebar.
    """
    st.sidebar.subheader("Chat History")
    if not chat_history:
        st.sidebar.write("No interactions yet.")
    else:
        for entry in chat_history:
            st.sidebar.markdown(f"**User:** {entry['question']}")
            st.sidebar.markdown(f"**Assistant:** {entry['answer']}")
            st.sidebar.markdown("---")

def render_header():
    """
    Renders the header with a logo and title.
    """
    # Update the paths below to the correct locations for your logo images.
    LOGO_PATH = r"logo.jpeg"
    USER_LOGO_PATH = r"user.jpeg"
    logo_b64 = image_to_base64(LOGO_PATH)
    user_logo_b64 = image_to_base64(USER_LOGO_PATH)
    st.markdown(
        f"""
        <style>
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #f9f9f9;
            padding: 10px 20px;
            border-bottom: 2px solid #000;
        }}
        .header img {{
            height: 60px;
        }}
        .user-logo {{
            border-radius: 50%;
            width: 50px;
            height: 50px;
            border: 2px solid #000;
        }}
        </style>
        <div class="header">
            <img src="data:image/jpeg;base64,{logo_b64}" alt="Logo">
            <h1>Audio Chat with RAG : AI Assistant</h1>
            <img src="data:image/jpeg;base64,{user_logo_b64}" alt="User Logo" class="user-logo">
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_footer():
    """
    Renders the footer.
    """
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            border-top: 1px solid #ccc;
        }
        </style>
        <div class="footer">
            <p>Developed by harshad warokar</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# Main Application (Streamlit UI)
# ------------------------------------------------------------------------------

def main():
    # Render header and set page configuration
    st.set_page_config(page_title="Audio Chat with RAG: AI-Powered Audio Transcription and Q&A", layout="wide")
    render_header()
    st.write("Chat with your Audio.")

    # Initialize the main application instance
    app = AudioRAGApplication()

    # Sidebar: File uploader and control buttons
    st.sidebar.header("Audio File Configuration")
    uploaded_audio = st.sidebar.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
    transcribe_button = st.sidebar.button("Transcribe & Index Audio")
    view_history_button = st.sidebar.button("View Chat History")

    # Process uploaded audio file
    if transcribe_button:
        if uploaded_audio is not None:
            try:
                audio_path = app.audio_processor.save_audio_file(uploaded_audio)
                st.info("Processing audio file. This might take several minutes...")
                app.process_audio_file(audio_path)
                st.success("Audio successfully transcribed and indexed!")
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
        else:
            st.error("Please upload an audio file first.")

    # Section: Ask a question
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your query here:")
    if st.button("Get Answer"):
        if user_question:
            try:
                answer = app.ask_question(user_question)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error obtaining answer: {str(e)}")
        else:
            st.error("Please enter a valid question.")

    # Display chat history if requested
    if view_history_button:
        display_chat_history(app.chat_history)

    # Render footer
    render_footer()

# ------------------------------------------------------------------------------
# Run the Application
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as main_exception:
        logger.error("Unhandled exception in main: %s", main_exception)
        st.error("An unexpected error occurred. Please try again later.")

# ------------------------------------------------------------------------------
# End of File
# ------------------------------------------------------------------------------
