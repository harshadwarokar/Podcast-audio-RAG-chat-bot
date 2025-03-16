# Podcast-audio-RAG-chat-bot

**Author:** [harshadwarokar](https://github.com/harshadwarokar)

Podcast-audio-RAG-chat-bot is an industry-grade Retrieval-Augmented Generation (RAG) application designed to process podcast audio files. It transcribes audio using the AssemblyAI API, generates embeddings with Sentence Transformers, indexes the transcript in a Chroma vector database, and sets up a retrieval-based QA system powered by a local Llama 3.1 model (via ChatOllama). An intuitive Streamlit UI allows you to upload audio files, ask questions, and review chat history.

---

## Features

- **Audio Transcription:** Converts audio to text using AssemblyAI.
- **Embedding Generation:** Uses Sentence Transformer for generating text embeddings.
- **Vector Database:** Indexes transcripts with Chroma DB for efficient retrieval.
- **Retrieval-based QA:** Integrates LangChain with a local Llama 3.1 model (ChatOllama) to answer user queries.
- **Interactive UI:** Streamlit-based interface for uploading audio files, querying, and viewing chat history.

---

## Prerequisites

- Python 3.7 or higher
- An AssemblyAI API key. Set it in the `.env` file as `ASSEMBLYAI_API_KEY`.
- A running instance of the local Llama 3.1 model accessible via the Ollama API. Configure the URL in the `.env` file as `OLLAMA_URL`.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/harshadwarokar/Podcast-audio-RAG-chat-bot.git
   cd Podcast-audio-RAG-chat-bot
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   - Open the `.env` file and set your AssemblyAI API key.
   - Verify the `OLLAMA_URL` is correctly configured.
   - Ensure the `CHROMA_DB_PERSIST_DIRECTORY` path is set as desired.

---

## Usage

1. **Run the Application:**

   Launch the Streamlit application using:

   ```bash
   streamlit run audio.py
   ```

2. **Interact with the App:**

   - **Upload Audio File:** Use the sidebar to upload an audio file (supports formats like WAV, MP3, M4A).
   - **Transcription & Indexing:** Click on "Transcribe & Index Audio" to process the file. The app will upload the audio, transcribe it via AssemblyAI, generate embeddings, and index the transcript.
   - **Ask a Question:** Enter your query in the provided text box and click "Get Answer" to interact with the AI assistant.
   - **Chat History:** View past interactions via the "View Chat History" button.

---

## File Structure

- **`.env`**  
  Contains configuration variables such as API keys and endpoint URLs.

- **`audio.py`**  
  Main application file that implements audio processing, transcription, embedding generation, document indexing, and the retrieval-based QA system.

- **`requirements.txt`**  
  Lists all required Python packages.

---

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and open pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

For any questions or suggestions, please feel free to contact me via [GitHub Issues](https://github.com/harshadwarokar/Podcast-audio-RAG-chat-bot/issues).

