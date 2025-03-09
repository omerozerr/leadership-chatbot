## Demo Video and Live App

-   **Demo Video on YouTube:**  
    Watch the demo video to see a brief walkthrough of the Leadership Coach Chatbot in action.  
    [Watch the Demo Video](https://www.youtube.com/watch?v=ez9Ra2uFikc)

-   **Live Streamlit App:**  
    Experience the Leadership Coach Chatbot live by visiting the deployed application:  
    [Access the Live App](https://omerozer-enocta.streamlit.app/)

# Leadership Coach Chatbot

This project is a practical demonstration for an AI Engineer recruitment process. The goal is to build a specialized AI chatbot – the **Leadership Coach** – that provides refined and context-specific guidance on leadership strategies, professional development, and business acumen. The chatbot achieves this by combining a curated knowledge base derived from a YouTube playlist of leadership interviews with hybrid search capabilities, semantic retrieval, and web search

## Table of Contents

-   [Overview](#overview)
-   [Architecture and Workflow](#architecture-and-workflow)
-   [Setup and Installation](#setup-and-installation)
-   [Usage Instructions](#usage-instructions)

## Overview

The Leadership Coach Chatbot is designed to:

-   **Leverage Curated Knowledge:** Extract and process transcripts from a YouTube playlist of leadership interviews.
-   **Semantic Retrieval:** Use Weaviate hybrid search to semantically search the curated transcripts.
-   **Hybrid Search:** Combine specialized transcript context with web search if necessary.
-   **Natural Language Generation:** Use a ChatOpenAI model (e.g., `gpt-4o-mini`) to generate context-informed answers.
-   **Voice Synthesis:** Synthesize the final answer to speech using ElevenLabs for enhanced interactivity.
-   **Reference Inclusion:** Ensure that the answer includes references (YouTube video links or websites searched) for transparency.

## Architecture and Workflow

1. **Transcript Extraction & Preprocessing:**

    - **Input:** I have created a single transcript text file for each video and saved as a JSON object containing the video URL and transcript text. These JSON objects are stored in the `transc_single_jsons/` folder.

2. **Vector Embeddings and Data Preparation:**

    - **Embeddings Script:** The transcript JSON files are loaded and processed to compute embeddings using OpenAI's `text-embedding-3-small` model.
    - **Output:** The embeddings (with the corresponding transcript text and metadata) are stored in a CSV file (`output/embedded_transcripts_with_embeddings.csv`). ( they are not in repo because of the size limit)
    - **I did not use** that method at the end. I switched to Weaviate for easier access to embeddings

3. **Weaviate Ingestion:**

    - **Schema:** A Weaviate collection (named `Transcript`) is created with properties for the video URL and transcript text.
    - **Data Splitting:** To handle token limits, each transcript is split into smaller segments (eight parts are used in this example).
    - **Data Upload:** Each segment is added as a separate object into Weaviate. The collection is configured to use the OpenAI-based vectorizer (`text2vec_openai`) and OpenAI generative module.

4. **Main Streamlit Chatbot Application:**
    - **User Query:** The user submits a question through the chat interface.
    - **Search Query Generation:** An OpenAI model refines the user's prompt to extract key search terms.
    - **Hybrid Search in Weaviate:** The refined search query is used to perform a hybrid search in Weaviate, retrieving relevant transcript objects.
    - **Context Augmentation:** If the retrieved context exceeds a defined similarity threshold, it is concatenated and inserted as a system message along with references (video URLs) before the main agent call.
    - **Agent Response:** The LangChain agent (using ChatOpenAI) processes the conversation (augmented with context) to generate a final answer.
    - **Voice Synthesis:** The assistant’s answer is synthesized to audio via ElevenLabs, and a button in the UI allows playback. ( ElevenLabs play function did not work in streamlit cloud. That's why I created a html element )

## Setup and Installation

### Prerequisites

-   Python 3.10+
-   Weaviate Cloud or a local Weaviate instance (for hybrid search)
-   Docker (if running Weaviate locally)

### Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage Instructions

**Transcript Extraction:**

-   Run the transcript extraction script `get_transcripts.py` and get transcripts with timestapms
-   Then run `merge_transc.py` to merge raw transcripts from the `transc/` folder into JSON files in `transc_single_jsons/`

**Weaviate Ingestion:**

-   Run the Weaviate script `weaviate_store_datacreate.py` to create `Transcript` collection (splitting each transcript into eight parts) and upload the objects.

**Start the Chatbot App:**

-   Launch the main Streamlit app:

    ```bash
    streamlit run app.py
    ```
