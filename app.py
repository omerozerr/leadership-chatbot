import os
import json
import faiss
import numpy as np
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from elevenlabs.client import ElevenLabs  # Official ElevenLabs SDK
from elevenlabs import play
import openai  # For embedding calls
import pandas as pd
import time


load_dotenv()  # Loads variables from .env in the project root
from openai import OpenAI
client_openai = OpenAI()

# --- Load environment variables ---

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("OpenAI API key not found in environment. Please add it to your .env file.")
    st.stop()

# Load ElevenLabs credentials
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    st.error("ElevenLabs API key not found in environment. Please add it to your .env file.")
    st.stop()

voice_id = os.getenv("ELEVENLABS_VOICE_ID")
if not voice_id:
    st.error("ElevenLabs voice ID not found in environment. Please add it to your .env file.")
    st.stop()

def synthesize_speech_sdk(text):
    """
    Convert the provided text to speech using the ElevenLabs SDK.
    If the returned object is a generator, join the chunks into a bytes object.
    """
    client_eleven = ElevenLabs(api_key=elevenlabs_api_key)
    audio = client_eleven.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    if hasattr(audio, '__iter__') and not isinstance(audio, bytes):
        audio = b"".join(audio)
    return audio

def get_embedding(text, model="text-embedding-3-small"):
    """
    Get the embedding vector for the given text using OpenAI's embeddings API.
    """
    cleaned_text = text.replace("\n", " ")
    response = client_openai.embeddings.create(input=[cleaned_text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Load the CSV with precomputed embeddings ---
df = pd.read_csv('output/embedded_transcripts_with_embeddings.csv')
# The CSV contains an 'ada_embedding' column as a string; convert each to np.array.
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

# --- Search function exactly as your example ---
def search_videos(df, product_description, n=3, pprint=True):
    embedding = get_embedding(product_description, model='text-embedding-3-small')
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    if pprint:
        for r in res.itertuples():
            print(f"{r.filename} (Part {r.chunk_index}) - Similarity: {r.similarities:.4f}")
            snippet = r.combined.replace("\n", " ") + "..."
            print(snippet)
            print()
    return res

# --- Define a threshold for using specialized context ---
SIMILARITY_THRESHOLD = 0.7  # adjust as needed

# --- Sidebar Setup ---
with st.sidebar:
    st.markdown("**Leadership Coach Chatbot**")
    st.markdown("[View the source code](https://github.com/your-repo)")
    if st.button("Clear Conversation"):
        st.session_state["messages"] = [
            {"role": "system", "content": (
                "You are a knowledgeable Leadership Coach specializing in leadership "
                "strategies, professional development, and business acumen. Provide detailed "
                "and insightful responses tailored to leadership topics. "
                "The current year is 2025. When performing any actions (such as web searches), "
                "update all date references to 2025 and ignore any data labeled with earlier years."
            )},
            {"role": "assistant", "content": "Hi, I'm your Leadership Coach. How can I help you today?"}
        ]
        st.experimental_rerun()

# --- Main Title and Description ---
st.title("Leadership Coach Chatbot")
st.write("This prototype integrates transcript-based context retrieval with a chat interface using LangChain, Streamlit, and ElevenLabs for voice synthesis.")

# --- Initialize Conversation History if not present ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": (
             "You are a knowledgeable Leadership Coach specializing in leadership "
                "strategies, professional development, and business acumen. Provide detailed "
                "and insightful responses tailored to leadership topics. "
                "The current year is 2025. When performing any actions (such as web searches), "
                "update all date references to 2025 and ignore any data labeled with earlier years."
        )},
        {"role": "assistant", "content": "Hi, I'm your Leadership Coach. How can I help you today?"}
    ]

# --- Display Conversation History (excluding system messages) ---
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

# --- Chat Input and Agent Response ---
if prompt := st.chat_input("Enter your question here..."):
    # Append user message to conversation history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # First, search the specialized transcript CSV for context.
    results = search_videos(df, prompt, n=3, pprint=False)
    # Get the top similarity from the results.
    top_similarity = results.similarities.max()
    retrieved_context = ""
    if top_similarity >= SIMILARITY_THRESHOLD:
        # Concatenate the combined texts of the results as context.
        retrieved_context = "\n".join(results.combined.tolist())
    print("-----------------------")
    print(retrieved_context) 
    print("-----------------------")

    if retrieved_context:
        context_message = (
            f"Context from specialized transcripts:\n{retrieved_context}\n\n"
            "Use the above context to inform your answer."
        )
        # Insert the context as a system message before the user's query.
        st.session_state["messages"].append({"role": "system", "content": context_message})
    
    # Initialize LLM with streaming enabled
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=openai.api_key,
        streaming=True
    )
    
    # Initialize the DuckDuckGo search tool (optional)
    search_tool = DuckDuckGoSearchRun(name="Search")
    
    # Initialize the LangChain agent with the search tool
    agent = initialize_agent(
        [search_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
    
    # Create a callback handler to stream the agent's internal thoughts (optional)
    callback_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    
    # Run the agent using the conversation history as context
    response = agent.run(st.session_state["messages"], callbacks=[callback_handler])
    
    # Append and display the assistant's response
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    
    # Synthesize the assistant's response into audio and store it in session state
    audio_bytes = synthesize_speech_sdk(response)
    st.session_state["audio_bytes"] = audio_bytes

# --- Voice Output Button ---
if st.session_state.get("audio_bytes"):
    if st.button("Play Voice Output"):
        play(st.session_state["audio_bytes"])
