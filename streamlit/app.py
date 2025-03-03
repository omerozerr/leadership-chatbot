import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from elevenlabs.client import ElevenLabs  # Official ElevenLabs SDK
from elevenlabs import play

# --- Load environment variables from .env file ---
load_dotenv()  # Loads variables from .env in the project root

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
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
    Ensure that if the returned object is a generator, we convert it to a bytes object.
    """
    client = ElevenLabs(api_key=elevenlabs_api_key)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    # If the audio is a generator, join the chunks into a bytes object.
    if hasattr(audio, '__iter__') and not isinstance(audio, bytes):
        audio = b"".join(audio)
    return audio

# Initialize audio storage in session state if not present
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None

# --- Sidebar Setup ---
with st.sidebar:
    st.markdown("**Leadership Coach Chatbot**")
    st.markdown("[View the source code](https://github.com/your-repo)")
    if st.button("Clear Conversation"):
        # Reset conversation history with system prompt and initial greeting
        st.session_state["messages"] = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable Leadership Coach specializing in leadership "
                    "strategies, professional development, and business acumen. Provide detailed "
                    "and insightful responses tailored to leadership topics. "
                    "The current year is 2025. When performing any actions (such as web searches), "
                    "update all date references to 2025 and ignore any data labeled with earlier years."
                )
            },
            {
                "role": "assistant",
                "content": "Hi, I'm your Leadership Coach. How can I help you today?"
            }
        ]
        st.experimental_rerun()  # Rerun to update the UI

# --- Main Title and Description ---
st.title("Leadership Coach Chatbot")
st.write("This prototype integrates web search with a chat interface using LangChain, Streamlit, and ElevenLabs for voice synthesis.")

# --- Initialize Conversation History if not present ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable Leadership Coach specializing in leadership "
                "strategies, professional development, and business acumen. Provide detailed "
                "and insightful responses tailored to leadership topics."
            )
        },
        {
            "role": "assistant",
            "content": "Hi, I'm your Leadership Coach. How can I help you today?"
        }
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

    # Initialize LLM with streaming enabled
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=openai_api_key,
        streaming=True
    )

    # Initialize the DuckDuckGo search tool
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

# --- Voice Output Button using ElevenLabs play function ---
if st.session_state.get("audio_bytes"):
    if st.button("Play Voice Output"):
        play(st.session_state["audio_bytes"])

