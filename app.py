import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import PyPDF2
import sqlite3
import uuid

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(" GROQ_API_KEY not found in .env file.")
    st.stop()

# ==========================
# Initialize Groq Client
# ==========================
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# ==========================
# Database Functions (SQLite)
# ==========================
DB_NAME = "chat_history.db"

def init_db():
    """Initializes the database and creates the messages table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_message(session_id, role, content):
    """Saves a single message to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def load_messages(session_id):
    """Loads all user and assistant messages for a specific session."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY timestamp ASC", 
        (session_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def clear_db_history(session_id):
    """Deletes the chat history for a specific session from the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()

# Initialize the DB when the script runs
init_db()

# ==========================
# System Prompts (Tones)
# ==========================
TONE_PROMPTS = {
    "Default": "You are a helpful AI assistant.",
    "Friendly": "You are a friendly and cheerful assistant. Always be encouraging and supportive. Use simple language and a positive attitude.",
    "Humorous": "You are a witty assistant. Add light humor where appropriate. Keep answers informative and friendly.",
    "Unfriendly": "You are an unfriendly assistant. Always don't be encouraging and supportive. Use simple language and a negative attitude.",
    "Hindi Language": "You are a helpful assistant. Answer only in Hindi. Use simple and clear language.",
    "Professional": "You are a highly professional corporate assistant. Provide concise, objective, and well-structured answers without fluff.",
    "Sarcastic": "You are a highly sarcastic assistant. Answer questions accurately but with a thick layer of sarcasm and dry wit.",
    "Pirate": "You are a swashbuckling pirate captain. Answer all questions using pirate slang, nautical terms, and an adventurous tone.",
    "Academic": "You are an academic scholar. Provide highly detailed, analytical, and well-researched answers using formal vocabulary."
}

# ==========================
# Helper Function: Extract Text
# ==========================
def extract_text_from_file(uploaded_file):
    """Extracts text from TXT or PDF files."""
    text = ""
    try:
        if uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="AI Chatbot",
    layout="wide",
)

# ==========================
# Session Management
# ==========================
# Generate a unique session ID for the user's current browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.title("Settings")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

    model = st.selectbox(
        "Choose Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            "qwen/qwen3-32b",
            "qwen/qwen3-14b",
            "qwen/qwen3-8b",
            "mistral-saba-24b",
            "mixtral-8x7b-32768",
        ],
    )

    st.divider()
    
    # --- Tone Selection ---
    selected_tones = st.multiselect(
        "Choose Tone(s)",
        list(TONE_PROMPTS.keys()),
        default=["Default"],
        help="You can select multiple tones to combine them!"
    )

    # --- Custom System Prompt Input ---
    custom_system_prompt = st.text_area(
        "Custom System Prompt",
        placeholder="Type any additional instructions for the AI here...",
    )

    st.divider()

    # --- Document Upload (RAG) ---
    st.subheader("Upload Document (RAG)")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file to chat with it.", type=["pdf", "txt"])
    
    document_context = ""
    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            document_context = extract_text_from_file(uploaded_file)
            if len(document_context) > 40000:
                document_context = document_context[:40000] + "\n\n...[TEXT TRUNCATED DUE TO LENGTH]..."
            st.success("Document loaded successfully!")

    st.divider()

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
    )

    if st.button("Clear Chat"):
        clear_db_history(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

# ==========================
# Build the Final System Prompt
# ==========================
active_prompts = []

for tone in selected_tones:
    active_prompts.append(TONE_PROMPTS[tone])

if custom_system_prompt.strip():
    active_prompts.append(f"Additional Instructions from user:\n{custom_system_prompt.strip()}")

if document_context.strip():
    active_prompts.append(
        "--- DOCUMENT CONTEXT ---\n"
        "Use the information provided in the document below to answer the user's questions. "
        "If the answer is not contained within this document, you may use your general knowledge, "
        "but prioritize the document's information.\n\n"
        f"{document_context}\n"
        "------------------------"
    )

combined_system_prompt = "\n\n".join(active_prompts)
if not combined_system_prompt:
    combined_system_prompt = "You are a helpful AI assistant."

# ==========================
# Main UI
# ==========================
st.title("AI Chatbot")
st.caption("Powered by Groq API | Chat History saved to SQLite")

with st.expander("View Current System Prompt & Context"):
    st.text(combined_system_prompt)

# ==========================
# Session State & DB Loading
# ==========================
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    # Load past messages from database for this session
    db_history = load_messages(st.session_state.session_id)
    
    # Initialize session state with the system prompt followed by DB history
    st.session_state.messages = [
        {"role": "system", "content": combined_system_prompt}
    ] + db_history
else:
    # Update the system prompt dynamically if settings change
    if st.session_state.messages[0]["role"] == "system":
        st.session_state.messages[0]["content"] = combined_system_prompt

# ==========================
# Display Chat History
# ==========================
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ==========================
# User Input
# ==========================
prompt = st.chat_input("Type your message...")

if prompt:
    # 1. Save User Message to Session State & DB
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.session_id, "user", prompt)

    # 2. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate and Display Assistant Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                temperature=temperature,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            # 4. Save Assistant Message to Session State & DB
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message(st.session_state.session_id, "assistant", full_response)

        except Exception as e:
            st.error(f"Error: {e}")