import os
import time
import streamlit as st
import uuid
from src.agent_engine import execute_twin_agent
from src.memory_manager import DualLayerMemoryManager

# 1. Page Configuration
st.set_page_config(
    page_title="Andrew Ng Digital Twin", 
    layout="wide",
    initial_sidebar_state="expanded"
)

memory_manager = DualLayerMemoryManager()

# 2. Persist a static user session ID across interactions
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize conversational display state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Welcome! I am glad you are here. Let's explore machine learning architectures, step through deep learning concepts, or chat about agentic workflows. What project are you building today?",
            "sources": None
        }
    ]

# 3. Sidebar: Memory Dashboard Panel
st.sidebar.title("Twin Memory Dashboard")
st.sidebar.markdown("---")

# ACTUAL WORKING NEW CHAT FEATURE
if st.sidebar.button("🗑️ Clear Chat & New Session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Welcome! Fresh session initialized. What machine learning architecture or research paper should we look at today?",
            "sources": None
        }
    ]
    st.rerun()

st.sidebar.subheader("Active Session ID:")
st.sidebar.caption(f"`{st.session_state.session_id}`")
st.sidebar.markdown("---")

st.sidebar.subheader("Long-Term Brain State:")
profile_data = memory_manager.get_long_term_summary(st.session_state.session_id)
st.sidebar.info(f"{profile_data}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: #888888; font-size: 0.9rem;'>"
    "Made with ❤️ by <a href='https://github.com/gargi-m21' target='_blank' style='text-decoration: none; color: #ff4b4b; font-weight: bold;'>Gargi Mishra</a>"
    "</div>",
    unsafe_allow_html=True
)

# 4. Main Chat Interface
st.title("Digital Twin of Prof. Andrew Ng")
st.subheader("An intelligent agentic twin emulating structural pedagogy, domain expertise, and data-centric reasoning.")
st.caption("Developed by **Gargi Mishra** | ECE, DTU")
st.markdown("---")

# Render rolling chat history with main-window sources expanders
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # If the message has RAG sources attached, render them cleanly under the message
        if msg.get("sources"):
            with st.expander("View Grounding Context Sources"):
                for idx, source_info in enumerate(msg["sources"]):
                    st.markdown(f"**Source [{idx+1}]:** {source_info}")

# 5. Chat Input Handling
# Inside app.py - Update Section 5 (Chat Input Handling) completely:

if prompt := st.chat_input("Ask Professor Andrew Ng..."):
    # Append human turn to the display state
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})
    with st.chat_message("user"):
        st.write(prompt)
        
    # Execute AI inference and streaming display
    with st.chat_message("assistant"):
        with st.spinner("Analyzing text corpus & reasoning step-by-step..."):
            try:
                # 1. Unpack all 3 values returned by execute_twin_agent
                response, retrieved_docs, hallucination_info = execute_twin_agent(st.session_state.session_id, prompt)
                
                st.write(response)
                
                # 2. Build a clean list of descriptive strings from the retrieved documents
                formatted_sources_list = []
                for doc in retrieved_docs:
                    filename = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
                    page_num = doc.metadata.get('page', 'N/A')
                    # Format a beautiful summary snippet string
                    snippet_entry = f"{filename} (Page {page_num})\n\n*{doc.page_content.strip()}*"
                    formatted_sources_list.append(snippet_entry)
                
                # 3. Render the sources immediately for the current live turn
                with st.expander("📚 View Grounding Context Sources"):
                    for idx, source_info in enumerate(formatted_sources_list):
                        st.markdown(f"**Source [{idx+1}]:** {source_info}")
                        st.markdown("---")
                
                # 4. Save to session history state using our uniform string format
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": formatted_sources_list,
                    "hallucination_info": hallucination_info
                })
                
                # Injects a 1.5s baseline pause to let the async thread write its JSON memory profile cleanly
                import time
                time.sleep(1.5) 
                
                # Auto-refresh the Streamlit window to sync the sidebar long-term brain state display
                st.rerun()
            except Exception as e:
                st.error(f"Execution Error: {e}")