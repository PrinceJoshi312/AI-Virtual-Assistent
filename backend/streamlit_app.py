import streamlit as st
from assistant_engine import AssistantEngine
import time

# Page config
st.set_page_config(page_title="AI Virtual Assistant", page_icon="🤖")

# Initialize assistant engine
@st.cache_resource
def get_engine():
    return AssistantEngine()

engine = get_engine()

st.title("🤖 AI Virtual Assistant")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get response from engine
        with st.spinner("Thinking..."):
            assistant_response = engine.process_query(prompt)
        
        # Simulate typing effect
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
