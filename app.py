# frontend , streamlit 
import streamlit as st
import os
from retrieval import set_agent_clearance as set_clearance_v1, AccessControlledRetriever
from retrieval_v2 import set_agent_clearance as set_clearance_v2, JSONAccessControlledRetriever

ERASE_FILE = ".erased.lock"
REQUIRED_CIPHER = "EK7-ΣΔ19-βXQ//4437"

# Erase mechanism (unchanged)
if os.path.exists(ERASE_FILE):
    st.markdown(
        """
        <style>
        .main, .block-container { background-color: black !important; }
        </style>
        <h1 style='text-align:center; color:red; font-size:80px; margin-top:200px;'>ERASED</h1>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# CSS styling (unchanged)
st.markdown("""
    <style>
    .main { background-color: black; color: white; }
    h1, h2, h3, h4, h5, h6, p, label, span { color: white !important; }
    input { background-color: #333 !important; color: white !important; }
    .css-18e3th9 { background-color: black; }
    .centered-title { 
        text-align: center; font-size: 36px; font-weight: bold;
        margin-top: 20px; margin-bottom: 40px; letter-spacing: 2px; color: white;
    }
    </style>""", unsafe_allow_html=True)

# Session state initialization
if "cipher_granted" not in st.session_state:
    st.session_state.cipher_granted = False

# Header
st.markdown('<div class="centered-title">RAW AGENT - ONLINE ACCESS</div>', unsafe_allow_html=True)
level_input = st.text_input("Enter Agent Level", value="", max_chars=2)

# Main logic
if level_input:
    if level_input.isdigit():
        level = int(level_input)
        if level > 13:
            st.error("🚨 Unauthorized access, lockdown initiated.")
        else:
            st.success(f"Access Level {level} granted.")
            
            # Set clearance for both versions
            set_clearance_v1(level)
            set_clearance_v2(level)
            
            # Initialize both retrievers
            retriever_v1 = AccessControlledRetriever()
            retriever_v2 = JSONAccessControlledRetriever("final_enriched_rules.json")

            if level >= 7:
                st.markdown("### Select Communication Mode")
                mode = st.radio("Choose Mode", ["Normal Chat", "Secret Chat"])

                if mode == "Normal Chat":
                    st.markdown("#### Normal Chat")
                    normal_prompt = st.text_input("You:")
                    if normal_prompt:
                        # Use V2 retriever with greetings
                        response = retriever_v2.retrieve(normal_prompt, use_llm=True)
                        if isinstance(response, list):
                            st.error(response[0])
                        else:
                            st.write(response)

                elif mode == "Secret Chat":
                    if not st.session_state.cipher_granted:
                        st.markdown("#### Enter CIPHER Area First")
                        cipher_access = st.text_input("Enter CIPHER Code:")

                        if cipher_access:
                            if cipher_access == REQUIRED_CIPHER:
                                st.session_state.cipher_granted = True
                                st.success("Access to Secret Chat Granted.")
                            else:
                                with open(ERASE_FILE, "w") as f:
                                    f.write("burned")
                                st.rerun()

                    if st.session_state.cipher_granted:
                        st.markdown("#### Secret Chat Channel")
                        secret_prompt = st.text_input("You (Encrypted):")
                        if secret_prompt:
                            # Use V1 retriever without greetings
                            response = retriever_v1.retrieve(secret_prompt, use_llm=True)
                            if isinstance(response, list):
                                st.error(response[0])
                            else:
                                st.write(f"Shadow Response: {response}")
    else:
        st.warning("Please enter a valid numeric level.")