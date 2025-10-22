import streamlit as st
from projects import neuro_steady, facial_nerve, dysarthria

st.set_page_config(page_title="Neuro-Diagnostic Toolkit", layout="wide")

def main():
    st.title("Neuro-Diagnostic Toolkit 🧠")
    
    with st.sidebar:
        st.header("Select an Assessment")
        tool = st.radio("Choose a tool:",
                        ("Home", 
                         "1. Neuro-Steady: Dexterity Analyzer", 
                         "2. Facial Nerve Analyzer",
                         "3. Dysarthria (Speech) Detector"))
        
        st.markdown("---")
        st.info("""
        **About:** This is a collection of non-invasive 
        neurological assessment tools.
        
        **Built by:** Amey Chhaya (using Gemini)
        
        **Disclaimer:** This is an educational 
        prototype and not a medical device.
        """)

    if tool == "Home":
        st.header("Welcome to the Neuro-Diagnostic Toolkit")
        st.write("""
        Select an assessment tool from the sidebar on the left to begin.
        
        **Available Tools:**
        * **1. Neuro-Steady:** Analyzes fine-motor hand tremors and dexterity.
        * **2. Facial Nerve Analyzer:** Quantifies facial symmetry for C.N. VII assessment (e.g., for stroke or Bell's Palsy).
        * **3. Dysarthria Detector:** Analyzes speech patterns for neurological motor impairment using acoustic metrics.
        """)
        st.image("https://i.imgur.com/3Z1XyJc.png", 
                 caption="A conceptual overview of neuro-diagnostic tools.")

    
    elif tool == "1. Neuro-Steady: Dexterity Analyzer":
        neuro_steady.run()
    
    elif tool == "2. Facial Nerve Analyzer":
        facial_nerve.run()
    
    elif tool == "3. Dysarthria (Speech) Detector":
        dysarthria.run()

if __name__ == "__main__":
    main()