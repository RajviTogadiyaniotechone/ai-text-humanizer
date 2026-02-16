import streamlit as st
from nlp_humanizer import NLPHumanizer

st.set_page_config(page_title="Free AI Humanizer (NLP)", layout="wide")

st.title("🧠 AI Text Humanizer ")


# Initialize Logic
@st.cache_resource
def get_humanizer():
    return NLPHumanizer()

humanizer = get_humanizer()

# Layout
st.sidebar.header("🎛️ Settings")
clean_mode = st.sidebar.checkbox("High Quality (Clean) Mode", value=True, help="Avoids short-form words (gonna, don't) and slang. Best for professional output.")
messiness = st.sidebar.slider("Messiness (Imperfections)", 0.0, 1.0, 0.10, help="Higher = More fragmented sentences. Disabled in Clean Mode.")
synonym_freq = st.sidebar.slider("Vocabulary Change", 0.0, 1.0, 0.10, help="Higher = More words replaced with synonyms.")

col1, col2 = st.columns(2)

with col1:
    st.header("📝 AI Input")
    input_text = st.text_area("Paste AI Content Here:", height=300, placeholder="Artificial Intelligence leverages comprehensive datasets to facilitate...")

with col2:
    st.header("✨ Human Output")
    
    if st.button("Humanize Text", type="primary"):
        if input_text:
            with st.spinner("Applying humanization algorithms..."):
                # Start Humanizing
                
                humanized_text = humanizer.humanize(
                    input_text, 
                    messiness=messiness, 
                    synonym_freq=synonym_freq,
                    clean_mode=clean_mode
                ) 
                
                st.text_area("Result:", value=humanized_text, height=300)
                
                st.success("Transformation Complete!")
                
                # Simple Diff Stats
                original_words = len(input_text.split())
                new_words = len(humanized_text.split())
                st.caption(f"Original word count: {original_words} | New word count: {new_words}")
        else:
            st.info("Paste some text to get started.")

st.markdown("---")
st.markdown("### How it works")
st.markdown("""
1.  **Vocabulary Simplification**: Replaces 'SAT words' (e.g., *utilize, leverage*) with simple daily language.
2.  **Contraction Enforcement**: Forces *do not* -> *don't*, *it is* -> *it's*.
3.  **Conversational Noise**: Adds words humans use subconsciously like *basically, actually, honestly*.
""")
