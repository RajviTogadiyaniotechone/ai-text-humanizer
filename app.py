import streamlit as st
from nlp_humanizer import NLPHumanizer
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="AI Humanizer", layout="wide", page_icon="🧠")

# Custom CSS for UI
st.markdown("""
<style>
    .main { max-width: 1200px; margin: 0 auto; }
    .header-container { text-align: center; padding: 2rem 0; }
    .header-title { font-size: 2.5rem !important; font-weight: 800 !important; color: #1E1E1E; margin-bottom: 0; }
    .header-subtitle { font-size: 1.1rem !important; color: #666; margin-top: 0.5rem; }
    .card-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 15px; color: #1E1E1E; display: flex; align-items: center; gap: 8px; }
    
    .stButton > button { border-radius: 8px !important; font-weight: 600 !important; }
    .main-action-btn > div > button { 
        background-color: #FFFFFF !important; color: #1E1E1E !important; 
        border: 1px solid #E0E4E8 !important; padding: 0.5rem 2rem !important; 
        font-size: 1.1rem !important;
    }
    .main-action-btn > div > button:hover { border-color: #FF4B4B !important; color: #FF4B4B !important; }
    
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #E0E4E8 !important;
        background-color: #FFFFFF !important;
        color: #1E1E1E !important;
        font-size: 1rem !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #1E1E1E !important;
    }
    
    .stTextArea textarea:disabled {
        background-color: #FAFBFC !important;
        color: #1E1E1E !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .humanized-highlight {
        background-color: #d4edda !important;
        color: #155724 !important;
        padding: 2px 4px !important;
        border-radius: 4px !important;
        border: 1px solid #c3e6cb !important;
        display: inline-block !important;
        margin: 1px 0 !important;
        font-weight: 500 !important;
    }
    
    .output-container {
        height: 350px; 
        overflow-y: auto; 
        padding: 1.5rem; 
        border: 1px solid #E0E4E8; 
        border-radius: 12px; 
        background-color: #FFFFFF;
        color: #1E1E1E;
        line-height: 1.6;
        font-size: 1rem;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Logic Initialization
def get_humanizer():
    return NLPHumanizer()

humanizer = get_humanizer()

# Initialize Session State
if "ai_input" not in st.session_state:
    st.session_state.ai_input = ""
if "human_output" not in st.session_state:
    st.session_state.human_output = ""
if "human_output_editable" not in st.session_state:
    st.session_state.human_output_editable = ""
if "human_output_highlighted" not in st.session_state:
    st.session_state.human_output_highlighted = ""

# --- Humanize Logic Callback ---
def run_humanization():
    input_text = st.session_state.get("ai_input", "").strip()
    if input_text:
        try:
            # Use keys for values to ensure they are fresh in callback context
            m_synonym_freq = st.session_state.get("synonym_freq_key", 0.1)
            m_clean_mode = st.session_state.get("clean_mode_key", True)
            
            result = humanizer.humanize(
                input_text,
                messiness=0.1,
                synonym_freq=m_synonym_freq,
                clean_mode=m_clean_mode
            )
            st.session_state.human_output = result
            # Set the keyed widget value BEFORE it is rendered
            st.session_state.human_output_editable = result
            # Generate highlighted version
            st.session_state.human_output_highlighted = humanizer.get_highlighted_diff(input_text, result)
            st.session_state.success_toast = True
        except Exception as e:
            st.session_state.error_msg = str(e)
    else:
        st.session_state.warning_msg = "⚠️ Please enter some text first."

# Handle Notifications
if st.session_state.get("success_toast"):
    st.toast("✅ Success!", icon="✨")
    del st.session_state.success_toast
if st.session_state.get("error_msg"):
    st.error(f"Error: {st.session_state.error_msg}")
    del st.session_state.error_msg
if st.session_state.get("warning_msg"):
    st.warning(st.session_state.warning_msg)
    del st.session_state.warning_msg

# Sidebar Settings
st.sidebar.header("🎛️ Settings")
st.sidebar.checkbox("High Quality (Clean) Mode", value=True, key="clean_mode_key")
st.sidebar.select_slider(
    "Vocabulary Change", 
    options=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0],
    value=0.10,
    key="synonym_freq_key"
)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 AI Humanizer</h1>
    <p class="header-subtitle">Transform AI content into natural, human-sounding language.</p>
</div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card-title">🤳 AI-Generated Text</div>', unsafe_allow_html=True)
    
    btn_c1, btn_c2 = st.columns(2)
    # with btn_c1:
    #     if st.button("📋 Paste Text", key="paste_trigger_btn", use_container_width=True):
    #         st.info("💡 Tip: Use Ctrl+V (or Cmd+V) to paste quickly.")
            
    with btn_c2:
        if st.button("🗑️ Clear All", key="clear_all_btn", use_container_width=True):
            st.session_state.ai_input = ""
            st.session_state.human_output = ""
            st.session_state.human_output_editable = ""
            st.rerun()

    st.text_area(
        "",
        height=350,
        placeholder="Paste your AI text here...",
        key="ai_input",
        label_visibility="collapsed"
    )
    st.markdown(f'<div style="text-align: right; color: #666; font-size: 0.8rem;">{len(st.session_state.ai_input)} characters</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-title">✍️ Humanized Output</div>', unsafe_allow_html=True)
    
    if st.session_state.human_output:
        safe_text = json.dumps(st.session_state.human_output)
        copy_html = f"""
            <button id="copy-btn" style="
                width: 100%; padding: 8px; background: white; border: 1px solid #E0E4E8;
                border-radius: 8px; cursor: pointer; font-weight: 600; color: #1E1E1E;
            ">📋 Copy to Clipboard</button>
            <script>
            document.getElementById('copy-btn').addEventListener('click', () => {{
                const text = {safe_text};
                navigator.clipboard.writeText(text).then(() => {{
                    const btn = document.getElementById('copy-btn');
                    btn.innerHTML = '✅ Copied!';
                    btn.style.borderColor = '#28a745';
                    btn.style.color = '#28a745';
                    setTimeout(() => {{
                        btn.innerHTML = '📋 Copy to Clipboard';
                        btn.style.borderColor = '#E0E4E8';
                        btn.style.color = '#1E1E1E';
                    }}, 2000);
                }});
            }});
            </script>
        """
        components.html(copy_html, height=50)
    else:
        st.button("📋 Copy to Clipboard", disabled=True, use_container_width=True)

    if st.session_state.human_output:
        st.markdown(f"""
            <div class="output-container">
                {st.session_state.human_output_highlighted or st.session_state.human_output}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.text_area(
            "",
            height=350,
            placeholder="Humanized text will appear here...",
            key="human_output_placeholder",
            label_visibility="collapsed",
            disabled=True
        )
    
    char_count = len(st.session_state.human_output)
    st.markdown(f'<div style="text-align: right; color: #666; font-size: 0.8rem;">{char_count} characters</div>', unsafe_allow_html=True)

# Bottom Action Button
st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
_, center_col, _ = st.columns([1, 1, 1])
with center_col:
    st.markdown('<div class="main-action-btn">', unsafe_allow_html=True)
    st.button(
        "✨ Humanize Text", 
        type="primary", 
        use_container_width=True,
        on_click=run_humanization
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
