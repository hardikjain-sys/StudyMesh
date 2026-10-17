import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from main import process_study_materials, ask_gemma_about_concept, fetch_arxiv_papers_text

st.set_page_config(
    page_title="Interactive Knowledge Teacher",
    layout="wide"
)

if "node_chats" not in st.session_state:
    st.session_state["node_chats"] = {}
if "last_click" not in st.session_state:
    st.session_state["last_click"] = None

_mermaid_graph = components.declare_component(
    "mermaid_graph",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mermaid_component"),
)

st.title("Visual Study and Research focused Helper")
st.caption("throw in anything and it will explain anything with proper connected ideas.")

def clean(i):
    return str(i).strip().replace("-", "_").replace(" ", "_")

@st.dialog("Explanation", width="small")
def show_card(node, topic):
    node_id = clean(node.get("id"))
    st.markdown(f"### {node.get('label')}")
    st.write(node.get("description"))
    st.divider()

    chat = st.session_state["node_chats"].setdefault(node_id, [])
    history_box = st.container()

    with st.form(key=f"q_form_{node_id}", clear_on_submit=True):
        q = st.text_input("Ask Gemma about this concept")
        sent = st.form_submit_button("Send 🚀")

    if sent and q.strip():
        chat.append({"role": "user", "content": q})
        with st.spinner("Gemma 4 is answering..."):
            ans = ask_gemma_about_concept(
                topic=topic,
                concept_label=node.get("label"),
                concept_desc=node.get("description"),
                user_question=q,
            )
        chat.append({"role": "assistant", "content": ans})

    with history_box:
        for m in chat:
            st.chat_message(m["role"]).write(m["content"])

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Study Materials")

    research_mode = st.toggle("🔬 Research Mode (Auto-fetch Papers)")
    
    if research_mode:
        st.info("Fetches open-access arXiv research papers matching your topic.")
        num_papers = st.slider("Number of Papers", min_value=1, max_value=5, value=2)
    else:
        num_papers = 0

    uploaded_files = st.file_uploader(
        "Upload Diagrams or Notes (Images)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    st.markdown("""
        <style>
        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] {
            border: 1px solid white !important;
        }
        .stButton > button[kind="primary"] {
            color: black !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
    user_notes = st.text_area(
        "Additional Context / Text Notes / Research Topic",
        placeholder="e.g. Focus on truth values and logical connective rules..."
    )

    generate_btn = st.button("Learn", type="primary", use_container_width=True)

# --- Logic ---
if generate_btn:
    if not uploaded_files and not user_notes.strip() and not research_mode:
        st.warning("Please upload an image, enter text notes, or enable Research Mode.")
    elif research_mode and not user_notes.strip():
        st.warning("Please enter a research topic in the text area for Research Mode.")
    else:
        temp_paths = []
        research_text = ""
        research_papers_found = []
        try:
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_paths.append(tmp_file.name)

            if research_mode and user_notes.strip():
                with st.spinner(f"🔍 Searching arXiv & extracting text from top {num_papers} paper(s)..."):
                    research_text, research_papers_found = fetch_arxiv_papers_text(
                        query=user_notes.strip(),
                        max_papers=num_papers,
                        max_pages_per_paper=3
                    )

            with st.spinner("Gemma 4 is synthesizing materials into a Knowledge Graph..."):
                data = process_study_materials(temp_paths, user_notes, research_text)

            if research_papers_found:
                data["research_papers"] = research_papers_found

            st.session_state["graph_data"] = data
            st.session_state["node_chats"] = {}

        except Exception as e:
            st.error(f"Error processing materials: {str(e)}")

        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

# --- Display Output ---
if "graph_data" in st.session_state:
    data = st.session_state["graph_data"]
    nodes = data.get("nodes", [])

    st.subheader(f"Main Topic: {data.get('topic', 'Overview')}")
    st.info(data.get("summary", ""))

    if "research_papers" in data and data["research_papers"]:
        with st.expander("Analyzed arXiv Research Papers", expanded=False):
            for paper in data["research_papers"]:
                st.markdown(f"**[{paper['title']}]({paper['url']})**")
                st.caption(paper["summary"])

    st.caption("Click any node (or use the selector) to open its card.")

    node_options = {"None": None}
    for n in nodes:
        node_options[f"{n.get('label')} ({clean(n.get('id'))})"] = clean(n.get("id"))

    def on_dropdown_change():
        st.session_state["open_node"] = node_options[st.session_state["node_dropdown"]]

    st.selectbox("Inspect Concept Node:", list(node_options.keys()),
                 key="node_dropdown", on_change=on_dropdown_change)

    mermaid_syntax = data.get("mermaid", "")
    if mermaid_syntax:
        clicked = _mermaid_graph(code=mermaid_syntax, height=900, key="graph", default=None)
        if clicked and clicked != st.session_state["last_click"]:
            st.session_state["last_click"] = clicked
            st.session_state["open_node"] = clicked["id"]

    open_id = st.session_state.pop("open_node", None)
    if open_id:
        target = next((n for n in nodes if clean(n.get("id")) == open_id), None)
        if target:
            show_card(target, data.get("topic", "General"))