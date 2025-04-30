"""Pharmacist Assistant – Streamlit front‑end
English UI • Quick‑question shortcuts • Token‑streamed answers • UX goodies
"""
import os
import json
from datetime import datetime

import requests
import streamlit as st

# ───────────────────────────── Page & theme ──────────────────────────────
st.set_page_config(page_title="Pharmacist Assistant", page_icon="💊", layout="wide")

# CSS : sidebar plus large, bulles colorées, bouton copie discret
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{min-width:380px;width:380px}

    /* chat colours */
    .stChatMessage:nth-child(even) [data-testid="stMarkdown"]>div{background:#f7f9fc!important;border-radius:8px;padding:6px 10px;}
    .stChatMessage:nth-child(odd)  [data-testid="stMarkdown"]>div{background:#e8f5e9!important;border-radius:8px;padding:6px 10px;}

    /* copy button */
    .copy-btn{border:none;background:none;cursor:pointer;font-size:0.9rem;opacity:0.0;transition:opacity .2s;margin-left:6px}
    .stChatMessage:hover .copy-btn{opacity:0.8}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────── Constantes ────────────────────────────────
BACKEND_URL   = os.getenv("API_ENDPOINT")
ENABLE_STREAM = "true"
TIME_FMT      = "%H:%M"
AVATAR = {
    "user": "user__avatar.png",          # image pour l'utilisateur
    "assistant": "a.png" # image pour l'assistant
}
EXAMPLE_Q = [
    "What are the main contraindications of Flonase nasal spray?",
    "Can I take ibuprofen while I'm pregnant?",
    "Is stannous fluoride toothpaste safe for a 6-year-old child?",
]

# ───────────────────────────── Helpers ───────────────────────────────────

def add_msg(role: str, content: str) -> None:
    st.session_state.setdefault("conversation", []).append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime(TIME_FMT),
    })


def backend_stream(question: str):
    payload = {
        "query": question,
        "stream": ENABLE_STREAM,
        "conversation_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.get("conversation", [])
        ],
    }

    if not ENABLE_STREAM:
        resp = requests.post(BACKEND_URL, json=payload, timeout=60)
        resp.raise_for_status()
        yield resp.json().get("message", "No answer returned.")
        return

    try:
        with requests.post(BACKEND_URL, json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            yield resp.json()["response"]
    except Exception:
        resp = requests.post(BACKEND_URL, json={**payload, "stream": False}, timeout=60)
        resp.raise_for_status()
        yield resp.json().get("message", "Backend error")


def process_prompt(prompt: str) -> None:
    add_msg("user", prompt)

    with chat_container:
        with st.chat_message("user", avatar=AVATAR["user"]):
            st.markdown(prompt)
            st.caption(datetime.now().strftime(TIME_FMT))

    with chat_container:
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            placeholder = st.empty()
            answer = ""
            for chunk in backend_stream(prompt):
                answer += chunk
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
            st.caption(datetime.now().strftime(TIME_FMT))

    add_msg("assistant", answer)

# ───────────────────────────── Barre latérale ────────────────────────────
with st.sidebar:
    st.title("💊 Pharmacist\nAssistant")
    st.image("pharmacist.png", use_container_width=True)
    st.markdown("---")

    if st.button("🧼 Reset conversation", use_container_width=True):
        st.session_state["conversation"] = []
        st.toast("New conversation started 🆕", icon="💬")

    st.markdown("#### Example questions")
    for q in EXAMPLE_Q:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["queued_prompt"] = q
            (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()

# ───────────────────────────── Historique chat ───────────────────────────
chat_container = st.container()
with chat_container:
    for idx, m in enumerate(st.session_state.get("conversation", [])):
        with st.chat_message(m["role"], avatar=AVATAR[m["role"]]):
            # message
            st.markdown(m["content"], unsafe_allow_html=True)
            # bouton copie pour l'assistant
            if m["role"] == "assistant":
                btn_id = f"copy{idx}"
                st.markdown(
                    f"<button class='copy-btn' id='{btn_id}'>📋</button>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <script>
                    const b=document.getElementById('{btn_id}');
                    if(b){{b.onclick=()=>navigator.clipboard.writeText({json.dumps(m['content'])});}}
                    </script>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(f"🕒 {m['timestamp']}")

# ───────────────────────────── Entrée utilisateur ────────────────────────
user_input = st.chat_input("How can I help you…")
queued = st.session_state.pop("queued_prompt", None)
if user_input or queued:
    process_prompt(user_input or queued)

# ───────────────────────────── Footer & auto‑scroll ───────────────────────
#st.markdown(
  #  "<small style='display:block;text-align:center;opacity:0.6;'>Powered by Streamlit • token‑streaming</small>",
 #   unsafe_allow_html=True,
#)

if "_FOCUS_DONE" not in st.session_state:
    st.markdown(
        """
        <script>
        setTimeout(()=>{
            const box=document.querySelector('textarea[data-testid="stChatInput"]');
            if(box){box.scrollIntoView({behavior:'smooth',block:'center'});} 
        },400);
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_FOCUS_DONE"] = True
