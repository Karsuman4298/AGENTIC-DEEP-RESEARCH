import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env ──────────────────────────────────────────────────
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Research",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark mode state ────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ── Theme tokens ───────────────────────────────────────────────
if st.session_state.dark_mode:
    T = {
        "bg":                  "#0d0f18",
        "bg_secondary":        "#13151f",
        "bg_card":             "#181b29",
        "bg_card_hover":       "#1e2236",
        "border":              "#252840",
        "border_hover":        "#5b7bff",
        "text_primary":        "#e4e6f2",
        "text_secondary":      "#858dba",
        "text_muted":          "#4e5472",
        "accent":              "#5b7bff",
        "accent2":             "#7c8fff",
        "accent_light":        "#141a38",
        "accent_text":         "#8baeff",
        "label_color":         "#3a3f60",
        "answer_bg":           "#13151f",
        "answer_border":       "#252840",
        "badge_ok_bg":         "#09200f",
        "badge_ok_color":      "#3fd474",
        "badge_ok_border":     "#134825",
        "badge_warn_bg":       "#220d00",
        "badge_warn_color":    "#f5923a",
        "badge_warn_border":   "#6e2810",
        "sidebar_bg":          "#0b0d16",
        "ref_link":            "#7c8fff",
        "ref_link_hover":      "#a5b4fc",
        "trace_border":        "#252840",
        "trace_head":          "#9098c0",
        "trace_text":          "#575e80",
        "input_bg":            "#111320",
        "scrollbar_thumb":     "#252840",
        "tag_bg":              "#181b29",
        "tag_border":          "#252840",
        "num_badge_bg":        "#5b7bff",
        "num_badge_text":      "#ffffff",
        "divider":             "#252840",
        "btn_text":            "#ffffff",
        "btn_secondary_bg":    "#181b29",
        "btn_secondary_text":  "#858dba",
        "code_bg":             "#1b1e30",
        "code_text":           "#99adff",
        "shadow":              "0 8px 36px rgba(0,0,0,0.6)",
        "shadow_sm":           "0 2px 10px rgba(0,0,0,0.45)",
        "shadow_card":         "0 4px 20px rgba(0,0,0,0.5)",
        "grad1":               "#0d0f18",
        "grad2":               "#111422",
        "purple":              "#9d6ef5",
        "mode_btn_bg":         "#181b29",
        "mode_btn_color":      "#858dba",
        "mode_btn_border":     "#252840",
        "spinner_color":       "#5b7bff",
        "status_bg":           "#181b29",
        "status_border":       "#252840",
        "status_text":         "#858dba",
        "expander_bg":         "#181b29",
        "expander_border":     "#252840",
        "expander_text":       "#858dba",
        "divider_line":        "#252840",
        "sidebar_section":     "#858dba",
    }
else:
    T = {
        "bg":                  "#f2f4fb",
        "bg_secondary":        "#e8ebf6",
        "bg_card":             "#ffffff",
        "bg_card_hover":       "#eef1ff",
        "border":              "#dde2f0",
        "border_hover":        "#5b5ef4",
        "text_primary":        "#0c0f2e",
        "text_secondary":      "#444c7a",
        "text_muted":          "#8c94bc",
        "accent":              "#4b46e0",
        "accent2":             "#5b5ef4",
        "accent_light":        "#eceafd",
        "accent_text":         "#3b38c0",
        "label_color":         "#9ba3c8",
        "answer_bg":           "#ffffff",
        "answer_border":       "#dde2f0",
        "badge_ok_bg":         "#edfaf3",
        "badge_ok_color":      "#146832",
        "badge_ok_border":     "#7de0a8",
        "badge_warn_bg":       "#fff6ed",
        "badge_warn_color":    "#b83c0a",
        "badge_warn_border":   "#f8c090",
        "sidebar_bg":          "#f7f8fe",
        "ref_link":            "#3b38c0",
        "ref_link_hover":      "#24228a",
        "trace_border":        "#dde2f0",
        "trace_head":          "#262c58",
        "trace_text":          "#444c7a",
        "input_bg":            "#ffffff",
        "scrollbar_thumb":     "#c8ceea",
        "tag_bg":              "#f2f4fb",
        "tag_border":          "#dde2f0",
        "num_badge_bg":        "#4b46e0",
        "num_badge_text":      "#ffffff",
        "divider":             "#dde2f0",
        "btn_text":            "#ffffff",
        "btn_secondary_bg":    "#e8ebf6",
        "btn_secondary_text":  "#444c7a",
        "code_bg":             "#e8ebf6",
        "code_text":           "#3b38c0",
        "shadow":              "0 4px 24px rgba(75,70,224,0.10)",
        "shadow_sm":           "0 2px 8px rgba(75,70,224,0.07)",
        "shadow_card":         "0 4px 16px rgba(75,70,224,0.08)",
        "grad1":               "#f2f4fb",
        "grad2":               "#e8ebf6",
        "purple":              "#9c55f0",
        "mode_btn_bg":         "#e8ebf6",
        "mode_btn_color":      "#444c7a",
        "mode_btn_border":     "#dde2f0",
        "spinner_color":       "#4b46e0",
        "status_bg":           "#ffffff",
        "status_border":       "#dde2f0",
        "status_text":         "#444c7a",
        "expander_bg":         "#ffffff",
        "expander_border":     "#dde2f0",
        "expander_text":       "#444c7a",
        "divider_line":        "#dde2f0",
        "sidebar_section":     "#444c7a",
    }

# ══════════════════════════════════════════════════════════════════
# SIDEBAR - MUST COME FIRST BEFORE CSS
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand row + dark-mode toggle
    col_brand, col_dm = st.columns([4, 1])
    with col_brand:
        st.markdown(
            '<div class="sb-brand">Deep Research</div>'
            '<div class="sb-sub">arXiv LLM-agent papers &middot; 2024&ndash;2026</div>',
            unsafe_allow_html=True,
        )
    with col_dm:
        st.markdown('<div class="dm-btn-wrap">', unsafe_allow_html=True)
        dm_label = "🌙" if st.session_state.dark_mode else "☀️"
        if st.button(dm_label, key="dark_toggle",
                     help="Switch between dark and light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Agent component toggles
    st.markdown("**Agent components**")
    use_planner   = st.toggle("Planner",           value=True, key="toggle_planner", help="Decompose the question into targeted sub-questions before retrieval")
    use_hybrid    = st.toggle("Hybrid retrieval",  value=True, key="toggle_hybrid", help="Combines BM25 keyword search with semantic embeddings via RRF fusion")
    use_reranker  = st.toggle("Reranker",          value=True, key="toggle_reranker", help="Cross-encoder reranking of retrieved passages for higher precision")
    use_reflector = st.toggle("Reflector",         value=True, key="toggle_reflector", help="Iterative evidence loop — repeats retrieval up to 3 rounds if gaps are found")
    use_verifier  = st.toggle("Citation verifier", value=True, key="toggle_verifier", help="Lexical grounding check to flag hallucinated citations")

    st.markdown("---")

    # Index paths
    st.markdown("**Index paths**")
    index_dir  = st.text_input("ChromaDB index directory",  value="data/index", key="txt_index")
    chunks_dir = st.text_input("Chunks directory", value="data/chunks", key="txt_chunks")

    st.markdown("---")

    # Ablation results table
    results_path = Path("eval/results.json")
    if results_path.exists():
        st.markdown("**Ablation results**")
        try:
            with open(results_path) as f:
                res_data = json.load(f)
            import pandas as pd
            df = pd.DataFrame([{
                "Config":  r["config"],
                "Acc":     r.get("accuracy",       "-"),
                "Faith":   r.get("faithfulness",   "-"),
                "Cite-P":  r.get("cite_precision", "-"),
                "Lat(s)":  r.get("avg_latency_s",  "-"),
            } for r in res_data])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.caption("Could not parse results. Re-run evaluate.py.")
    else:
        st.caption("Run `python eval/evaluate.py` to populate the ablation table.")

# ── CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ══════════════════════════════════════
   BASE
══════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}}

.stApp {{
    background: linear-gradient(155deg, {T['grad1']} 0%, {T['grad2']} 100%) !important;
    min-height: 100vh;
}}

/* Only hide the default Streamlit menu and footer, NOT the sidebar */
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    padding: 2.25rem 2.75rem 6rem 2.75rem !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}}

/* ══════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════ */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: {T['scrollbar_thumb']};
    border-radius: 99px;
}}

/* ══════════════════════════════════════
   TYPOGRAPHY — global overrides
══════════════════════════════════════ */
p, span, div, li {{
    color: {T['text_primary']};
}}

/* ══════════════════════════════════════
   INPUTS
══════════════════════════════════════ */
textarea, input[type="text"] {{
    background: {T['input_bg']} !important;
    color: {T['text_primary']} !important;
    border: 1.5px solid {T['border']} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: {T['shadow_sm']} !important;
    caret-color: {T['accent']} !important;
}}
textarea:focus, input[type="text"]:focus {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 0 3px {T['accent_light']}, {T['shadow_sm']} !important;
    outline: none !important;
}}
textarea::placeholder, input::placeholder {{
    color: {T['text_muted']} !important;
    font-size: 0.91rem !important;
}}
label, .stTextInput label, .stTextArea label {{
    color: {T['text_secondary']} !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.01em !important;
}}

/* ══════════════════════════════════════
   BUTTONS — primary
══════════════════════════════════════ */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.855rem !important;
    border-radius: 9px !important;
    letter-spacing: 0.015em !important;
    transition: all 0.17s cubic-bezier(.4,0,.2,1) !important;
    cursor: pointer !important;
    border: none !important;
    outline: none !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {T['accent']} 0%, {T['accent2']} 100%) !important;
    color: {T['btn_text']} !important;
    box-shadow: 0 4px 14px rgba(75,70,224,0.32) !important;
    padding: 0.5rem 1.25rem !important;
}}
.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 7px 20px rgba(75,70,224,0.44) !important;
}}
.stButton > button[kind="primary"]:active {{
    transform: translateY(0) !important;
    filter: brightness(0.96) !important;
    box-shadow: 0 2px 8px rgba(75,70,224,0.3) !important;
}}

/* ══════════════════════════════════════
   BUTTONS — secondary / clear
══════════════════════════════════════ */
.stButton > button:not([kind="primary"]) {{
    background: {T['btn_secondary_bg']} !important;
    color: {T['btn_secondary_text']} !important;
    border: 1.5px solid {T['border']} !important;
}}
.stButton > button:not([kind="primary"]):hover {{
    border-color: {T['accent']} !important;
    color: {T['accent_text']} !important;
    background: {T['accent_light']} !important;
    transform: translateY(-1px) !important;
    box-shadow: {T['shadow_sm']} !important;
}}
.stButton > button:not([kind="primary"]):active {{
    transform: translateY(0) !important;
}}

/* ══════════════════════════════════════
   BUTTONS — suggestion pills
══════════════════════════════════════ */
button[data-testid*="baseButton-secondary"] {{
    text-align: left !important;
    background: {T['bg_card']} !important;
    color: {T['text_secondary']} !important;
    border: 1.5px solid {T['border']} !important;
    padding: 0.7rem 1rem !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    line-height: 1.5 !important;
    box-shadow: {T['shadow_sm']} !important;
    border-radius: 9px !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 3rem !important;
}}
button[data-testid*="baseButton-secondary"]:hover {{
    border-color: {T['accent']} !important;
    color: {T['accent_text']} !important;
    background: {T['bg_card_hover']} !important;
    transform: translateY(-2px) !important;
    box-shadow: {T['shadow_card']} !important;
}}

/* ══════════════════════════════════════
   TOGGLE
══════════════════════════════════════ */
.stToggle label, [data-testid="stToggle"] label {{
    color: {T['text_secondary']} !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}}
[data-testid="stToggle"] {{
    padding: 2px 0 !important;
}}

/* ══════════════════════════════════════
   STATUS WIDGET
══════════════════════════════════════ */
[data-testid="stStatusWidget"] {{
    background: {T['status_bg']} !important;
    border: 1.5px solid {T['status_border']} !important;
    border-radius: 12px !important;
    box-shadow: {T['shadow_sm']} !important;
    color: {T['status_text']} !important;
}}
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] span {{
    color: {T['status_text']} !important;
    font-size: 0.86rem !important;
}}

/* ══════════════════════════════════════
   EXPANDER
══════════════════════════════════════ */
[data-testid="stExpander"] {{
    border: 1.5px solid {T['expander_border']} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: {T['shadow_sm']} !important;
}}
[data-testid="stExpander"] summary {{
    background: {T['expander_bg']} !important;
    color: {T['expander_text']} !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    padding: 0.8rem 1.1rem !important;
    border: none !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s, color 0.15s !important;
    list-style: none !important;
}}
[data-testid="stExpander"] summary:hover {{
    background: {T['bg_card_hover']} !important;
    color: {T['accent_text']} !important;
}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    background: {T['expander_bg']} !important;
    border-top: 1.5px solid {T['expander_border']} !important;
    padding: 1rem 1.25rem 1.1rem 1.25rem !important;
}}

/* ══════════════════════════════════════
   SIDEBAR
══════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1.5px solid {T['border']} !important;
    min-width: 300px !important;
    max-width: 300px !important;
    visibility: visible !important;
    display: block !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    padding: 1.6rem 1.25rem 2rem 1.25rem !important;
}}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] small {{
    color: {T['text_muted']} !important;
    font-size: 0.76rem !important;
    line-height: 1.5 !important;
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: {T['sidebar_section']} !important;
    font-size: 0.84rem !important;
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{
    color: {T['text_primary']} !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.01em !important;
}}
section[data-testid="stSidebar"] hr {{
    border: none !important;
    border-top: 1px solid {T['divider_line']} !important;
    margin: 0.9rem 0 !important;
}}
section[data-testid="stSidebar"] input[type="text"] {{
    font-size: 0.82rem !important;
    background: {T['bg_card']} !important;
}}
section[data-testid="stSidebar"] label {{
    font-size: 0.80rem !important;
    color: {T['text_secondary']} !important;
    font-weight: 500 !important;
}}
section[data-testid="stSidebar"] [data-testid="stDataFrame"] {{
    border: 1.5px solid {T['border']} !important;
    border-radius: 9px !important;
    overflow: hidden !important;
}}

/* ══════════════════════════════════════
   ALERT
══════════════════════════════════════ */
[data-testid="stAlert"] {{
    border-radius: 9px !important;
    border: 1.5px solid {T['border']} !important;
    background: {T['bg_card']} !important;
}}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {{
    color: {T['text_secondary']} !important;
}}

/* ══════════════════════════════════════
   SPINNER
══════════════════════════════════════ */
[data-testid="stSpinner"] {{
    color: {T['spinner_color']} !important;
}}

/* ══════════════════════════════════════
   INLINE CODE
══════════════════════════════════════ */
code {{
    background: {T['code_bg']} !important;
    color: {T['code_text']} !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    font-size: 0.775rem !important;
}}

/* ══════════════════════════════════════
   COMPONENT: Main Header
══════════════════════════════════════ */
.main-header {{
    padding-bottom: 1.5rem;
    margin-bottom: 1.75rem;
    border-bottom: 1.5px solid {T['border']};
}}
.main-title {{
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.1;
    margin-bottom: 0.4rem;
    background: linear-gradient(125deg, {T['accent']} 0%, {T['accent2']} 48%, {T['purple']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.main-sub {{
    color: {T['text_secondary']};
    font-size: 0.90rem;
    font-weight: 400;
    line-height: 1.55;
    margin: 0;
}}

/* ══════════════════════════════════════
   COMPONENT: Sidebar Brand
══════════════════════════════════════ */
.sb-brand {{
    font-size: 1.0rem;
    font-weight: 800;
    color: {T['text_primary']};
    letter-spacing: -0.025em;
    line-height: 1.25;
    margin-bottom: 0.15rem;
}}
.sb-sub {{
    font-size: 0.73rem;
    color: {T['text_muted']};
    line-height: 1.45;
}}

/* ══════════════════════════════════════
   COMPONENT: Dark Mode Button
══════════════════════════════════════ */
.dm-btn-wrap button {{
    background: {T['mode_btn_bg']} !important;
    color: {T['mode_btn_color']} !important;
    border: 1.5px solid {T['mode_btn_border']} !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.35rem 0.6rem !important;
    letter-spacing: 0.02em !important;
    transition: border-color 0.15s, color 0.15s, background 0.15s !important;
}}
.dm-btn-wrap button:hover {{
    border-color: {T['accent']} !important;
    color: {T['accent_text']} !important;
    background: {T['accent_light']} !important;
}}

/* ══════════════════════════════════════
   COMPONENT: Section Label
══════════════════════════════════════ */
.slabel {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.635rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {T['label_color']};
    margin: 1.85rem 0 0.75rem 0;
    user-select: none;
}}
.slabel::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, {T['border']} 0%, transparent 100%);
}}

/* ══════════════════════════════════════
   COMPONENT: Search Wrap
══════════════════════════════════════ */
.search-wrap {{
    background: {T['bg_card']};
    border: 1.5px solid {T['border']};
    border-radius: 14px;
    padding: 18px 18px 14px 18px;
    margin-bottom: 1.25rem;
    box-shadow: {T['shadow']};
}}

/* ══════════════════════════════════════
   COMPONENT: Stats Row
══════════════════════════════════════ */
.stats-row {{
    display: flex;
    gap: 0;
    margin: 1.3rem 0 0.5rem 0;
    background: {T['bg_card']};
    border: 1.5px solid {T['border']};
    border-radius: 13px;
    overflow: hidden;
    box-shadow: {T['shadow_sm']};
    flex-wrap: wrap;
}}
.stat {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 16px 22px;
    flex: 1;
    min-width: 90px;
    border-right: 1px solid {T['border']};
    transition: background 0.15s;
    cursor: default;
}}
.stat:last-child {{ border-right: none; }}
.stat:hover {{ background: {T['bg_card_hover']}; }}
.stat-v {{
    font-size: 1.32rem;
    font-weight: 800;
    color: {T['accent']};
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
    line-height: 1;
}}
.stat-l {{
    font-size: 0.60rem;
    color: {T['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
}}

/* ══════════════════════════════════════
   COMPONENT: Source Cards
══════════════════════════════════════ */
.source-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(232px, 1fr));
    gap: 12px;
    margin: 0.75rem 0 1.7rem 0;
}}
.source-card {{
    background: {T['bg_card']};
    border: 1.5px solid {T['border']};
    border-radius: 12px;
    padding: 15px 16px;
    text-decoration: none !important;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    transition: border-color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
    box-shadow: {T['shadow_sm']};
}}
.source-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']}, {T['purple']});
    opacity: 0;
    transition: opacity 0.2s ease;
    border-radius: 12px 12px 0 0;
}}
.source-card:hover {{
    border-color: {T['border_hover']};
    transform: translateY(-3px);
    box-shadow: {T['shadow_card']};
    text-decoration: none !important;
}}
.source-card:hover::before {{ opacity: 1; }}
.source-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, {T['num_badge_bg']}, {T['accent2']});
    color: {T['num_badge_text']};
    font-size: 0.60rem;
    font-weight: 700;
    width: 21px; height: 21px;
    border-radius: 6px;
    margin-bottom: 9px;
    font-family: 'JetBrains Mono', monospace;
    box-shadow: 0 2px 6px rgba(75,70,224,0.25);
    flex-shrink: 0;
    line-height: 1;
}}
.source-title {{
    font-size: 0.81rem;
    font-weight: 600;
    color: {T['text_primary']};
    line-height: 1.42;
    margin-bottom: 7px;
    flex: 1;
}}
.source-meta {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.67rem;
    color: {T['text_muted']};
    font-family: 'JetBrains Mono', monospace;
    background: {T['tag_bg']};
    border: 1px solid {T['tag_border']};
    padding: 2px 7px;
    border-radius: 5px;
    margin-bottom: 8px;
    width: fit-content;
}}
.source-snippet {{
    font-size: 0.72rem;
    color: {T['text_secondary']};
    line-height: 1.52;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    margin-top: auto;
}}

/* ══════════════════════════════════════
   COMPONENT: Answer Block
══════════════════════════════════════ */
.answer-block {{
    position: relative;
    background: {T['answer_bg']};
    border: 1.5px solid {T['answer_border']};
    border-radius: 14px;
    padding: 30px 34px;
    line-height: 1.92;
    font-size: 0.945rem;
    color: {T['text_primary']};
    margin: 0.5rem 0 1.8rem 0;
    box-shadow: {T['shadow']};
}}
.answer-block::before {{
    content: '';
    position: absolute;
    top: -1px; left: -1px; right: -1px;
    height: 4px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']} 55%, {T['purple']});
    border-radius: 14px 14px 0 0;
}}
.answer-block p {{
    margin-bottom: 0.9rem;
    color: {T['text_primary']};
    line-height: 1.9;
}}
.answer-block p:last-child {{ margin-bottom: 0; }}

/* ══════════════════════════════════════
   COMPONENT: Citation Superscript
══════════════════════════════════════ */
.cref {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: {T['accent_light']};
    border: 1px solid {T['border_hover']};
    color: {T['accent_text']};
    font-size: 0.58rem;
    font-weight: 700;
    min-width: 17px;
    height: 17px;
    border-radius: 5px;
    padding: 0 3px;
    vertical-align: super;
    font-family: 'JetBrains Mono', monospace;
    text-decoration: none !important;
    margin: 0 1.5px;
    transition: background 0.14s, transform 0.14s, color 0.14s, border-color 0.14s;
    line-height: 1;
}}
.cref:hover {{
    background: {T['accent']};
    color: #ffffff;
    border-color: {T['accent']};
    transform: scale(1.14) translateY(-1px);
    text-decoration: none !important;
}}

/* ══════════════════════════════════════
   COMPONENT: Reference List
══════════════════════════════════════ */
.ref-list {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin: 0.5rem 0 1.6rem 0;
}}
.ref-item {{
    display: flex;
    gap: 12px;
    padding: 11px 14px;
    align-items: flex-start;
    border-radius: 9px;
    border: 1.5px solid transparent;
    transition: background 0.14s, border-color 0.14s, box-shadow 0.14s;
}}
.ref-item:hover {{
    background: {T['bg_card']};
    border-color: {T['border']};
    box-shadow: {T['shadow_sm']};
}}
.ref-num {{
    min-width: 30px;
    font-size: 0.71rem;
    font-weight: 700;
    color: {T['text_muted']};
    padding-top: 2px;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
    line-height: 1.5;
}}
.ref-body {{
    flex: 1;
    min-width: 0;
}}
.ref-title {{
    font-size: 0.87rem;
    font-weight: 600;
    color: {T['ref_link']};
    text-decoration: none !important;
    line-height: 1.48;
    display: block;
    transition: color 0.14s;
    word-break: break-word;
}}
.ref-title:hover {{
    color: {T['ref_link_hover']};
    text-decoration: underline !important;
}}
.ref-meta {{
    font-size: 0.73rem;
    color: {T['text_muted']};
    margin-top: 5px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 5px;
    line-height: 1.5;
}}
.ref-meta span {{
    color: {T['text_muted']};
}}
.ref-sep {{
    color: {T['border']};
    font-weight: 300;
}}

/* ══════════════════════════════════════
   COMPONENT: Badges
══════════════════════════════════════ */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 0.615rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 5px;
    letter-spacing: 0.02em;
    flex-shrink: 0;
}}
.badge-ok {{
    background: {T['badge_ok_bg']};
    color: {T['badge_ok_color']};
    border: 1px solid {T['badge_ok_border']};
}}
.badge-warn {{
    background: {T['badge_warn_bg']};
    color: {T['badge_warn_color']};
    border: 1px solid {T['badge_warn_border']};
}}

/* ══════════════════════════════════════
   COMPONENT: Trace Items
══════════════════════════════════════ */
.trace-list {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.trace-item {{
    border-left: 3px solid {T['accent']};
    background: {T['bg_card']};
    padding: 11px 15px 11px 16px;
    font-size: 0.81rem;
    color: {T['trace_text']};
    border-radius: 0 9px 9px 0;
    border-top: 1px solid {T['trace_border']};
    border-bottom: 1px solid {T['trace_border']};
    border-right: 1px solid {T['trace_border']};
    line-height: 1.65;
}}
.trace-head {{
    font-weight: 700;
    color: {T['trace_head']};
    margin-bottom: 5px;
    font-size: 0.745rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.trace-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {T['accent']};
    flex-shrink: 0;
}}
.trace-body {{
    color: {T['trace_text']};
    font-size: 0.80rem;
}}
.trace-body strong {{
    color: {T['text_secondary']};
    font-weight: 600;
}}
.trace-body code {{
    font-size: 0.74rem !important;
}}

/* ══════════════════════════════════════
   UTILITY: Mono tag
══════════════════════════════════════ */
.mono-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: {T['code_bg']};
    color: {T['code_text']};
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid {T['border']};
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_metadata(raw_dir: str = "data/raw") -> dict:
    """Load paper metadata from metadata.jsonl -> {arxiv_id: paper_dict}"""
    meta_path = Path(raw_dir) / "metadata.jsonl"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    p = json.loads(line)
                    meta[p["arxiv_id"]] = p
    return meta


@st.cache_resource(show_spinner=False)
def load_retriever(index_dir, chunks_dir, use_hybrid, use_reranker):
    from indexer.retriever import Retriever
    r = Retriever(
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
    )
    r.load()
    return r


def linkify_citations(text: str, id_to_num: dict) -> str:
    def _replace(m):
        aid = m.group(1).strip()
        num = id_to_num.get(aid, "?")
        return (
            f'<a href="https://arxiv.org/abs/{aid}" target="_blank" '
            f'class="cref" title="{aid}">{num}</a>'
        )
    return re.sub(r"\[arxiv:([^\]]+)\]", _replace, text, flags=re.IGNORECASE)


def format_answer_html(answer: str, id_to_num: dict) -> str:
    linked = linkify_citations(answer, id_to_num)
    paras = [
        f"<p>{p.strip()}</p>"
        for p in re.split(r"\n{2,}", linked)
        if p.strip()
    ]
    return "".join(paras) if paras else f"<p>{linked}</p>"


# ══════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════

# Header
st.markdown(
    '<div class="main-header">'
    '<div class="main-title">Deep Research</div>'
    '<div class="main-sub">'
    'Agentic retrieval-augmented research over arXiv LLM-agent papers'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Search box
st.markdown('<div class="search-wrap">', unsafe_allow_html=True)

question = st.text_area(
    label="Research question",
    label_visibility="collapsed",
    placeholder="Ask a research question — e.g. What are the main approaches to agent memory in 2024-2025?",
    height=88,
    key="q_input",
)

c1, c2, _ = st.columns([1.1, 0.9, 6])
run_btn = c1.button("Search", type="primary", use_container_width=True)
clr_btn = c2.button("Clear",                  use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if clr_btn:
    st.rerun()

# Suggested questions — shown only on empty state
if not question.strip() and not run_btn:
    st.markdown('<div class="slabel">Suggested questions</div>', unsafe_allow_html=True)
    suggestions = [
        "What is the ReAct framework and how does it combine reasoning and acting?",
        "Compare Self-RAG and standard RAG for retrieval in agentic systems.",
        "What agent memory architectures have been proposed in 2024-2025?",
        "How do multi-agent systems coordinate tasks in recent research?",
        "What benchmarks are used to evaluate LLM agents in 2024-2025?",
        "Survey the main failure modes of LLM agents identified in recent papers.",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"s{i}", use_container_width=True):
            question = s
            run_btn  = True


# ══════════════════════════════════════════════════════════════════
# AGENT EXECUTION
# ══════════════════════════════════════════════════════════════════
if run_btn and question.strip():

    metadata = load_metadata()

    # Load index
    with st.spinner("Loading index..."):
        try:
            retriever = load_retriever(index_dir, chunks_dir, use_hybrid, use_reranker)
        except Exception as e:
            st.error(
                f"**Could not load index:** {e}\n\n"
                "Run first: `python run_pipeline.py --max 5 --skip_pdfs`"
            )
            st.stop()

    from agent.Agent import plan, reflect, synthesize, verify_citations

    t0 = time.time()

    # Research pipeline — live status
    status = st.status("Researching...", expanded=True)
    with status:
        st.write("Planning search strategy...")
        sub_qs = plan(question, use_planner=use_planner)

        all_chunks, cur_queries = [], sub_qs
        round_num = 1
        while True:
            st.write(f"Retrieving sources — round {round_num}...")
            new_chunks = retriever.retrieve_multi(cur_queries, top_k=5)
            seen_ids = {c["chunk_id"] for c in all_chunks}
            for c in new_chunks:
                if c["chunk_id"] not in seen_ids:
                    all_chunks.append(c)

            st.write(f"Evaluating evidence ({len(all_chunks)} passages)...")
            done, refined = reflect(question, all_chunks, round_num, use_reflector)
            if done or not refined:
                break
            cur_queries = refined
            round_num  += 1

        st.write("Synthesizing answer...")
        answer, cited_ids = synthesize(question, all_chunks)

        st.write("Verifying citations...")
        verification = verify_citations(answer, cited_ids, all_chunks, use_verifier)

    status.update(label="Research complete", state="complete", expanded=False)

    latency  = round(time.time() - t0, 1)
    verified = verification.get("verified_ids", [])
    halluc   = verification.get("hallucinated_ids", [])
    faith    = verification.get("faithfulness", 1.0)

    # ── Stats ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat">
        <span class="stat-v">{len(all_chunks)}</span>
        <span class="stat-l">Passages</span>
      </div>
      <div class="stat">
        <span class="stat-v">{len(cited_ids)}</span>
        <span class="stat-l">Citations</span>
      </div>
      <div class="stat">
        <span class="stat-v">{int(faith * 100)}%</span>
        <span class="stat-l">Faithfulness</span>
      </div>
      <div class="stat">
        <span class="stat-v">{latency}s</span>
        <span class="stat-l">Latency</span>
      </div>
      <div class="stat">
        <span class="stat-v">{round_num}</span>
        <span class="stat-l">Rounds</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Build ordered paper list (cited first) ─────────────────
    paper_order, seen_aids = [], set()
    for aid in cited_ids:
        if aid not in seen_aids:
            paper_order.append(aid)
            seen_aids.add(aid)
    for c in all_chunks:
        aid = c.get("arxiv_id", "")
        if aid and aid not in seen_aids:
            paper_order.append(aid)
            seen_aids.add(aid)

    # First snippet per paper
    chunk_text_for: dict = {}
    for c in all_chunks:
        aid = c.get("arxiv_id", "")
        if aid and aid not in chunk_text_for:
            chunk_text_for[aid] = c.get("text", "")

    id_to_num = {aid: i + 1 for i, aid in enumerate(paper_order)}

    # ── Source cards ──────────────────────────────────────────
    st.markdown('<div class="slabel">Sources</div>', unsafe_allow_html=True)

    cards_html = '<div class="source-grid">'
    for i, aid in enumerate(paper_order[:9], 1):
        meta_paper = metadata.get(aid, {})
        title = meta_paper.get("title") or next(
            (c.get("title", aid) for c in all_chunks if c.get("arxiv_id") == aid),
            aid,
        )
        raw_snippet = chunk_text_for.get(aid, "")[:160].replace("\n", " ").strip()
        snippet     = raw_snippet + ("..." if raw_snippet else "")
        year        = meta_paper.get("published", "")[:4] or ""
        short_title = (title[:72] + "...") if len(title) > 72 else title
        meta_str    = aid + (" &middot; " + year if year else "")

        cards_html += (
            f'<a href="https://arxiv.org/abs/{aid}" target="_blank" class="source-card">'
            f'<div class="source-num">{i}</div>'
            f'<div class="source-title">{short_title}</div>'
            f'<div class="source-meta">{meta_str}</div>'
            f'<div class="source-snippet">{snippet}</div>'
            f'</a>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Answer ────────────────────────────────────────────────
    st.markdown('<div class="slabel">Answer</div>', unsafe_allow_html=True)

    answer_html = format_answer_html(answer, id_to_num)
    st.markdown(
        f'<div class="answer-block">{answer_html}</div>',
        unsafe_allow_html=True,
    )

    # ── References ────────────────────────────────────────────
    if cited_ids:
        st.markdown('<div class="slabel">References</div>', unsafe_allow_html=True)

        refs_html = '<div class="ref-list">'
        for i, aid in enumerate(cited_ids, 1):
            meta_paper = metadata.get(aid, {})
            title = meta_paper.get("title") or next(
                (c.get("title", aid) for c in all_chunks if c.get("arxiv_id") == aid),
                aid,
            )
            authors    = meta_paper.get("authors", [])
            pub        = meta_paper.get("published", "")
            year       = pub[:4] if pub else ""
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

            if aid in halluc:
                badge = '<span class="badge badge-warn">Not in corpus</span>'
            elif aid in verified:
                badge = '<span class="badge badge-ok">Verified</span>'
            else:
                badge = ""

            meta_parts = []
            if author_str:
                meta_parts.append(f'<span>{author_str}</span>')
            if year:
                meta_parts.append(f'<span>{year}</span>')
            meta_parts.append(f'<code>{aid}</code>')
            if badge:
                meta_parts.append(badge)

            meta_inner = '<span class="ref-sep">&middot;</span>'.join(meta_parts)

            refs_html += (
                f'<div class="ref-item">'
                f'<div class="ref-num">[{i}]</div>'
                f'<div class="ref-body">'
                f'<a href="https://arxiv.org/abs/{aid}" target="_blank" class="ref-title">{title}</a>'
                f'<div class="ref-meta">{meta_inner}</div>'
                f'</div>'
                f'</div>'
            )
        refs_html += '</div>'
        st.markdown(refs_html, unsafe_allow_html=True)

    # ── Research trace ────────────────────────────────────────
    with st.expander("Research trace", expanded=False):

        trace_html = '<div class="trace-list">'

        # Planner block
        sub_qs_str = "<br>".join(
            f'&nbsp;&nbsp;&bull;&nbsp;{q}' for q in sub_qs
        ) if isinstance(sub_qs, list) else str(sub_qs)

        planner_status = "enabled" if use_planner else "disabled &mdash; using original question directly"
        trace_html += (
            f'<div class="trace-item">'
            f'<div class="trace-head"><div class="trace-dot"></div>Planner</div>'
            f'<div class="trace-body">'
            f'Status: <strong>{planner_status}</strong><br>'
            f'Sub-questions ({len(sub_qs) if isinstance(sub_qs, list) else 1}):<br>'
            f'{sub_qs_str}'
            f'</div>'
            f'</div>'
        )

        # Retrieval rounds
        for rn in range(1, round_num + 1):
            q_shown = cur_queries if rn == round_num else sub_qs
            q_str = "<br>".join(
                f'&nbsp;&nbsp;&bull;&nbsp;{q}' for q in q_shown
            ) if isinstance(q_shown, list) else str(q_shown)

            hybrid_note = "hybrid (BM25 + semantic + RRF)" if use_hybrid else "semantic only"
            rerank_note = "cross-encoder reranking applied" if use_reranker else "no reranking"

            trace_html += (
                f'<div class="trace-item">'
                f'<div class="trace-head"><div class="trace-dot"></div>Retrieval &mdash; Round {rn}</div>'
                f'<div class="trace-body">'
                f'Mode: <strong>{hybrid_note}</strong> &middot; {rerank_note}<br>'
                f'Queries ({len(q_shown) if isinstance(q_shown, list) else 1}):<br>{q_str}<br>'
                f'Passages accumulated after round {rn}: <strong>{len(all_chunks)}</strong>'
                f'</div>'
                f'</div>'
            )

        # Reflector
        reflector_note = (
            f"ran for {round_num} round(s)" if use_reflector
            else "disabled &mdash; single retrieval pass"
        )
        trace_html += (
            f'<div class="trace-item">'
            f'<div class="trace-head"><div class="trace-dot"></div>Reflector</div>'
            f'<div class="trace-body">'
            f'Status: <strong>{reflector_note}</strong><br>'
            f'Total passages: <strong>{len(all_chunks)}</strong>'
            f'</div>'
            f'</div>'
        )

        # Synthesizer
        trace_html += (
            f'<div class="trace-item">'
            f'<div class="trace-head"><div class="trace-dot"></div>Synthesizer</div>'
            f'<div class="trace-body">'
            f'Citations produced: <strong>{len(cited_ids)}</strong><br>'
            f'Unique papers cited: <strong>{len(set(cited_ids))}</strong>'
            f'</div>'
            f'</div>'
        )

        # Verifier
        verifier_note = "enabled" if use_verifier else "disabled &mdash; all citations unverified"
        halluc_note   = (
            f'<strong style="color:{T["badge_warn_color"]}">{len(halluc)}</strong>'
            if halluc else f'<strong>{len(halluc)}</strong>'
        )
        trace_html += (
            f'<div class="trace-item">'
            f'<div class="trace-head"><div class="trace-dot"></div>Citation Verifier</div>'
            f'<div class="trace-body">'
            f'Status: <strong>{verifier_note}</strong><br>'
            f'Verified: <strong style="color:{T["badge_ok_color"]}">{len(verified)}</strong>'
            f'&nbsp;&middot;&nbsp;'
            f'Hallucinated: {halluc_note}'
            f'&nbsp;&middot;&nbsp;'
            f'Faithfulness: <strong>{int(faith * 100)}%</strong><br>'
            f'Latency: <strong>{latency}s</strong>'
            f'</div>'
            f'</div>'
        )

        trace_html += '</div>'
        st.markdown(trace_html, unsafe_allow_html=True)

elif run_btn and not question.strip():
    st.warning("Please enter a research question before searching.")
