"""
design.py — Healthcare AI Portal design system.

Aesthetic: warm editorial. Inspired by Claude / Anthropic.
  * Cream canvas, pure-white cards. NO dark theme.
  * Serif display face (Source Serif 4) for headlines. Inter for UI.
  * One warm terracotta accent (#C15F3C). No gradients, no glow, no neon.
  * Soft borders, minimal shadows, generous whitespace.

Public surface
--------------
    inject_global_css()
    hero(title, subtitle, eyebrow=None, stats=None)
    section_header(title, subtitle=None, eyebrow=None)
    feature_card(icon, title, desc, tag=None, icon_bg=None)
    metric_card(label, value, hint=None, variant='default')
    result_card(title, subtitle=None, variant='neutral')
    info_callout(title, body, variant='info', icon=None)
    status_badge(text, variant='neutral') -> str
    disclaimer(text)
    page_divider()
    footer(text=...)

    Color, Space, Radius, Shadow, Type  -- tokens for bespoke surfaces
"""

from __future__ import annotations

import streamlit as st


# ════════════════════════════════════════════════════════════════════
# Tokens
# ════════════════════════════════════════════════════════════════════

class Color:
    # Canvas
    bg          = "#FAF9F5"   # warm cream
    bg_alt      = "#F3F1EA"   # slightly warmer — for nested surfaces
    surface     = "#FFFFFF"   # cards lift gently off cream
    surface_tint= "#FBF9F3"   # subtle variation
    overlay     = "rgba(20, 20, 19, 0.4)"

    # Borders
    border        = "#E8E3D3"   # warm neutral
    border_soft   = "#EFEADB"
    border_strong = "#D1CAB6"
    border_focus  = "#C15F3C"

    # Text
    text_primary   = "#141413"   # rich near-black, high-contrast
    text_secondary = "#615C50"   # warm charcoal
    text_muted     = "#8F8778"   # pebble
    text_inverse   = "#FAF9F5"   # on dark surfaces (rare)

    # Brand — warm terracotta
    brand        = "#C15F3C"
    brand_hover  = "#A8522F"
    brand_active = "#8E4424"
    brand_soft   = "rgba(193, 95, 60, 0.08)"
    brand_tint   = "rgba(193, 95, 60, 0.14)"
    brand_ring   = "rgba(193, 95, 60, 0.22)"

    # Semantic — muted so they don't shout at the user
    success      = "#3D7D56"
    success_soft = "rgba(61, 125, 86, 0.10)"
    warning      = "#A5741B"
    warning_soft = "rgba(165, 116, 27, 0.10)"
    danger       = "#9A3B3B"
    danger_soft  = "rgba(154, 59, 59, 0.10)"
    info         = "#2E6E8C"
    info_soft    = "rgba(46, 110, 140, 0.10)"

    # Per-feature accents — MUTED earth tones so the home page feels
    # editorial, not a rainbow. Each is a soft tint only (used as
    # icon-tile background), never as a saturated fill.
    accent_ecg      = "#C15F3C"   # terracotta
    accent_xray     = "#2E6E8C"   # teal
    accent_risk     = "#3D7D56"   # forest
    accent_cbc      = "#7D5497"   # plum
    accent_diabetes = "#A5741B"   # amber
    accent_lipid    = "#356B7C"   # slate-teal
    accent_kidney   = "#9A3B3B"   # brick
    accent_lab      = "#495B79"   # dusk-blue


class Space:
    xs  = "4px"
    sm  = "8px"
    md  = "12px"
    lg  = "16px"
    xl  = "24px"
    x2  = "32px"
    x3  = "48px"
    x4  = "72px"


class Radius:
    # Claude uses relatively tight corners. 12px max on cards.
    sm   = "4px"
    md   = "8px"
    lg   = "12px"
    xl   = "16px"
    pill = "9999px"


class Shadow:
    # Light-theme shadows are WHISPERS, not bangs.
    none = "none"
    xs   = "0 1px 2px rgba(20, 20, 19, 0.04)"
    sm   = "0 1px 3px rgba(20, 20, 19, 0.06), 0 1px 2px rgba(20, 20, 19, 0.03)"
    md   = "0 4px 8px rgba(20, 20, 19, 0.05), 0 2px 4px rgba(20, 20, 19, 0.03)"
    lg   = "0 8px 16px rgba(20, 20, 19, 0.07), 0 4px 8px rgba(20, 20, 19, 0.04)"
    ring = "0 0 0 3px rgba(193, 95, 60, 0.18)"


class Type:
    serif    = "'Source Serif 4', 'Source Serif Pro', Georgia, 'Times New Roman', serif"
    sans     = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    mono     = "'JetBrains Mono', 'SF Mono', Menlo, monospace"

    # (size, weight, tracking, leading)
    display  = ("3.0rem",  "600", "-0.02em", "1.1")   # serif
    h1       = ("2.1rem",  "600", "-0.02em", "1.2")   # serif
    h2       = ("1.4rem",  "600", "-0.01em", "1.3")   # sans
    h3       = ("1.05rem", "600", "-0.005em","1.4")   # sans
    body     = ("1.0rem",  "400", "0",       "1.65")
    small    = ("0.88rem", "400", "0",       "1.55")
    tiny     = ("0.78rem", "500", "0.01em",  "1.4")
    eyebrow  = ("0.72rem", "600", "0.1em",   "1.3")   # uppercase


# ════════════════════════════════════════════════════════════════════
# Global CSS
# ════════════════════════════════════════════════════════════════════

def inject_global_css() -> None:
    st.markdown(f"""
<style>
    /* ── Fonts ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: {Type.sans};
        color: {Color.text_primary};
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* ── Canvas ────────────────────────────────────────────────── */
    .stApp, .main {{
        background: {Color.bg};
    }}
    .block-container {{
        padding-top: {Space.x2};
        padding-bottom: {Space.x4};
        max-width: 1100px;
    }}

    /* Kill Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ── Typography defaults ────────────────────────────────────── */
    h1, .ds-serif-h1 {{
        font-family: {Type.serif};
        font-size: {Type.h1[0]};
        font-weight: {Type.h1[1]};
        letter-spacing: {Type.h1[2]};
        line-height: {Type.h1[3]};
        color: {Color.text_primary};
    }}
    h2 {{
        font-size: {Type.h2[0]};
        font-weight: {Type.h2[1]};
        letter-spacing: {Type.h2[2]};
        line-height: {Type.h2[3]};
        color: {Color.text_primary};
    }}
    h3 {{
        font-size: {Type.h3[0]};
        font-weight: {Type.h3[1]};
        letter-spacing: {Type.h3[2]};
        line-height: {Type.h3[3]};
        color: {Color.text_primary};
    }}
    p, li, span {{
        color: {Color.text_primary};
        line-height: {Type.body[3]};
    }}

    /* ── Sidebar — editorial side rail ──────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {Color.bg_alt} !important;
        border-right: 1px solid {Color.border} !important;
        min-width: 260px !important;
        max-width: 260px !important;
        width: 260px !important;
        transform: none !important;
        position: relative !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{ width: 260px !important; padding: {Space.xl} {Space.lg}; }}
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{ display: none !important; }}
    [data-testid="stSidebar"] * {{ color: {Color.text_primary} !important; }}
    [data-testid="stSidebar"] hr {{
        border: none !important;
        border-top: 1px solid {Color.border} !important;
        margin: {Space.md} 0 !important;
    }}
    [data-testid="stSidebar"] .stRadio > div {{
        gap: 2px !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {Color.text_secondary} !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: {Radius.md} !important;
        transition: background 0.15s ease, color 0.15s ease;
        font-size: 0.92rem !important;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: {Color.brand_soft} !important;
        color: {Color.text_primary} !important;
    }}
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {Color.text_muted} !important;
    }}

    /* ── Buttons ────────────────────────────────────────────────── */
    .stButton > button {{
        background: {Color.brand} !important;
        color: {Color.text_inverse} !important;
        border: 1px solid {Color.brand} !important;
        border-radius: {Radius.md} !important;
        padding: 10px 20px !important;
        font-family: {Type.sans} !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        letter-spacing: -0.005em !important;
        box-shadow: {Shadow.xs} !important;
        transition: background 0.15s ease, border-color 0.15s ease,
                    transform 0.15s ease, box-shadow 0.15s ease !important;
    }}
    .stButton > button:hover {{
        background: {Color.brand_hover} !important;
        border-color: {Color.brand_hover} !important;
        box-shadow: {Shadow.sm} !important;
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        background: {Color.brand_active} !important;
        transform: translateY(0);
    }}
    .stButton > button:focus {{
        box-shadow: {Shadow.ring} !important;
        outline: none !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: {Color.surface} !important;
        color: {Color.text_primary} !important;
        border: 1px solid {Color.border_strong} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: {Color.bg_alt} !important;
        border-color: {Color.brand} !important;
        color: {Color.brand} !important;
    }}

    /* ── Inputs ─────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stTextArea textarea,
    .stDateInput input {{
        background: {Color.surface} !important;
        border: 1px solid {Color.border_strong} !important;
        border-radius: {Radius.md} !important;
        color: {Color.text_primary} !important;
        font-family: {Type.sans} !important;
        font-size: 0.95rem !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {{
        border-color: {Color.brand} !important;
        box-shadow: {Shadow.ring} !important;
        outline: none !important;
    }}
    label, .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stRadio > label, .stCheckbox > label {{
        color: {Color.text_primary} !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
    }}

    /* Sliders */
    .stSlider [data-baseweb="slider"] > div > div > div {{
        background: {Color.brand} !important;
    }}
    .stSlider [role="slider"] {{
        background: {Color.brand} !important;
        border: 2px solid {Color.surface} !important;
        box-shadow: {Shadow.sm} !important;
    }}

    /* ── Tabs ───────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0 !important;
        background: transparent !important;
        border-bottom: 1px solid {Color.border} !important;
        padding: 0 !important;
        border-radius: 0 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 0 !important;
        padding: 12px 18px !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        color: {Color.text_secondary} !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -1px !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {Color.text_primary} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {Color.brand} !important;
        border-bottom-color: {Color.brand} !important;
        background: transparent !important;
    }}

    /* ── Tables / dataframes ────────────────────────────────────── */
    .stDataFrame, .dataframe {{
        border-radius: {Radius.md};
        border: 1px solid {Color.border};
        overflow: hidden;
        box-shadow: {Shadow.xs};
    }}

    /* ── File uploader ──────────────────────────────────────────── */
    [data-testid="stFileUploader"] section {{
        background: {Color.surface} !important;
        border: 1.5px dashed {Color.border_strong} !important;
        border-radius: {Radius.lg} !important;
        padding: {Space.xl} !important;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: {Color.brand} !important;
        background: {Color.brand_soft} !important;
    }}

    /* ── Native Streamlit alerts — soften them ──────────────────── */
    [data-testid="stAlert"] {{
        border-radius: {Radius.md} !important;
        border-left-width: 3px !important;
        box-shadow: {Shadow.xs} !important;
        padding: {Space.md} {Space.lg} !important;
    }}

    /* ── Column stretch (equal-height cards) ────────────────────── */
    [data-testid="stHorizontalBlock"] {{
        display: flex !important;
        align-items: stretch !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        display: flex !important;
        flex-direction: column !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {{
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }}

    /* ════════════════════════════════════════════════════════════
       Components (used by helper functions below)
       ════════════════════════════════════════════════════════════ */

    /* Hero — editorial, no gradient blob */
    .ds-hero {{
        margin: {Space.sm} 0 {Space.x2};
        padding: 0 0 {Space.x2};
        border-bottom: 1px solid {Color.border};
    }}
    .ds-hero__eyebrow {{
        font-family: {Type.sans};
        font-size: {Type.eyebrow[0]};
        font-weight: {Type.eyebrow[1]};
        letter-spacing: {Type.eyebrow[2]};
        text-transform: uppercase;
        color: {Color.brand};
        margin: 0 0 {Space.md} 0;
    }}
    .ds-hero__title {{
        font-family: {Type.serif};
        font-size: {Type.display[0]};
        font-weight: {Type.display[1]};
        letter-spacing: {Type.display[2]};
        line-height: {Type.display[3]};
        color: {Color.text_primary};
        margin: 0 0 {Space.md} 0;
        max-width: 820px;
    }}
    .ds-hero__subtitle {{
        font-family: {Type.sans};
        font-size: 1.1rem;
        font-weight: 400;
        color: {Color.text_secondary};
        margin: 0;
        max-width: 680px;
        line-height: 1.6;
    }}
    .ds-hero__stats {{
        display: flex;
        gap: {Space.xl};
        margin-top: {Space.xl};
        flex-wrap: wrap;
    }}
    .ds-stat {{
        padding: 0;
    }}
    .ds-stat__value {{
        font-family: {Type.serif};
        font-size: 1.7rem;
        font-weight: 600;
        color: {Color.text_primary};
        letter-spacing: -0.01em;
        line-height: 1;
    }}
    .ds-stat__label {{
        font-size: {Type.tiny[0]};
        font-weight: {Type.tiny[1]};
        color: {Color.text_muted};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 6px;
    }}

    /* Section header */
    .ds-section-head {{ margin: 0 0 {Space.xl}; }}
    .ds-section-head__eyebrow {{
        font-family: {Type.sans};
        font-size: {Type.eyebrow[0]};
        font-weight: {Type.eyebrow[1]};
        letter-spacing: {Type.eyebrow[2]};
        text-transform: uppercase;
        color: {Color.brand};
        margin: 0 0 {Space.sm} 0;
    }}
    .ds-section-head__title {{
        font-family: {Type.serif};
        font-size: {Type.h1[0]};
        font-weight: {Type.h1[1]};
        letter-spacing: {Type.h1[2]};
        line-height: {Type.h1[3]};
        color: {Color.text_primary};
        margin: 0 0 {Space.sm} 0;
    }}
    .ds-section-head__subtitle {{
        font-size: {Type.body[0]};
        color: {Color.text_secondary};
        margin: 0;
        line-height: {Type.body[3]};
        max-width: 720px;
    }}

    /* Feature card */
    .ds-feature {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.lg};
        padding: {Space.xl};
        box-shadow: {Shadow.xs};
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        height: 100%;
        min-height: 240px;
        display: flex;
        flex-direction: column;
    }}
    .ds-feature:hover {{
        transform: translateY(-2px);
        box-shadow: {Shadow.md};
        border-color: {Color.border_strong};
    }}
    .ds-feature__icon {{
        width: 42px;
        height: 42px;
        border-radius: {Radius.md};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        margin-bottom: {Space.lg};
    }}
    .ds-feature__title {{
        font-family: {Type.sans};
        font-size: {Type.h3[0]};
        font-weight: {Type.h3[1]};
        color: {Color.text_primary};
        margin: 0 0 {Space.sm} 0;
    }}
    .ds-feature__desc {{
        font-size: {Type.small[0]};
        color: {Color.text_secondary};
        line-height: {Type.small[3]};
        margin: 0;
        flex: 1;
    }}
    .ds-feature__tag {{
        align-self: flex-start;
        display: inline-block;
        background: transparent;
        color: {Color.text_muted};
        font-size: {Type.tiny[0]};
        font-weight: {Type.tiny[1]};
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0;
        border: none;
        margin-top: {Space.lg};
    }}

    /* Metric card */
    .ds-metric {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.md};
        padding: {Space.lg} {Space.xl};
        box-shadow: {Shadow.xs};
    }}
    .ds-metric--success  {{ border-color: {Color.success}; }}
    .ds-metric--warning  {{ border-color: {Color.warning}; }}
    .ds-metric--danger   {{ border-color: {Color.danger};  }}
    .ds-metric__label {{
        font-size: {Type.tiny[0]};
        font-weight: {Type.tiny[1]};
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {Color.text_muted};
        margin: 0 0 {Space.xs} 0;
    }}
    .ds-metric__value {{
        font-family: {Type.serif};
        font-size: 1.9rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: {Color.text_primary};
        margin: 0;
    }}
    .ds-metric--success .ds-metric__value {{ color: {Color.success}; }}
    .ds-metric--warning .ds-metric__value {{ color: {Color.warning}; }}
    .ds-metric--danger  .ds-metric__value {{ color: {Color.danger};  }}
    .ds-metric__hint {{
        font-size: {Type.small[0]};
        color: {Color.text_muted};
        margin: {Space.xs} 0 0 0;
    }}

    /* Result card */
    .ds-result {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.lg};
        padding: {Space.x2} {Space.xl};
        box-shadow: {Shadow.sm};
        margin: {Space.lg} 0;
        text-align: center;
    }}
    .ds-result--success {{ border-top: 3px solid {Color.success}; }}
    .ds-result--warning {{ border-top: 3px solid {Color.warning}; }}
    .ds-result--danger  {{ border-top: 3px solid {Color.danger};  }}
    .ds-result--info    {{ border-top: 3px solid {Color.brand};   }}
    .ds-result--neutral {{ border-top: 3px solid {Color.border_strong}; }}
    .ds-result__title {{
        font-family: {Type.serif};
        font-size: 2.0rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0;
        color: {Color.text_primary};
    }}
    .ds-result--success .ds-result__title {{ color: {Color.success}; }}
    .ds-result--warning .ds-result__title {{ color: {Color.warning}; }}
    .ds-result--danger  .ds-result__title {{ color: {Color.danger};  }}
    .ds-result--info    .ds-result__title {{ color: {Color.brand};   }}
    .ds-result__subtitle {{
        color: {Color.text_secondary};
        margin: {Space.sm} 0 0 0;
        font-size: {Type.body[0]};
    }}

    /* Callout */
    .ds-callout {{
        display: flex;
        gap: {Space.md};
        align-items: flex-start;
        padding: {Space.md} {Space.lg};
        border-radius: {Radius.md};
        border-left: 3px solid {Color.brand};
        background: {Color.brand_soft};
        margin: {Space.md} 0;
    }}
    .ds-callout--success {{ border-left-color: {Color.success}; background: {Color.success_soft}; }}
    .ds-callout--warning {{ border-left-color: {Color.warning}; background: {Color.warning_soft}; }}
    .ds-callout--danger  {{ border-left-color: {Color.danger};  background: {Color.danger_soft};  }}
    .ds-callout--info    {{ border-left-color: {Color.info};    background: {Color.info_soft};    }}
    .ds-callout__icon {{ font-size: 1.05rem; line-height: 1.4; margin-top: 1px; }}
    .ds-callout__body {{ flex: 1; }}
    .ds-callout__title {{
        font-size: {Type.body[0]};
        font-weight: 600;
        margin: 0 0 2px 0;
        color: {Color.text_primary};
    }}
    .ds-callout__text {{
        font-size: {Type.small[0]};
        color: {Color.text_secondary};
        margin: 0;
        line-height: {Type.small[3]};
    }}

    /* Status badge */
    .ds-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: {Radius.pill};
        font-size: {Type.tiny[0]};
        font-weight: {Type.tiny[1]};
        letter-spacing: 0.02em;
        border: 1px solid transparent;
        white-space: nowrap;
    }}
    .ds-badge--success {{ color: {Color.success}; background: {Color.success_soft}; border-color: {Color.success_soft}; }}
    .ds-badge--warning {{ color: {Color.warning}; background: {Color.warning_soft}; border-color: {Color.warning_soft}; }}
    .ds-badge--danger  {{ color: {Color.danger};  background: {Color.danger_soft};  border-color: {Color.danger_soft};  }}
    .ds-badge--info    {{ color: {Color.info};    background: {Color.info_soft};    border-color: {Color.info_soft};    }}
    .ds-badge--brand   {{ color: {Color.brand};   background: {Color.brand_soft};   border-color: {Color.brand_tint};  }}
    .ds-badge--neutral {{ color: {Color.text_secondary}; background: {Color.bg_alt}; border-color: {Color.border}; }}

    /* Disclaimer */
    .ds-disclaimer {{
        background: {Color.warning_soft};
        border-left: 3px solid {Color.warning};
        border-radius: {Radius.sm};
        padding: {Space.md} {Space.lg};
        color: {Color.warning};
        font-size: {Type.small[0]};
        margin-top: {Space.xl};
    }}

    /* Divider */
    .ds-divider {{
        height: 1px;
        background: {Color.border};
        margin: {Space.x2} 0;
        border: none;
    }}

    /* Footer */
    .ds-footer {{
        text-align: center;
        padding: {Space.x2} 0 {Space.md};
        color: {Color.text_muted};
        font-size: {Type.small[0]};
        border-top: 1px solid {Color.border};
        margin-top: {Space.x3};
    }}
    .ds-footer a {{ color: {Color.brand}; text-decoration: none; font-weight: 500; }}
    .ds-footer a:hover {{ text-decoration: underline; }}

    /* ════════════════════════════════════════════════════════════
       Legacy class overrides — re-skin older markup for cream theme
       so any section still using the old class names renders
       correctly. These will be migrated one by one and removed.
       ════════════════════════════════════════════════════════════ */

    /* Legacy hero (old dark) */
    .hero {{
        margin: {Space.sm} 0 {Space.x2};
        padding: 0 0 {Space.x2};
        border: none;
        border-bottom: 1px solid {Color.border};
        background: transparent;
    }}
    .hero h1 {{
        font-family: {Type.serif};
        font-size: {Type.display[0]};
        font-weight: 600;
        color: {Color.text_primary};
        margin: 0 0 {Space.md};
        letter-spacing: -0.02em;
    }}
    .hero p {{ color: {Color.text_secondary}; font-size: 1.1rem; margin: 0; max-width: 680px; }}
    .hero::before {{ display: none; }}

    /* Legacy stat pills */
    .stat-row {{ display: flex; gap: {Space.xl}; margin-top: {Space.xl}; flex-wrap: wrap; }}
    .stat-pill {{
        background: transparent;
        border: none;
        padding: 0;
        color: {Color.text_secondary};
        font-size: {Type.tiny[0]};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .stat-pill strong {{
        display: block;
        font-family: {Type.serif};
        font-size: 1.5rem;
        font-weight: 600;
        color: {Color.text_primary};
        text-transform: none;
        letter-spacing: -0.01em;
    }}

    /* Legacy feature-card → clean white card */
    .feature-card {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.lg};
        padding: {Space.xl};
        box-shadow: {Shadow.xs};
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        height: 100%;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .feature-card:hover {{ transform: translateY(-2px); box-shadow: {Shadow.md}; border-color: {Color.border_strong}; }}
    .feature-icon {{ width: 42px; height: 42px; border-radius: {Radius.md}; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-bottom: {Space.lg}; }}
    .icon-ecg      {{ background: rgba(193, 95, 60, 0.12); }}
    .icon-xray     {{ background: rgba(46, 110, 140, 0.12); }}
    .icon-risk     {{ background: rgba(61, 125, 86, 0.12); }}
    .icon-cbc      {{ background: rgba(125, 84, 151, 0.12); }}
    .icon-diabetes {{ background: rgba(165, 116, 27, 0.12); }}
    .icon-lipid    {{ background: rgba(53, 107, 124, 0.12); }}
    .icon-kidney   {{ background: rgba(154, 59, 59, 0.12); }}
    .icon-lab      {{ background: rgba(73, 91, 121, 0.12); }}

    .feature-card h3 {{ font-family: {Type.sans}; font-size: {Type.h3[0]}; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.sm} 0; }}
    .feature-card p  {{ color: {Color.text_secondary}; font-size: {Type.small[0]}; line-height: {Type.small[3]}; margin: 0; }}
    .feature-tag {{
        display: inline-block;
        background: transparent;
        color: {Color.text_muted};
        font-size: {Type.tiny[0]};
        font-weight: 500;
        padding: 0;
        border: none;
        margin-top: {Space.lg};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    /* Legacy metric card */
    .metric-card {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-left: 3px solid {Color.brand};
        border-radius: {Radius.md};
        padding: {Space.lg} {Space.xl};
        box-shadow: {Shadow.xs};
        margin-bottom: {Space.sm};
    }}
    .metric-value {{ font-family: {Type.serif}; font-size: 1.9rem; font-weight: 600; color: {Color.text_primary}; margin: 0; letter-spacing: -0.01em; }}
    .metric-label {{ font-size: {Type.tiny[0]}; color: {Color.text_muted}; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin-bottom: 4px; }}
    .risk-high   {{ border-left-color: {Color.danger}  !important; }}
    .risk-high .metric-value {{ color: {Color.danger}; }}
    .risk-medium {{ border-left-color: {Color.warning} !important; }}
    .risk-medium .metric-value {{ color: {Color.warning}; }}
    .risk-low    {{ border-left-color: {Color.success} !important; }}
    .risk-low .metric-value {{ color: {Color.success}; }}

    /* Legacy section header (already replaced on most pages) */
    .section-header {{
        font-family: {Type.serif};
        font-size: {Type.h1[0]};
        font-weight: 600;
        color: {Color.text_primary};
        margin: 0 0 {Space.sm};
        letter-spacing: -0.02em;
    }}
    .section-sub {{
        color: {Color.text_secondary};
        font-size: {Type.body[0]};
        margin: 0 0 {Space.xl};
        line-height: {Type.body[3]};
    }}

    /* Legacy result-box */
    .result-box {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-top: 3px solid {Color.brand};
        color: {Color.text_primary};
        padding: {Space.x2} {Space.xl};
        border-radius: {Radius.lg};
        text-align: center;
        margin: {Space.lg} 0;
        box-shadow: {Shadow.sm};
    }}
    .result-box h2 {{ font-family: {Type.serif}; color: {Color.brand}; margin: 0; font-size: 2rem; font-weight: 600; letter-spacing: -0.02em; }}
    .result-box p  {{ color: {Color.text_secondary}; margin: {Space.sm} 0 0; font-size: {Type.body[0]}; }}

    /* Legacy info-card */
    .info-card {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.md};
        padding: {Space.lg} {Space.xl};
        box-shadow: {Shadow.xs};
        margin-bottom: {Space.sm};
    }}
    .info-card h4 {{ font-size: 1rem; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.xs}; }}
    .info-card p  {{ font-size: {Type.small[0]}; color: {Color.text_secondary}; margin: 0; line-height: {Type.small[3]}; }}

    /* Legacy upload zone */
    .upload-zone {{
        background: {Color.surface};
        border: 1.5px dashed {Color.border_strong};
        border-radius: {Radius.lg};
        padding: {Space.x2} {Space.xl};
        text-align: center;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    .upload-zone:hover {{ border-color: {Color.brand}; background: {Color.brand_soft}; }}
    .upload-icon {{ font-size: 2rem; margin-bottom: {Space.sm}; color: {Color.text_muted}; }}
    .upload-text {{ color: {Color.text_secondary}; font-size: {Type.body[0]}; }}

    /* Legacy flag badges */
    .flag-critical {{ background: {Color.danger_soft};  color: {Color.danger};  padding: 2px 10px; border-radius: {Radius.pill}; font-size: {Type.tiny[0]}; font-weight: 600; }}
    .flag-high     {{ background: {Color.warning_soft}; color: {Color.warning}; padding: 2px 10px; border-radius: {Radius.pill}; font-size: {Type.tiny[0]}; font-weight: 600; }}
    .flag-low      {{ background: {Color.info_soft};    color: {Color.info};    padding: 2px 10px; border-radius: {Radius.pill}; font-size: {Type.tiny[0]}; font-weight: 600; }}
    .flag-normal   {{ background: {Color.success_soft}; color: {Color.success}; padding: 2px 10px; border-radius: {Radius.pill}; font-size: {Type.tiny[0]}; font-weight: 600; }}

    /* Legacy lab-table / ckd-grid */
    .lab-table, .ckd-grid {{ width: 100%; border-collapse: collapse; font-size: {Type.small[0]}; background: {Color.surface}; border-radius: {Radius.md}; overflow: hidden; border: 1px solid {Color.border}; }}
    .lab-table th, .ckd-grid th {{ background: {Color.bg_alt}; padding: 10px 14px; text-align: left; font-weight: 600; color: {Color.text_muted}; border-bottom: 1px solid {Color.border}; text-transform: uppercase; letter-spacing: 0.04em; font-size: {Type.tiny[0]}; }}
    .lab-table td, .ckd-grid td {{ padding: 10px 14px; border-bottom: 1px solid {Color.border}; color: {Color.text_primary}; }}
    .ckd-grid th {{ text-align: center; }}
    .ckd-grid td {{ text-align: center; }}
    .ckd-cell {{ padding: 8px; border-radius: {Radius.sm}; text-align: center; font-weight: 600; font-size: 0.82rem; }}

    /* Legacy step-card / step-item / benefit-card */
    .step-card, .step-item, .benefit-card {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.lg};
        padding: {Space.xl};
        box-shadow: {Shadow.xs};
    }}
    .step-card h3  {{ font-size: 1.1rem; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.xs}; }}
    .step-card p   {{ color: {Color.text_secondary}; font-size: {Type.small[0]}; margin: 0 0 {Space.lg}; }}
    .benefit-card h4 {{ font-size: 1rem; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.xs}; }}
    .benefit-card p  {{ color: {Color.text_secondary}; font-size: {Type.small[0]}; margin: 0; line-height: {Type.small[3]}; }}

    /* Legacy progress bar */
    .progress-bar-bg   {{ background: {Color.border}; border-radius: {Radius.pill}; height: 6px; margin-bottom: {Space.xl}; overflow: hidden; }}
    .progress-bar-fill {{ background: {Color.brand}; height: 100%; border-radius: {Radius.pill}; transition: width 0.3s ease; }}

    /* Legacy status badges */
    .status-badge {{ display: inline-block; padding: 3px 10px; border-radius: {Radius.pill}; font-size: {Type.tiny[0]}; font-weight: 600; margin: 2px; }}
    .status-loaded {{ background: {Color.success_soft}; color: {Color.success} !important; border: 1px solid {Color.success_soft}; }}
    .status-demo   {{ background: {Color.warning_soft}; color: {Color.warning} !important; border: 1px solid {Color.warning_soft}; }}

    /* Legacy footer (original) */
    .footer {{ text-align: center; padding: {Space.x2} 0 {Space.md}; color: {Color.text_muted}; font-size: {Type.small[0]}; border-top: 1px solid {Color.border}; margin-top: {Space.x3}; }}
    .footer a {{ color: {Color.brand}; text-decoration: none; font-weight: 500; }}
    .footer-divider {{ display: none; }}

    /* Responsive */
    @media (max-width: 768px) {{
        .block-container {{ padding: {Space.lg}; }}
        .ds-hero__title, .section-header {{ font-size: 2rem; }}
        .ds-feature, .feature-card {{ min-height: auto; }}
    }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# Components
# ════════════════════════════════════════════════════════════════════

def hero(title: str, subtitle: str, eyebrow: str | None = None,
         stats: list[tuple[str, str]] | None = None) -> None:
    """Editorial-style hero: eyebrow, serif title, subtitle, optional
    inline stats. No gradient, no glow blob.

    stats: list of (label, value) tuples, e.g. [("Datasets", "3")]
    """
    parts = ['<section class="ds-hero">']
    if eyebrow:
        parts.append(f'<div class="ds-hero__eyebrow">{eyebrow}</div>')
    parts.append(f'<h1 class="ds-hero__title">{title}</h1>')
    parts.append(f'<p class="ds-hero__subtitle">{subtitle}</p>')
    if stats:
        stat_html = "".join(
            f'<div class="ds-stat">'
            f'  <div class="ds-stat__value">{v}</div>'
            f'  <div class="ds-stat__label">{k}</div>'
            f'</div>'
            for k, v in stats
        )
        parts.append(f'<div class="ds-hero__stats">{stat_html}</div>')
    parts.append('</section>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None,
                   eyebrow: str | None = None) -> None:
    parts = ['<div class="ds-section-head">']
    if eyebrow:
        parts.append(f'<div class="ds-section-head__eyebrow">{eyebrow}</div>')
    parts.append(f'<h1 class="ds-section-head__title">{title}</h1>')
    if subtitle:
        parts.append(f'<p class="ds-section-head__subtitle">{subtitle}</p>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def feature_card(icon: str, title: str, desc: str,
                 tag: str | None = None,
                 icon_bg: str | None = None) -> None:
    bg_style = f"background: {icon_bg};" if icon_bg else f"background: {Color.brand_soft};"
    tag_html = f'<span class="ds-feature__tag">{tag}</span>' if tag else ""
    st.markdown(f"""
        <div class="ds-feature">
            <div class="ds-feature__icon" style="{bg_style}">{icon}</div>
            <h3 class="ds-feature__title">{title}</h3>
            <p class="ds-feature__desc">{desc}</p>
            {tag_html}
        </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, hint: str | None = None,
                variant: str = "default") -> None:
    variant_class = "" if variant == "default" else f" ds-metric--{variant}"
    hint_html = f'<p class="ds-metric__hint">{hint}</p>' if hint else ""
    st.markdown(f"""
        <div class="ds-metric{variant_class}">
            <p class="ds-metric__label">{label}</p>
            <p class="ds-metric__value">{value}</p>
            {hint_html}
        </div>
    """, unsafe_allow_html=True)


def result_card(title: str, subtitle: str | None = None,
                variant: str = "info") -> None:
    sub = f'<p class="ds-result__subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
        <div class="ds-result ds-result--{variant}">
            <h2 class="ds-result__title">{title}</h2>
            {sub}
        </div>
    """, unsafe_allow_html=True)


def info_callout(title: str, body: str, variant: str = "info",
                 icon: str | None = None) -> None:
    icon_map = {"info": "ℹ", "success": "✓", "warning": "⚠", "danger": "⚠"}
    icon = icon or icon_map.get(variant, "ℹ")
    st.markdown(f"""
        <div class="ds-callout ds-callout--{variant}">
            <div class="ds-callout__icon">{icon}</div>
            <div class="ds-callout__body">
                <p class="ds-callout__title">{title}</p>
                <p class="ds-callout__text">{body}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, variant: str = "neutral") -> str:
    return f'<span class="ds-badge ds-badge--{variant}">{text}</span>'


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="ds-disclaimer">{text}</div>', unsafe_allow_html=True)


def page_divider() -> None:
    st.markdown('<hr class="ds-divider" />', unsafe_allow_html=True)


def footer(text: str = "Healthcare AI Prediction Portal") -> None:
    st.markdown(f"""
        <div class="ds-footer">
            {text} · Abdullah Abdul Sami · Northwestern University
        </div>
    """, unsafe_allow_html=True)
