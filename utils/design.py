"""
design.py — Healthcare AI Portal design system.

Aesthetic target: Linear / Vercel dashboard applied to a clinical
product. Premium dark, restrained, typographic. NOT a Streamlit
template, NOT editorial magazine.

Design rules
------------
  * ONE primary accent: #00B4A6 (clinical teal). Never paired with
    another bright color.
  * 9-step gray scale (gray_50 … gray_950) — no arbitrary grays.
  * Typography: Inter for UI, JetBrains Mono for numeric output.
    Tight letter-spacing on headings, generous line-height on body.
  * Spacing on an 8px grid. Every margin/padding comes from Space.
  * NO emoji as UI. All icons are inline stroked SVGs (lucide-style).
  * NO gradients. NO glows. NO glassmorphism.
  * Shadows capped at 0 1px 2px rgba(0,0,0,0.1). Elevation via
    background + border, not drop shadows.
  * Buttons: primary (filled), secondary (bordered), ghost (text).
  * 150ms transitions on all interactive state changes.

Public API
----------
    inject_global_css()
    hero(title, subtitle, eyebrow=None, stats=None)
    section_header(title, subtitle=None, eyebrow=None)
    feature_card(icon_key, title, desc, tag=None)
    metric_card(label, value, hint=None, variant='default')
    result_card(title, subtitle=None, variant='default')
    info_callout(title, body, variant='info')
    status_badge(text, variant='neutral') -> str
    risk_bar(label, pct, variant='default')
    disclaimer(text)
    page_divider()
    footer(text=...)
    icon(key, size=16, color=None) -> str  # returns inline <svg>
"""

from __future__ import annotations

import streamlit as st


# ════════════════════════════════════════════════════════════════════
# Tokens
# ════════════════════════════════════════════════════════════════════

class Color:
    # Canvas (never pure black)
    bg             = "#0A0A0B"   # app background
    bg_elevated    = "#131316"   # card / surface
    bg_overlay     = "#17171B"   # hover / nested surface
    bg_input       = "#0F0F11"   # inputs (slightly recessed)

    # 9-step gray scale (zinc-ish, warm)
    gray_50   = "#FAFAFA"
    gray_100  = "#F4F4F5"
    gray_200  = "#E4E4E7"
    gray_300  = "#D4D4D8"
    gray_400  = "#A1A1AA"
    gray_500  = "#71717A"
    gray_600  = "#52525B"
    gray_700  = "#3F3F46"
    gray_800  = "#27272A"
    gray_900  = "#18181B"
    gray_950  = "#09090B"

    # Text (semantic aliases on the scale)
    text_primary   = "#FAFAFA"   # gray_50
    text_secondary = "#A1A1AA"   # gray_400
    text_muted     = "#71717A"   # gray_500
    text_disabled  = "#52525B"   # gray_600

    # Borders — low-opacity white for subtle definition
    border         = "rgba(255, 255, 255, 0.06)"
    border_strong  = "rgba(255, 255, 255, 0.10)"
    border_hover   = "rgba(255, 255, 255, 0.14)"
    border_focus   = "#4E2A84"

    # Brand — Northwestern Purple
    #   accent        : official NU Purple (fills, buttons, borders)
    #   accent_bright : NU Light Purple (for text-on-dark — deep purple
    #                   reads too dim as a text color on #0A0A0B)
    accent         = "#4E2A84"   # NU Purple
    accent_bright  = "#836EAA"   # NU Light Purple — text highlights, active states
    accent_hover   = "#5F37A0"
    accent_active  = "#3D1E6E"
    accent_soft    = "rgba(78, 42, 132, 0.14)"
    accent_tint    = "rgba(78, 42, 132, 0.22)"
    accent_ring    = "rgba(78, 42, 132, 0.30)"

    # Semantic — muted so they don't scream (dark-theme tuned)
    success      = "#22C55E"
    success_soft = "rgba(34, 197, 94, 0.10)"
    warning      = "#EAB308"
    warning_soft = "rgba(234, 179, 8, 0.10)"
    danger       = "#EF4444"
    danger_soft  = "rgba(239, 68, 68, 0.10)"
    info         = "#3B82F6"
    info_soft    = "rgba(59, 130, 246, 0.10)"


class Space:
    # 8px grid (plus 4 for micro-adjustments)
    x0 = "0"
    x1 = "4px"
    x2 = "8px"
    x3 = "12px"
    x4 = "16px"
    x5 = "20px"
    x6 = "24px"
    x7 = "32px"
    x8 = "40px"
    x9 = "48px"
    x10 = "64px"
    x11 = "80px"


class Radius:
    sm   = "4px"
    md   = "6px"
    lg   = "8px"
    xl   = "12px"
    pill = "9999px"


class Shadow:
    # HARD rule: nothing heavier than 0 1px 2px rgba(0,0,0,0.1)
    none = "none"
    xs   = "0 1px 2px rgba(0, 0, 0, 0.1)"
    ring = "0 0 0 3px rgba(0, 180, 166, 0.2)"


class Type:
    sans = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    mono = "'JetBrains Mono', 'SF Mono', ui-monospace, 'Cascadia Code', Menlo, monospace"

    # (size, weight, letter-spacing, line-height)
    display = ("2.75rem", "600", "-0.035em", "1.1")   # hero title
    h1      = ("1.75rem", "600", "-0.02em",  "1.25")
    h2      = ("1.25rem", "600", "-0.015em", "1.3")
    h3      = ("1.0rem",  "600", "-0.01em",  "1.4")
    body    = ("0.938rem","400", "0",        "1.65")   # 15px
    small   = ("0.875rem","400", "0",        "1.55")   # 14px
    tiny    = ("0.75rem", "500", "0.01em",   "1.4")    # 12px
    eyebrow = ("0.72rem", "600", "0.12em",   "1.3")    # 11.5px uppercase


# ════════════════════════════════════════════════════════════════════
# Lucide-style SVG icons (stroke, 2px, round caps)
# ════════════════════════════════════════════════════════════════════
#
# Keep ALL icon paths in one dict. Use icon() to render them.
# Paths pulled from lucide.dev — 24×24 viewBox, stroke currentColor.

_ICONS: dict[str, str] = {
    "home":
        '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<path d="M9 22V12h6v10"/>',
    "activity":
        '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.5.5 0 0 1-.96 0L9.24 2.18a.5.5 0 0 0-.96 0l-2.35 8.36A2 2 0 0 1 4 12H2"/>',
    "heart-pulse":
        '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>'
        '<path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "scan":
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/>'
        '<path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
        '<path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<path d="M7 12h10"/>',
    "clipboard-check":
        '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/>'
        '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
        '<path d="m9 14 2 2 4-4"/>',
    "droplet":
        '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>',
    "flask-conical":
        '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/>'
        '<path d="M8.5 2h7"/>'
        '<path d="M7 16h10"/>',
    "activity-square":
        '<rect width="18" height="18" x="3" y="3" rx="2"/>'
        '<path d="M17 12h-2l-2 5-2-10-2 5H7"/>',
    "file-text":
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v6h6"/>'
        '<path d="M16 13H8"/>'
        '<path d="M16 17H8"/>'
        '<path d="M10 9H8"/>',
    "sparkles":
        '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>'
        '<path d="M5 3v4"/>'
        '<path d="M19 17v4"/>'
        '<path d="M3 5h4"/>'
        '<path d="M17 19h4"/>',
    "shield-check":
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>',
    "search":
        '<circle cx="11" cy="11" r="8"/>'
        '<path d="m21 21-4.3-4.3"/>',
    "mouse-pointer-click":
        '<path d="M14 4.1 12 6"/>'
        '<path d="m5.1 8-2.9-.8"/>'
        '<path d="m6 12-1.9 2"/>'
        '<path d="M7.2 2.2 8 5.1"/>'
        '<path d="M9.037 9.69a.498.498 0 0 1 .653-.653l11 4.5a.5.5 0 0 1-.074.949l-4.349 1.041a1 1 0 0 0-.74.739l-1.04 4.35a.5.5 0 0 1-.95.074z"/>',
    "upload":
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/>'
        '<line x1="12" y1="3" x2="12" y2="15"/>',
    "bar-chart":
        '<line x1="12" y1="20" x2="12" y2="10"/>'
        '<line x1="18" y1="20" x2="18" y2="4"/>'
        '<line x1="6"  y1="20" x2="6"  y2="16"/>',
    "zap":
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "microscope":
        '<path d="M6 18h8"/>'
        '<path d="M3 22h18"/>'
        '<path d="M14 22a7 7 0 1 0 0-14h-1"/>'
        '<path d="M9 14h2"/>'
        '<path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/>'
        '<path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>',
    "test-tube":
        '<path d="M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5c-1.4 0-2.5-1.1-2.5-2.5V2"/>'
        '<path d="M8.5 2h7"/>'
        '<path d="M14.5 16h-5"/>',
    "book":
        '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
    "lock":
        '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>'
        '<path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "check-circle":
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="m9 12 2 2 4-4"/>',
    "alert-triangle":
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        '<path d="M12 9v4"/>'
        '<path d="M12 17h.01"/>',
    "info":
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M12 16v-4"/>'
        '<path d="M12 8h.01"/>',
    "x-circle":
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="m15 9-6 6"/>'
        '<path d="m9 9 6 6"/>',
    "arrow-right":
        '<path d="M5 12h14"/>'
        '<path d="m12 5 7 7-7 7"/>',
}


def icon(key: str, size: int = 16, color: str | None = None,
         stroke_width: float = 2) -> str:
    """Return an inline <svg> string. Use inside st.markdown(...).

    size: px. color: hex; defaults to currentColor so it inherits from
    the surrounding text color (important for hover/active states).
    """
    body = _ICONS.get(key, _ICONS["info"])
    stroke = color or "currentColor"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;">'
        f'{body}'
        f'</svg>'
    )


# ════════════════════════════════════════════════════════════════════
# Global CSS
# ════════════════════════════════════════════════════════════════════

def inject_global_css() -> None:
    st.markdown(f"""
<style>
    /* ── Fonts ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, .stApp, [class*="css"] {{
        font-family: {Type.sans};
        color: {Color.text_primary};
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        font-feature-settings: "cv02", "cv03", "cv04", "cv11";
    }}

    /* ── Canvas ─────────────────────────────────────────────────── */
    .stApp, .main {{ background: {Color.bg}; }}
    .block-container {{
        padding-top: {Space.x7};
        padding-bottom: {Space.x10};
        max-width: 1160px;
    }}

    /* Kill Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ── Scrollbars (dark, minimal) ─────────────────────────────── */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: {Color.gray_800};
        border-radius: {Radius.pill};
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {Color.gray_700}; }}

    /* ── Typography ─────────────────────────────────────────────── */
    h1 {{
        font-size: {Type.h1[0]}; font-weight: {Type.h1[1]};
        letter-spacing: {Type.h1[2]}; line-height: {Type.h1[3]};
        color: {Color.text_primary};
    }}
    h2 {{
        font-size: {Type.h2[0]}; font-weight: {Type.h2[1]};
        letter-spacing: {Type.h2[2]}; line-height: {Type.h2[3]};
        color: {Color.text_primary};
    }}
    h3 {{
        font-size: {Type.h3[0]}; font-weight: {Type.h3[1]};
        letter-spacing: {Type.h3[2]}; line-height: {Type.h3[3]};
        color: {Color.text_primary};
    }}
    p, li, span, div {{ color: {Color.text_primary}; line-height: {Type.body[3]}; }}
    code, pre, .ds-mono {{ font-family: {Type.mono}; }}

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {Color.bg_elevated} !important;
        border-right: 1px solid {Color.border} !important;
        min-width: 240px !important;
        max-width: 240px !important;
        width: 240px !important;
        transform: none !important;
        position: relative !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        width: 240px !important;
        padding: {Space.x6} {Space.x4} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{ display: none !important; }}
    [data-testid="stSidebar"] * {{ color: {Color.text_primary}; }}
    [data-testid="stSidebar"] hr {{
        border: none !important;
        border-top: 1px solid {Color.border} !important;
        margin: {Space.x4} -{Space.x4} !important;
    }}

    /* Sidebar radio → nav list */
    [data-testid="stSidebar"] .stRadio > div {{ gap: 2px !important; }}
    [data-testid="stSidebar"] .stRadio label {{
        display: flex !important;
        align-items: center !important;
        gap: {Space.x3} !important;
        padding: {Space.x2} {Space.x3} !important;
        border-radius: {Radius.md} !important;
        color: {Color.text_secondary} !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        border-left: 2px solid transparent !important;
        transition: background 150ms ease, color 150ms ease, border-color 150ms ease;
        cursor: pointer;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: {Color.bg_overlay} !important;
        color: {Color.text_primary} !important;
    }}
    [data-testid="stSidebar"] .stRadio input:checked + div,
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {{
        background: {Color.accent_soft} !important;
        color: {Color.accent_bright} !important;
        border-left-color: {Color.accent} !important;
    }}
    /* Hide radio dots */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:first-child {{
        display: none !important;
    }}

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: {Color.text_muted} !important;
        font-size: 0.78rem !important;
    }}

    /* ── Buttons ────────────────────────────────────────────────── */
    .stButton > button {{
        background: {Color.accent} !important;
        color: {Color.gray_950} !important;
        border: 1px solid {Color.accent} !important;
        border-radius: {Radius.md} !important;
        padding: 8px 16px !important;
        font-family: {Type.sans} !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        letter-spacing: -0.005em !important;
        box-shadow: {Shadow.xs} !important;
        transition: background 150ms ease, border-color 150ms ease,
                    transform 150ms ease, box-shadow 150ms ease !important;
    }}
    .stButton > button:hover {{
        background: {Color.accent_hover} !important;
        border-color: {Color.accent_hover} !important;
    }}
    .stButton > button:active {{
        background: {Color.accent_active} !important;
        transform: translateY(0);
    }}
    .stButton > button:focus {{
        box-shadow: {Shadow.ring} !important;
        outline: none !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {Color.text_primary} !important;
        border: 1px solid {Color.border_strong} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: {Color.bg_overlay} !important;
        border-color: {Color.border_hover} !important;
    }}

    /* ── Inputs ─────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stTextArea textarea,
    .stDateInput input {{
        background: {Color.bg_input} !important;
        border: 1px solid {Color.border_strong} !important;
        border-radius: {Radius.md} !important;
        color: {Color.text_primary} !important;
        font-family: {Type.sans} !important;
        font-size: 0.9rem !important;
        padding: 8px 12px !important;
        transition: border-color 150ms ease, box-shadow 150ms ease;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {{
        border-color: {Color.accent} !important;
        box-shadow: {Shadow.ring} !important;
        outline: none !important;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        background: {Color.bg_input} !important;
        border: 1px solid {Color.border_strong} !important;
        border-radius: {Radius.md} !important;
        color: {Color.text_primary} !important;
        font-size: 0.9rem !important;
    }}
    .stSelectbox [data-baseweb="select"] > div:hover {{
        border-color: {Color.border_hover} !important;
    }}
    label, .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stRadio > label, .stCheckbox > label {{
        color: {Color.text_secondary} !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em;
    }}

    /* Sliders */
    .stSlider [data-baseweb="slider"] > div {{
        background: {Color.gray_800} !important;
    }}
    .stSlider [data-baseweb="slider"] > div > div > div {{
        background: {Color.accent} !important;
    }}
    .stSlider [role="slider"] {{
        background: {Color.accent} !important;
        border: 2px solid {Color.bg_elevated} !important;
        box-shadow: {Shadow.xs} !important;
        width: 16px !important;
        height: 16px !important;
    }}

    /* Checkbox */
    .stCheckbox [data-baseweb="checkbox"] label > div:first-child {{
        border-color: {Color.border_strong} !important;
        background: {Color.bg_input} !important;
        border-radius: {Radius.sm} !important;
    }}
    .stCheckbox [data-baseweb="checkbox"] [data-checked="true"] {{
        background: {Color.accent} !important;
        border-color: {Color.accent} !important;
    }}

    /* ── Tabs (underline style) ─────────────────────────────────── */
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
        padding: 12px 16px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        color: {Color.text_secondary} !important;
        border-bottom: 2px solid transparent !important;
        margin-bottom: -1px !important;
        transition: color 150ms ease, border-color 150ms ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {Color.text_primary} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {Color.accent_bright} !important;
        border-bottom-color: {Color.accent} !important;
    }}

    /* ── Tables / dataframes ────────────────────────────────────── */
    .stDataFrame, .dataframe {{
        border-radius: {Radius.lg};
        border: 1px solid {Color.border};
        overflow: hidden;
        box-shadow: none;
    }}
    .stDataFrame table th {{
        background: {Color.bg_overlay} !important;
        color: {Color.text_muted} !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid {Color.border} !important;
    }}
    .stDataFrame table td {{
        background: {Color.bg_elevated} !important;
        color: {Color.text_primary} !important;
        font-family: {Type.mono} !important;
        font-size: 0.82rem !important;
        border-bottom: 1px solid {Color.border} !important;
    }}

    /* ── File uploader ──────────────────────────────────────────── */
    [data-testid="stFileUploader"] section {{
        background: {Color.bg_elevated} !important;
        border: 1px dashed {Color.border_strong} !important;
        border-radius: {Radius.lg} !important;
        padding: {Space.x7} !important;
        transition: border-color 150ms ease, background 150ms ease;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: {Color.accent} !important;
        background: {Color.accent_soft} !important;
    }}

    /* ── Native alerts — restyled ───────────────────────────────── */
    [data-testid="stAlert"] {{
        background: {Color.bg_elevated} !important;
        border: 1px solid {Color.border} !important;
        border-left-width: 3px !important;
        border-radius: {Radius.md} !important;
        box-shadow: none !important;
        padding: {Space.x3} {Space.x4} !important;
        color: {Color.text_primary} !important;
    }}

    /* ── Column equal-height ────────────────────────────────────── */
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
       Design-system components
       ════════════════════════════════════════════════════════════ */

    /* Hero */
    .ds-hero {{
        margin: {Space.x2} 0 {Space.x7};
        padding: 0 0 {Space.x7};
        border-bottom: 1px solid {Color.border};
    }}
    .ds-hero__eyebrow {{
        display: flex; align-items: center; gap: {Space.x2};
        font-size: {Type.eyebrow[0]}; font-weight: {Type.eyebrow[1]};
        letter-spacing: {Type.eyebrow[2]}; text-transform: uppercase;
        color: {Color.accent_bright};
        margin: 0 0 {Space.x4};
    }}
    .ds-hero__title {{
        font-size: {Type.display[0]}; font-weight: {Type.display[1]};
        letter-spacing: {Type.display[2]}; line-height: {Type.display[3]};
        color: {Color.text_primary}; margin: 0 0 {Space.x4}; max-width: 820px;
    }}
    .ds-hero__subtitle {{
        font-size: 1.063rem; font-weight: 400;
        color: {Color.text_secondary}; margin: 0; max-width: 720px;
        line-height: 1.6;
    }}
    .ds-hero__stats {{
        display: flex; gap: {Space.x9}; margin-top: {Space.x7}; flex-wrap: wrap;
    }}
    .ds-stat__value {{
        font-family: {Type.mono}; font-size: 1.6rem; font-weight: 600;
        color: {Color.text_primary}; letter-spacing: -0.01em; line-height: 1;
    }}
    .ds-stat__label {{
        font-size: {Type.tiny[0]}; font-weight: {Type.tiny[1]};
        color: {Color.text_muted}; text-transform: uppercase;
        letter-spacing: 0.08em; margin-top: {Space.x2};
    }}

    /* Section header */
    .ds-section-head {{ margin: 0 0 {Space.x7}; }}
    .ds-section-head__eyebrow {{
        display: flex; align-items: center; gap: {Space.x2};
        font-size: {Type.eyebrow[0]}; font-weight: {Type.eyebrow[1]};
        letter-spacing: {Type.eyebrow[2]}; text-transform: uppercase;
        color: {Color.accent_bright}; margin: 0 0 {Space.x3};
    }}
    .ds-section-head__title {{
        font-size: {Type.h1[0]}; font-weight: {Type.h1[1]};
        letter-spacing: {Type.h1[2]}; line-height: {Type.h1[3]};
        color: {Color.text_primary}; margin: 0 0 {Space.x2};
    }}
    .ds-section-head__subtitle {{
        font-size: {Type.body[0]}; color: {Color.text_secondary};
        margin: 0; line-height: {Type.body[3]}; max-width: 720px;
    }}

    /* Feature card */
    .ds-feature {{
        background: {Color.bg_elevated};
        border: 1px solid {Color.border};
        border-radius: {Radius.xl};
        padding: {Space.x7};
        transition: border-color 150ms ease, transform 150ms ease, background 150ms ease;
        height: 100%;
        min-height: 220px;
        display: flex;
        flex-direction: column;
    }}
    .ds-feature:hover {{
        border-color: {Color.border_hover};
        background: {Color.bg_overlay};
    }}
    .ds-feature__icon {{
        width: 36px; height: 36px;
        border-radius: {Radius.md};
        background: {Color.accent_soft};
        color: {Color.accent_bright};
        display: flex; align-items: center; justify-content: center;
        margin-bottom: {Space.x5};
    }}
    .ds-feature__title {{
        font-size: {Type.h3[0]}; font-weight: {Type.h3[1]};
        color: {Color.text_primary};
        margin: 0 0 {Space.x2};
    }}
    .ds-feature__desc {{
        font-size: {Type.small[0]};
        color: {Color.text_secondary};
        line-height: {Type.small[3]};
        margin: 0; flex: 1;
    }}
    .ds-feature__tag {{
        align-self: flex-start;
        font-size: {Type.tiny[0]};
        font-weight: {Type.tiny[1]};
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: {Color.text_muted};
        margin-top: {Space.x5};
    }}

    /* Metric card */
    .ds-metric {{
        background: {Color.bg_elevated};
        border: 1px solid {Color.border};
        border-radius: {Radius.lg};
        padding: {Space.x5} {Space.x6};
    }}
    .ds-metric--success  {{ border-color: {Color.success_soft}; }}
    .ds-metric--warning  {{ border-color: {Color.warning_soft}; }}
    .ds-metric--danger   {{ border-color: {Color.danger_soft};  }}
    .ds-metric__label {{
        font-size: {Type.tiny[0]}; font-weight: {Type.tiny[1]};
        letter-spacing: 0.06em; text-transform: uppercase;
        color: {Color.text_muted}; margin: 0 0 {Space.x2};
    }}
    .ds-metric__value {{
        font-family: {Type.mono}; font-size: 1.75rem; font-weight: 600;
        letter-spacing: -0.01em; color: {Color.text_primary}; margin: 0;
    }}
    .ds-metric--success .ds-metric__value {{ color: {Color.success}; }}
    .ds-metric--warning .ds-metric__value {{ color: {Color.warning}; }}
    .ds-metric--danger  .ds-metric__value {{ color: {Color.danger};  }}
    .ds-metric__hint {{
        font-size: {Type.small[0]};
        color: {Color.text_muted}; margin: {Space.x1} 0 0;
        font-family: {Type.mono};
    }}

    /* Result card */
    .ds-result {{
        background: {Color.bg_elevated};
        border: 1px solid {Color.border};
        border-radius: {Radius.xl};
        padding: {Space.x7} {Space.x7};
        margin: {Space.x5} 0;
        position: relative;
        overflow: hidden;
    }}
    .ds-result::before {{
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 2px;
        background: {Color.accent};
    }}
    .ds-result--success::before {{ background: {Color.success}; }}
    .ds-result--warning::before {{ background: {Color.warning}; }}
    .ds-result--danger::before  {{ background: {Color.danger};  }}
    .ds-result__label {{
        font-size: {Type.tiny[0]}; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: {Color.text_muted}; margin: 0 0 {Space.x3};
    }}
    .ds-result__title {{
        font-size: 2.25rem; font-weight: 600;
        letter-spacing: -0.025em; line-height: 1.1;
        margin: 0; color: {Color.text_primary};
    }}
    .ds-result--success .ds-result__title {{ color: {Color.success}; }}
    .ds-result--warning .ds-result__title {{ color: {Color.warning}; }}
    .ds-result--danger  .ds-result__title {{ color: {Color.danger};  }}
    .ds-result__subtitle {{
        color: {Color.text_secondary};
        margin: {Space.x3} 0 0;
        font-size: {Type.body[0]};
        font-family: {Type.mono};
    }}

    /* Callout */
    .ds-callout {{
        display: flex; gap: {Space.x3}; align-items: flex-start;
        padding: {Space.x3} {Space.x5};
        border-radius: {Radius.md};
        border: 1px solid {Color.border};
        border-left: 3px solid {Color.accent};
        background: {Color.accent_soft};
        margin: {Space.x3} 0;
    }}
    .ds-callout--success {{ border-left-color: {Color.success}; background: {Color.success_soft}; }}
    .ds-callout--warning {{ border-left-color: {Color.warning}; background: {Color.warning_soft}; }}
    .ds-callout--danger  {{ border-left-color: {Color.danger};  background: {Color.danger_soft};  }}
    .ds-callout--info    {{ border-left-color: {Color.info};    background: {Color.info_soft};    }}
    .ds-callout__icon {{
        color: {Color.accent_bright}; flex-shrink: 0; margin-top: 1px;
    }}
    .ds-callout--success .ds-callout__icon {{ color: {Color.success}; }}
    .ds-callout--warning .ds-callout__icon {{ color: {Color.warning}; }}
    .ds-callout--danger  .ds-callout__icon {{ color: {Color.danger};  }}
    .ds-callout--info    .ds-callout__icon {{ color: {Color.info};    }}
    .ds-callout__title {{
        font-size: 0.9rem; font-weight: 600; margin: 0 0 2px;
        color: {Color.text_primary};
    }}
    .ds-callout__text {{
        font-size: {Type.small[0]}; color: {Color.text_secondary};
        margin: 0; line-height: {Type.small[3]};
    }}

    /* Status badge */
    .ds-badge {{
        display: inline-flex; align-items: center; gap: 4px;
        padding: 2px 8px; border-radius: {Radius.pill};
        font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.02em; border: 1px solid transparent;
    }}
    .ds-badge--success {{ color: {Color.success}; background: {Color.success_soft}; border-color: {Color.success_soft}; }}
    .ds-badge--warning {{ color: {Color.warning}; background: {Color.warning_soft}; border-color: {Color.warning_soft}; }}
    .ds-badge--danger  {{ color: {Color.danger};  background: {Color.danger_soft};  border-color: {Color.danger_soft};  }}
    .ds-badge--info    {{ color: {Color.info};    background: {Color.info_soft};    border-color: {Color.info_soft};    }}
    .ds-badge--accent  {{ color: {Color.accent_bright};  background: {Color.accent_soft};  border-color: {Color.accent_tint}; }}
    .ds-badge--neutral {{ color: {Color.text_secondary}; background: {Color.bg_overlay}; border-color: {Color.border}; }}

    /* Risk bar */
    .ds-riskbar {{
        padding: {Space.x3} 0;
    }}
    .ds-riskbar__header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: {Space.x2};
    }}
    .ds-riskbar__label {{
        font-size: 0.82rem; font-weight: 500; color: {Color.text_secondary};
    }}
    .ds-riskbar__value {{
        font-family: {Type.mono}; font-size: 0.82rem; font-weight: 600;
        color: {Color.text_primary};
    }}
    .ds-riskbar__track {{
        background: {Color.gray_800};
        height: 6px; border-radius: {Radius.pill}; overflow: hidden;
    }}
    .ds-riskbar__fill {{
        height: 100%; background: {Color.accent};
        transition: width 300ms ease;
        border-radius: {Radius.pill};
    }}
    .ds-riskbar--success .ds-riskbar__fill {{ background: {Color.success}; }}
    .ds-riskbar--warning .ds-riskbar__fill {{ background: {Color.warning}; }}
    .ds-riskbar--danger  .ds-riskbar__fill {{ background: {Color.danger};  }}

    /* Breadcrumb */
    .ds-breadcrumb {{
        font-size: 0.78rem;
        color: {Color.text_muted};
        margin: 0 0 {Space.x3};
        letter-spacing: 0.01em;
    }}
    .ds-breadcrumb__current {{
        color: {Color.text_primary};
        font-weight: 500;
    }}

    /* Upload hint */
    .ds-upload-hint {{
        background: {Color.bg_elevated};
        border: 1px dashed {Color.border_strong};
        border-radius: {Radius.lg};
        padding: {Space.x7} {Space.x6};
        margin-bottom: {Space.x4};
        text-align: center;
        transition: border-color 150ms ease, background 150ms ease;
    }}
    .ds-upload-hint:hover {{
        border-color: {Color.accent};
        background: {Color.accent_soft};
    }}
    .ds-upload-hint__icon {{
        color: {Color.text_muted};
        margin-bottom: {Space.x3};
        display: flex; justify-content: center;
    }}
    .ds-upload-hint__title {{
        font-size: {Type.body[0]};
        font-weight: 500;
        color: {Color.text_primary};
        margin: 0 0 {Space.x1};
    }}
    .ds-upload-hint__desc {{
        font-size: {Type.small[0]};
        color: {Color.text_secondary};
        margin: 0;
    }}
    .ds-upload-hint__formats {{
        font-family: {Type.mono};
        font-size: 0.78rem;
        color: {Color.text_muted};
        margin: {Space.x2} 0 0;
        letter-spacing: 0.02em;
    }}

    /* Disclaimer */
    .ds-disclaimer {{
        background: {Color.warning_soft};
        border: 1px solid {Color.warning_soft};
        border-left: 3px solid {Color.warning};
        border-radius: {Radius.md};
        padding: {Space.x3} {Space.x5};
        color: {Color.warning};
        font-size: {Type.small[0]};
        margin-top: {Space.x6};
    }}

    /* Divider */
    .ds-divider {{
        height: 1px; background: {Color.border};
        margin: {Space.x7} 0; border: none;
    }}

    /* Footer */
    .ds-footer {{
        text-align: center;
        padding: {Space.x7} 0 {Space.x4};
        color: {Color.text_muted};
        font-size: {Type.small[0]};
        border-top: 1px solid {Color.border};
        margin-top: {Space.x9};
    }}
    .ds-footer a {{ color: {Color.accent}; text-decoration: none; font-weight: 500; }}
    .ds-footer a:hover {{ text-decoration: underline; }}

    /* ════════════════════════════════════════════════════════════
       Legacy class shims — keep older markup rendering in the new
       dark theme until each section is refactored.
       ════════════════════════════════════════════════════════════ */

    .hero {{ margin: {Space.x2} 0 {Space.x7}; padding: 0 0 {Space.x7}; border: none; border-bottom: 1px solid {Color.border}; background: transparent; }}
    .hero h1 {{ font-size: {Type.display[0]}; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.x4}; letter-spacing: -0.025em; line-height: 1.1; }}
    .hero p  {{ color: {Color.text_secondary}; font-size: 1.063rem; margin: 0; max-width: 720px; line-height: 1.6; }}
    .hero::before {{ display: none; }}

    .stat-row {{ display: flex; gap: {Space.x9}; margin-top: {Space.x7}; flex-wrap: wrap; }}
    .stat-pill {{ background: transparent; border: none; padding: 0; color: {Color.text_muted}; font-size: {Type.tiny[0]}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat-pill strong {{ display: block; font-family: {Type.mono}; font-size: 1.4rem; font-weight: 600; color: {Color.text_primary}; text-transform: none; letter-spacing: -0.01em; margin-bottom: {Space.x2}; }}

    .feature-card {{ background: {Color.bg_elevated}; border: 1px solid {Color.border}; border-radius: {Radius.xl}; padding: {Space.x7}; box-shadow: none; transition: border-color 150ms ease, background 150ms ease; height: 100%; min-height: 240px; display: flex; flex-direction: column; justify-content: space-between; }}
    .feature-card:hover {{ border-color: {Color.border_hover}; background: {Color.bg_overlay}; transform: none; box-shadow: none; }}
    .feature-icon {{ width: 36px; height: 36px; border-radius: {Radius.md}; display: flex; align-items: center; justify-content: center; background: {Color.accent_soft}; color: {Color.accent_bright}; margin-bottom: {Space.x5}; font-size: 1.2rem; }}
    .icon-ecg, .icon-xray, .icon-risk, .icon-cbc, .icon-diabetes, .icon-lipid, .icon-kidney, .icon-lab {{ background: {Color.accent_soft}; color: {Color.accent_bright}; }}
    .feature-card h3 {{ font-size: {Type.h3[0]}; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.x2}; }}
    .feature-card p  {{ color: {Color.text_secondary}; font-size: {Type.small[0]}; line-height: {Type.small[3]}; margin: 0; }}
    .feature-tag {{ display: inline-block; background: transparent; color: {Color.text_muted}; font-size: {Type.tiny[0]}; font-weight: 500; padding: 0; border: none; margin-top: {Space.x5}; text-transform: uppercase; letter-spacing: 0.04em; }}

    .metric-card {{ background: {Color.bg_elevated}; border: 1px solid {Color.border}; border-left: 2px solid {Color.accent}; border-radius: {Radius.lg}; padding: {Space.x5} {Space.x6}; box-shadow: none; margin-bottom: {Space.x2}; }}
    .metric-value {{ font-family: {Type.mono}; font-size: 1.75rem; font-weight: 600; color: {Color.text_primary}; margin: 0; letter-spacing: -0.01em; }}
    .metric-label {{ font-size: {Type.tiny[0]}; color: {Color.text_muted}; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin-bottom: 4px; }}
    .risk-high   {{ border-left-color: {Color.danger}  !important; }}
    .risk-high .metric-value {{ color: {Color.danger}; }}
    .risk-medium {{ border-left-color: {Color.warning} !important; }}
    .risk-medium .metric-value {{ color: {Color.warning}; }}
    .risk-low    {{ border-left-color: {Color.success} !important; }}
    .risk-low .metric-value {{ color: {Color.success}; }}

    .section-header {{ font-size: {Type.h1[0]}; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.x2}; letter-spacing: -0.02em; line-height: 1.25; }}
    .section-sub {{ color: {Color.text_secondary}; font-size: {Type.body[0]}; margin: 0 0 {Space.x7}; line-height: {Type.body[3]}; }}

    .result-box {{ background: {Color.bg_elevated}; border: 1px solid {Color.border}; position: relative; color: {Color.text_primary}; padding: {Space.x7}; border-radius: {Radius.xl}; text-align: left; margin: {Space.x5} 0; box-shadow: none; overflow: hidden; }}
    .result-box::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: {Color.accent}; }}
    .result-box h2 {{ color: {Color.text_primary}; margin: 0; font-size: 2.25rem; font-weight: 600; letter-spacing: -0.025em; }}
    .result-box p  {{ color: {Color.text_secondary}; margin: {Space.x3} 0 0; font-size: {Type.body[0]}; font-family: {Type.mono}; }}

    .info-card {{ background: {Color.bg_elevated}; border: 1px solid {Color.border}; border-radius: {Radius.lg}; padding: {Space.x5} {Space.x6}; box-shadow: none; margin-bottom: {Space.x2}; }}
    .info-card h4 {{ font-size: 0.95rem; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.x1}; }}
    .info-card p  {{ font-size: {Type.small[0]}; color: {Color.text_secondary}; margin: 0; line-height: {Type.small[3]}; }}

    .upload-zone {{ background: {Color.bg_elevated}; border: 1px dashed {Color.border_strong}; border-radius: {Radius.lg}; padding: {Space.x7} {Space.x6}; text-align: center; transition: border-color 150ms ease, background 150ms ease; }}
    .upload-zone:hover {{ border-color: {Color.accent}; background: {Color.accent_soft}; }}
    .upload-icon {{ font-size: 1.6rem; margin-bottom: {Space.x2}; color: {Color.text_muted}; }}
    .upload-text {{ color: {Color.text_secondary}; font-size: {Type.body[0]}; }}

    .flag-critical {{ background: {Color.danger_soft};  color: {Color.danger};  padding: 2px 8px; border-radius: {Radius.pill}; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em; }}
    .flag-high     {{ background: {Color.warning_soft}; color: {Color.warning}; padding: 2px 8px; border-radius: {Radius.pill}; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em; }}
    .flag-low      {{ background: {Color.info_soft};    color: {Color.info};    padding: 2px 8px; border-radius: {Radius.pill}; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em; }}
    .flag-normal   {{ background: {Color.success_soft}; color: {Color.success}; padding: 2px 8px; border-radius: {Radius.pill}; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em; }}

    .lab-table, .ckd-grid {{ width: 100%; border-collapse: collapse; font-family: {Type.mono}; font-size: 0.82rem; background: {Color.bg_elevated}; border-radius: {Radius.lg}; overflow: hidden; border: 1px solid {Color.border}; }}
    .lab-table th, .ckd-grid th {{ background: {Color.bg_overlay}; padding: 10px 14px; text-align: left; font-weight: 600; color: {Color.text_muted}; border-bottom: 1px solid {Color.border}; text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.7rem; font-family: {Type.sans}; }}
    .lab-table td, .ckd-grid td {{ padding: 10px 14px; border-bottom: 1px solid {Color.border}; color: {Color.text_primary}; }}
    .ckd-grid th {{ text-align: center; }}
    .ckd-grid td {{ text-align: center; }}
    .ckd-cell {{ padding: 8px; border-radius: {Radius.sm}; text-align: center; font-weight: 600; font-size: 0.78rem; }}

    .step-card, .step-item, .benefit-card {{ background: {Color.bg_elevated}; border: 1px solid {Color.border}; border-radius: {Radius.lg}; padding: {Space.x6}; box-shadow: none; }}
    .step-card h3  {{ font-size: 1rem; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.x1}; }}
    .step-card p   {{ color: {Color.text_secondary}; font-size: {Type.small[0]}; margin: 0 0 {Space.x5}; }}
    .benefit-card h4 {{ font-size: 0.95rem; font-weight: 600; color: {Color.text_primary}; margin: 0 0 {Space.x1}; }}
    .benefit-card p  {{ color: {Color.text_secondary}; font-size: {Type.small[0]}; margin: 0; line-height: {Type.small[3]}; }}

    .progress-bar-bg   {{ background: {Color.gray_800}; border-radius: {Radius.pill}; height: 6px; margin-bottom: {Space.x6}; overflow: hidden; }}
    .progress-bar-fill {{ background: {Color.accent}; height: 100%; border-radius: {Radius.pill}; transition: width 300ms ease; }}

    .status-badge {{ display: inline-block; padding: 2px 8px; border-radius: {Radius.pill}; font-size: 0.7rem; font-weight: 600; margin: 2px; letter-spacing: 0.02em; }}
    .status-loaded {{ background: {Color.success_soft}; color: {Color.success} !important; border: 1px solid {Color.success_soft}; }}
    .status-demo   {{ background: {Color.warning_soft}; color: {Color.warning} !important; border: 1px solid {Color.warning_soft}; }}

    .footer {{ text-align: center; padding: {Space.x7} 0 {Space.x4}; color: {Color.text_muted}; font-size: {Type.small[0]}; border-top: 1px solid {Color.border}; margin-top: {Space.x9}; }}
    .footer a {{ color: {Color.accent}; text-decoration: none; font-weight: 500; }}
    .footer-divider {{ display: none; }}

    /* Responsive */
    @media (max-width: 768px) {{
        .block-container {{ padding: {Space.x4}; }}
        .ds-hero__title, .section-header {{ font-size: 1.75rem; }}
        .ds-feature, .feature-card {{ min-height: auto; }}
    }}

    /* Numeric cells get mono automatically */
    .ds-mono, .mono {{ font-family: {Type.mono} !important; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# Components
# ════════════════════════════════════════════════════════════════════

def hero(title: str, subtitle: str, eyebrow: str | None = None,
         eyebrow_icon: str | None = None,
         stats: list[tuple[str, str]] | None = None) -> None:
    parts = ['<section class="ds-hero">']
    if eyebrow:
        icon_html = icon(eyebrow_icon, size=14) if eyebrow_icon else ""
        parts.append(
            f'<div class="ds-hero__eyebrow">{icon_html}{eyebrow}</div>'
        )
    parts.append(f'<h1 class="ds-hero__title">{title}</h1>')
    parts.append(f'<p class="ds-hero__subtitle">{subtitle}</p>')
    if stats:
        stat_html = "".join(
            f'<div class="ds-stat">'
            f'<div class="ds-stat__value">{v}</div>'
            f'<div class="ds-stat__label">{k}</div>'
            f'</div>'
            for k, v in stats
        )
        parts.append(f'<div class="ds-hero__stats">{stat_html}</div>')
    parts.append('</section>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None,
                   eyebrow: str | None = None,
                   eyebrow_icon: str | None = None) -> None:
    parts = ['<div class="ds-section-head">']
    if eyebrow:
        icon_html = icon(eyebrow_icon, size=14) if eyebrow_icon else ""
        parts.append(f'<div class="ds-section-head__eyebrow">{icon_html}{eyebrow}</div>')
    parts.append(f'<h1 class="ds-section-head__title">{title}</h1>')
    if subtitle:
        parts.append(f'<p class="ds-section-head__subtitle">{subtitle}</p>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def feature_card(icon_key: str, title: str, desc: str,
                 tag: str | None = None) -> None:
    """icon_key refers to keys in _ICONS (e.g. 'heart-pulse'). Never emoji."""
    tag_html = f'<span class="ds-feature__tag">{tag}</span>' if tag else ""
    icon_html = icon(icon_key, size=20)
    st.markdown(f"""
        <div class="ds-feature">
            <div class="ds-feature__icon">{icon_html}</div>
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
                label: str | None = None,
                variant: str = "default") -> None:
    """Big prediction output. `label` (small eyebrow, e.g. 'PREDICTED DIAGNOSIS'),
    `title` (the value), `subtitle` (supporting detail, uses mono)."""
    variant_class = "" if variant == "default" else f" ds-result--{variant}"
    label_html = f'<p class="ds-result__label">{label}</p>' if label else ""
    sub_html = f'<p class="ds-result__subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
        <div class="ds-result{variant_class}">
            {label_html}
            <h2 class="ds-result__title">{title}</h2>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def info_callout(title: str, body: str, variant: str = "info") -> None:
    icon_map = {"info": "info", "success": "check-circle",
                "warning": "alert-triangle", "danger": "x-circle"}
    icon_html = icon(icon_map.get(variant, "info"), size=18)
    st.markdown(f"""
        <div class="ds-callout ds-callout--{variant}">
            <div class="ds-callout__icon">{icon_html}</div>
            <div class="ds-callout__body">
                <p class="ds-callout__title">{title}</p>
                <p class="ds-callout__text">{body}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, variant: str = "neutral") -> str:
    return f'<span class="ds-badge ds-badge--{variant}">{text}</span>'


def risk_bar(label: str, pct: float, variant: str = "default") -> None:
    """Colored bar (0-100%). variant ∈ default|success|warning|danger."""
    pct_clamped = max(0, min(100, pct))
    variant_class = "" if variant == "default" else f" ds-riskbar--{variant}"
    st.markdown(f"""
        <div class="ds-riskbar{variant_class}">
            <div class="ds-riskbar__header">
                <span class="ds-riskbar__label">{label}</span>
                <span class="ds-riskbar__value">{pct_clamped:.1f}%</span>
            </div>
            <div class="ds-riskbar__track">
                <div class="ds-riskbar__fill" style="width: {pct_clamped}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def breadcrumb(current: str, root: str = "Home") -> None:
    """Small muted crumb ABOVE the section header: 'Home / Current'."""
    st.markdown(
        f'<p class="ds-breadcrumb">{root}  /  '
        f'<span class="ds-breadcrumb__current">{current}</span></p>',
        unsafe_allow_html=True,
    )


def upload_hint(icon_key: str, title: str, desc: str,
                formats: str | None = None) -> None:
    """Rendered as the 'drop zone' copy above a st.file_uploader.
    Replaces the old dark-div-with-emoji pattern."""
    icon_html = icon(icon_key, size=24)
    formats_html = (
        f'<p class="ds-upload-hint__formats">{formats}</p>'
        if formats else ""
    )
    st.markdown(f"""
        <div class="ds-upload-hint">
            <div class="ds-upload-hint__icon">{icon_html}</div>
            <p class="ds-upload-hint__title">{title}</p>
            <p class="ds-upload-hint__desc">{desc}</p>
            {formats_html}
        </div>
    """, unsafe_allow_html=True)


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
