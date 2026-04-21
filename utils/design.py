"""
design.py — single source of truth for the Healthcare AI Portal look & feel.

Everything visual comes from here:
  * tokens (colors, spacing, radii, shadows, typography)
  * reusable CSS (inject_global_css)
  * reusable components (hero, section_header, feature_card,
    metric_card, result_card, info_callout, status_badge,
    disclaimer, upload_zone, footer, page_divider)

Usage (at the top of streamlit_app.py, right after set_page_config):
    from utils.design import inject_global_css
    inject_global_css()

    # then in sections:
    from utils.design import section_header, feature_card
    section_header("Heart / ECG", "Upload an ECG signal for analysis.")
    feature_card(icon="🫀", title="Heart Risk", desc="…", tag="Cardiology")

Design principles
-----------------
* ONE primary accent colour (sky-400 / #38BDF8). No more dual blues.
* Spacing / radius / shadow / typography each come from a small scale —
  nothing is picked ad-hoc.
* Every card/surface uses the same elevation and border rules so the
  whole app feels like one product, not eleven.
* Semantic colours (success / warning / danger / info) are used ONLY
  for status signals — never for decoration.
"""

from __future__ import annotations

import streamlit as st


# ════════════════════════════════════════════════════════════════════
# Design tokens
# ════════════════════════════════════════════════════════════════════

class Color:
    # Surfaces
    bg          = "#0B1221"   # app background (slightly deeper than slate-900)
    surface     = "#131B2E"   # cards / elevated surfaces
    surface_alt = "#0F172A"   # inputs, nested surfaces
    overlay     = "rgba(15, 23, 42, 0.72)"

    # Borders
    border        = "#1F2A3E"
    border_strong = "#2D3B56"
    border_focus  = "#38BDF8"

    # Text
    text_primary   = "#F1F5F9"
    text_secondary = "#A7B0C2"
    text_muted     = "#6B7588"
    text_inverse   = "#0B1221"

    # Brand (sky-400 / sky-500)
    brand         = "#38BDF8"
    brand_hover   = "#0EA5E9"
    brand_soft    = "rgba(56, 189, 248, 0.10)"
    brand_ring    = "rgba(56, 189, 248, 0.25)"

    # Semantic
    success      = "#34D399"
    success_soft = "rgba(52, 211, 153, 0.12)"
    warning      = "#FBBF24"
    warning_soft = "rgba(251, 191, 36, 0.12)"
    danger       = "#F87171"
    danger_soft  = "rgba(248, 113, 113, 0.12)"
    info         = "#60A5FA"
    info_soft    = "rgba(96, 165, 250, 0.12)"

    # Per-feature accents (used only for icon tiles on the home grid)
    accent_ecg      = "#F87171"
    accent_xray     = "#38BDF8"
    accent_risk     = "#34D399"
    accent_cbc      = "#C084FC"
    accent_diabetes = "#FBBF24"
    accent_lipid    = "#22D3EE"
    accent_kidney   = "#FB7185"
    accent_lab      = "#818CF8"


class Space:
    # 4-point scale
    xs  = "4px"
    sm  = "8px"
    md  = "12px"
    lg  = "16px"
    xl  = "24px"
    x2  = "32px"
    x3  = "48px"
    x4  = "64px"


class Radius:
    sm   = "6px"
    md   = "10px"
    lg   = "14px"
    xl   = "20px"
    pill = "9999px"


class Shadow:
    sm   = "0 1px 2px rgba(0, 0, 0, 0.25)"
    md   = "0 4px 12px rgba(0, 0, 0, 0.28)"
    lg   = "0 10px 32px rgba(0, 0, 0, 0.35)"
    glow = "0 10px 32px rgba(56, 189, 248, 0.18)"


class Type:
    # Font family — Inter everywhere
    family  = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    # Scale
    display = ("2.4rem", "800", "-0.03em", "1.2")   # size, weight, tracking, leading
    h1      = ("1.8rem", "800", "-0.02em", "1.25")
    h2      = ("1.35rem","700", "-0.01em", "1.3")
    h3      = ("1.05rem","700", "0",       "1.4")
    body    = ("0.95rem","400", "0",       "1.6")
    small   = ("0.85rem","400", "0",       "1.55")
    caption = ("0.72rem","600", "0.08em",  "1.4")   # uppercase


# ════════════════════════════════════════════════════════════════════
# Global CSS — inject once at app start
# ════════════════════════════════════════════════════════════════════

def inject_global_css() -> None:
    """Inject the full design-system stylesheet. Call this once, right
    after st.set_page_config()."""
    st.markdown(f"""
<style>
    /* ── Fonts ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: {Type.family};
        color: {Color.text_primary};
    }}

    /* ── Page canvas ────────────────────────────────────────────── */
    .main, .stApp {{
        background: {Color.bg};
    }}
    .block-container {{
        padding-top: {Space.x2};
        padding-bottom: {Space.x3};
        max-width: 1200px;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ── Sidebar — always open, fixed width ─────────────────────── */
    [data-testid="stSidebar"] {{
        background: {Color.surface} !important;
        border-right: 1px solid {Color.border} !important;
        min-width: 280px !important;
        max-width: 280px !important;
        width: 280px !important;
        transform: none !important;
        position: relative !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{ width: 280px !important; }}
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{ display: none !important; }}
    [data-testid="stSidebar"] * {{ color: {Color.text_secondary} !important; }}
    [data-testid="stSidebar"] hr {{ border-color: {Color.border} !important; }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {Color.text_secondary} !important;
        font-weight: 500;
        padding: {Space.sm} {Space.xs};
        border-radius: {Radius.md};
        transition: background 0.18s ease, color 0.18s ease;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: {Color.brand_soft};
        color: {Color.text_primary} !important;
    }}

    /* ── Typography primitives ──────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {{ color: {Color.text_primary}; letter-spacing: -0.01em; }}
    p, li, span {{ color: {Color.text_primary}; }}

    /* ── Streamlit widgets — unify look ─────────────────────────── */
    .stButton > button {{
        background: {Color.brand} !important;
        color: {Color.text_inverse} !important;
        border: none !important;
        border-radius: {Radius.md} !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
    }}
    .stButton > button:hover {{
        background: {Color.brand_hover} !important;
        box-shadow: {Shadow.glow} !important;
        transform: translateY(-1px);
    }}
    .stButton > button:focus {{
        box-shadow: 0 0 0 3px {Color.brand_ring} !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {Color.text_primary} !important;
        border: 1px solid {Color.border_strong} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: {Color.brand} !important;
        background: {Color.brand_soft} !important;
        box-shadow: none !important;
    }}

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stTextArea textarea {{
        background: {Color.surface_alt} !important;
        border: 1px solid {Color.border_strong} !important;
        border-radius: {Radius.md} !important;
        color: {Color.text_primary} !important;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {{
        border-color: {Color.brand} !important;
        box-shadow: 0 0 0 3px {Color.brand_ring} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: {Space.xs};
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.md};
        padding: {Space.xs};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: {Radius.sm};
        padding: 10px 18px;
        font-weight: 600;
        color: {Color.text_secondary};
    }}
    .stTabs [aria-selected="true"] {{
        background: {Color.brand} !important;
        color: {Color.text_inverse} !important;
    }}

    /* Dataframes */
    .stDataFrame, .dataframe {{
        border-radius: {Radius.md};
        border: 1px solid {Color.border};
        overflow: hidden;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] section {{
        background: {Color.surface} !important;
        border: 2px dashed {Color.border_strong} !important;
        border-radius: {Radius.lg} !important;
        transition: border-color 0.18s ease, background 0.18s ease;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: {Color.brand} !important;
        background: {Color.brand_soft} !important;
    }}

    /* Column alignment — equal-height cards inside columns */
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

    /* ══════════════════════════════════════════════════════════════
       Components — every class below is used by a helper in
       utils/design.py. Don't reference them directly from pages.
       ══════════════════════════════════════════════════════════ */

    /* Hero */
    .ds-hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, {Color.bg} 0%, {Color.surface} 45%, {Color.bg} 100%);
        border: 1px solid {Color.border_strong};
        border-radius: {Radius.xl};
        padding: {Space.x3} {Space.x2};
        margin-bottom: {Space.x2};
    }}
    .ds-hero::before {{
        content: '';
        position: absolute;
        top: -40%;
        right: -10%;
        width: 480px;
        height: 480px;
        background: radial-gradient(circle, {Color.brand_soft} 0%, transparent 65%);
        pointer-events: none;
    }}
    .ds-hero__eyebrow {{
        font-size: {Type.caption[0]};
        font-weight: {Type.caption[1]};
        letter-spacing: {Type.caption[2]};
        text-transform: uppercase;
        color: {Color.brand};
        margin: 0 0 {Space.sm} 0;
    }}
    .ds-hero__title {{
        font-size: {Type.display[0]};
        font-weight: {Type.display[1]};
        letter-spacing: {Type.display[2]};
        line-height: {Type.display[3]};
        color: {Color.text_primary};
        margin: 0 0 {Space.sm} 0;
    }}
    .ds-hero__subtitle {{
        font-size: 1.05rem;
        color: {Color.text_secondary};
        margin: 0;
        max-width: 640px;
        line-height: 1.6;
    }}
    .ds-hero__stats {{
        display: flex;
        gap: {Space.sm};
        margin-top: {Space.xl};
        flex-wrap: wrap;
    }}
    .ds-pill {{
        background: {Color.brand_soft};
        border: 1px solid {Color.border_strong};
        border-radius: {Radius.pill};
        padding: 6px 16px;
        color: {Color.text_secondary};
        font-size: {Type.small[0]};
        font-weight: 500;
    }}
    .ds-pill strong {{ color: {Color.brand}; font-weight: 700; }}

    /* Section header */
    .ds-section-head {{ margin: 0 0 {Space.xl} 0; }}
    .ds-section-head__eyebrow {{
        font-size: {Type.caption[0]};
        font-weight: {Type.caption[1]};
        letter-spacing: {Type.caption[2]};
        text-transform: uppercase;
        color: {Color.brand};
        margin: 0 0 {Space.xs} 0;
    }}
    .ds-section-head__title {{
        font-size: {Type.h1[0]};
        font-weight: {Type.h1[1]};
        letter-spacing: {Type.h1[2]};
        color: {Color.text_primary};
        margin: 0 0 {Space.sm} 0;
    }}
    .ds-section-head__subtitle {{
        font-size: {Type.body[0]};
        color: {Color.text_secondary};
        margin: 0;
        line-height: {Type.body[3]};
        max-width: 780px;
    }}

    /* Feature card (home grid) */
    .ds-feature {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-radius: {Radius.lg};
        padding: {Space.xl} {Space.lg} {Space.lg};
        box-shadow: {Shadow.sm};
        transition: transform 0.22s cubic-bezier(0.4, 0, 0.2, 1),
                    box-shadow 0.22s ease,
                    border-color 0.22s ease;
        height: 100%;
        min-height: 300px;
        display: flex;
        flex-direction: column;
    }}
    .ds-feature:hover {{
        transform: translateY(-3px);
        border-color: {Color.brand};
        box-shadow: {Shadow.glow};
    }}
    .ds-feature__icon {{
        width: 48px;
        height: 48px;
        border-radius: {Radius.md};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: {Space.md};
    }}
    .ds-feature__title {{
        font-size: {Type.h3[0]};
        font-weight: {Type.h3[1]};
        color: {Color.text_primary};
        margin: 0 0 {Space.xs} 0;
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
        background: {Color.surface_alt};
        color: {Color.brand};
        font-size: {Type.caption[0]};
        font-weight: {Type.caption[1]};
        letter-spacing: {Type.caption[2]};
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: {Radius.pill};
        margin-top: {Space.md};
        border: 1px solid {Color.border_strong};
    }}

    /* Metric card */
    .ds-metric {{
        background: {Color.surface};
        border: 1px solid {Color.border};
        border-left: 3px solid {Color.brand};
        border-radius: {Radius.md};
        padding: {Space.lg} {Space.lg};
        box-shadow: {Shadow.sm};
    }}
    .ds-metric--success  {{ border-left-color: {Color.success}; }}
    .ds-metric--warning  {{ border-left-color: {Color.warning}; }}
    .ds-metric--danger   {{ border-left-color: {Color.danger};  }}
    .ds-metric__label {{
        font-size: {Type.caption[0]};
        font-weight: {Type.caption[1]};
        letter-spacing: {Type.caption[2]};
        text-transform: uppercase;
        color: {Color.text_muted};
        margin: 0 0 {Space.xs} 0;
    }}
    .ds-metric__value {{
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
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
        background: linear-gradient(135deg, {Color.surface} 0%, {Color.bg} 100%);
        border: 1px solid {Color.border_strong};
        border-radius: {Radius.lg};
        padding: {Space.xl};
        text-align: center;
        box-shadow: {Shadow.lg};
        margin: {Space.lg} 0;
    }}
    .ds-result--success {{ border-top: 3px solid {Color.success}; }}
    .ds-result--warning {{ border-top: 3px solid {Color.warning}; }}
    .ds-result--danger  {{ border-top: 3px solid {Color.danger};  }}
    .ds-result--info    {{ border-top: 3px solid {Color.brand};   }}
    .ds-result__title {{
        font-size: 1.8rem;
        font-weight: 800;
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

    /* Info callout */
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
    .ds-callout__icon {{ font-size: 1.2rem; line-height: 1; margin-top: 2px; }}
    .ds-callout__body {{ flex: 1; }}
    .ds-callout__title {{
        font-size: {Type.body[0]};
        font-weight: 700;
        margin: 0 0 2px 0;
        color: {Color.text_primary};
    }}
    .ds-callout__text {{
        font-size: {Type.small[0]};
        color: {Color.text_secondary};
        margin: 0;
        line-height: {Type.small[3]};
    }}

    /* Status badge (pill) */
    .ds-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: {Radius.pill};
        font-size: {Type.caption[0]};
        font-weight: {Type.caption[1]};
        letter-spacing: {Type.caption[2]};
        text-transform: uppercase;
        border: 1px solid transparent;
    }}
    .ds-badge--success {{ color: {Color.success}; background: {Color.success_soft}; border-color: {Color.success_soft}; }}
    .ds-badge--warning {{ color: {Color.warning}; background: {Color.warning_soft}; border-color: {Color.warning_soft}; }}
    .ds-badge--danger  {{ color: {Color.danger};  background: {Color.danger_soft};  border-color: {Color.danger_soft};  }}
    .ds-badge--info    {{ color: {Color.info};    background: {Color.info_soft};    border-color: {Color.info_soft};    }}
    .ds-badge--neutral {{ color: {Color.text_secondary}; background: {Color.surface_alt}; border-color: {Color.border_strong}; }}

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
        margin: {Space.xl} 0;
        border: none;
    }}

    /* Footer */
    .ds-footer {{
        text-align: center;
        padding: {Space.xl} 0 {Space.md};
        color: {Color.text_muted};
        font-size: {Type.small[0]};
    }}
    .ds-footer__line {{
        width: 80px;
        height: 1px;
        background: linear-gradient(90deg, transparent, {Color.border_strong}, transparent);
        margin: 0 auto {Space.lg};
    }}
    .ds-footer a {{ color: {Color.brand}; text-decoration: none; font-weight: 600; }}
    .ds-footer a:hover {{ text-decoration: underline; }}

    /* Responsive */
    @media (max-width: 768px) {{
        .ds-hero {{ padding: {Space.xl} {Space.lg}; }}
        .ds-hero__title {{ font-size: 1.8rem; }}
        .ds-feature {{ min-height: auto; }}
    }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# Components
# ════════════════════════════════════════════════════════════════════

def hero(title: str, subtitle: str, eyebrow: str | None = None,
         stats: list[tuple[str, str]] | None = None) -> None:
    """Top-of-page hero banner.

    stats: list of (label, value) tuples rendered as pills. Example:
        stats=[("Models", "3"), ("Datasets", "PTB-XL · NIH · UCI")]
    """
    parts = ['<div class="ds-hero">']
    if eyebrow:
        parts.append(f'<div class="ds-hero__eyebrow">{eyebrow}</div>')
    parts.append(f'<h1 class="ds-hero__title">{title}</h1>')
    parts.append(f'<p class="ds-hero__subtitle">{subtitle}</p>')
    if stats:
        pills = "".join(
            f'<span class="ds-pill"><strong>{v}</strong> &nbsp;{k}</span>'
            for k, v in stats
        )
        parts.append(f'<div class="ds-hero__stats">{pills}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None,
                   eyebrow: str | None = None) -> None:
    """Standard page heading for each non-home section."""
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
    """Card used on the home grid. Render inside a st.columns() cell."""
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
    """A single metric display. variant ∈ default|success|warning|danger."""
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
    """Big centered result card used on prediction pages.
    variant ∈ info|success|warning|danger."""
    sub = f'<p class="ds-result__subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
        <div class="ds-result ds-result--{variant}">
            <h2 class="ds-result__title">{title}</h2>
            {sub}
        </div>
    """, unsafe_allow_html=True)


def info_callout(title: str, body: str, variant: str = "info",
                 icon: str = "ℹ️") -> None:
    """Inline note box. variant ∈ info|success|warning|danger."""
    icon_map = {"info": "ℹ️", "success": "✓", "warning": "⚠", "danger": "⚠"}
    icon = icon_map.get(variant, icon)
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
    """Returns HTML string for a small status pill — use inside markdown.
    variant ∈ success|warning|danger|info|neutral."""
    return f'<span class="ds-badge ds-badge--{variant}">{text}</span>'


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="ds-disclaimer">{text}</div>',
                unsafe_allow_html=True)


def page_divider() -> None:
    st.markdown('<hr class="ds-divider" />', unsafe_allow_html=True)


def footer(text: str = "Healthcare AI Prediction Portal") -> None:
    st.markdown(f"""
        <div class="ds-footer">
            <div class="ds-footer__line"></div>
            {text} · Abdullah Abdul Sami · Northwestern University
        </div>
    """, unsafe_allow_html=True)
