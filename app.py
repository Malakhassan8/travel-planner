"""
app.py - Streamlit UI for the Travel Itinerary Planner.

Run with: streamlit run app.py
Expects a `data/` folder next to this file containing cairo.txt, paris.txt,
bangkok.txt, lisbon.txt, and a GROQ_API_KEY set in the environment or Streamlit secrets.
"""

import os
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF

from pipeline import extract_preferences, build_itinerary, regenerate_day

# ---------- Page config ----------
st.set_page_config(page_title="Travel Itinerary Planner", page_icon="🧭", layout="wide")

# ---------- Styling pass: palette, spacing, card-style days ----------
ACCENT = "#3D7EA6"        # medium blue accent (buttons, highlights, borders)
ACCENT_SOFT = "#7FB3D5"   # baby blue for hover/secondary bits
BG_APP = "#152238"        # dark navy — main app background
BG_CARD = "#EAF4FB"       # baby-blue-tinted off-white for day cards
BG_SIDEBAR = "#0F1A2B"    # darker navy for sidebar
TEXT_LIGHT = "#E7F0F8"    # main body text on dark background
TEXT_MUTED = "#9FB3C8"    # secondary/caption text on dark background
TEXT_DARK_ON_CARD = "#152238"    # strong dark navy text on light cards (numbers, headers)
TEXT_MUTED_ON_CARD = "#5A6B7D"   # secondary text on light cards

st.markdown(f"""
<style>
    /* App-wide background */
    .stApp {{
        background-color: {BG_APP};
    }}

    /* Sidebar background */
    section[data-testid="stSidebar"] {{
        background-color: {BG_SIDEBAR};
    }}

    /* Default text color on the dark background */
    .stApp, .stApp p, .stApp li, .stApp label {{
        color: {TEXT_LIGHT};
    }}
    h1, h2, h3 {{
        color: {TEXT_LIGHT} !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* Text input / text area fields */
    .stTextInput input, .stTextArea textarea {{
        background-color: {BG_CARD};
        color: {TEXT_DARK_ON_CARD};
        border-radius: 8px;
    }}

    /* Overall spacing */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* Buttons */
    .stButton > button[kind="primary"] {{
        background-color: {ACCENT};
        border: none;
        color: white;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {ACCENT_SOFT};
    }}
    .stButton > button:not([kind="primary"]) {{
        border: 2px solid {ACCENT};
        color: {TEXT_DARK_ON_CARD};
        background-color: {BG_CARD};
        font-weight: 600;
    }}
    .stButton > button:not([kind="primary"]):hover {{
        border-color: {ACCENT};
        background-color: {ACCENT_SOFT};
        color: {TEXT_DARK_ON_CARD};
    }}

    /* Metric (trip total) — target the specific testids so the value is never invisible */
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border-radius: 10px;
        padding: 0.75rem 1rem;
        border-left: 4px solid {ACCENT};
    }}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {{
        color: {TEXT_MUTED_ON_CARD} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {{
        color: {TEXT_DARK_ON_CARD} !important;
        font-weight: 700;
    }}

    /* Day cards */
    .day-card {{
        background-color: {BG_CARD};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        border-left: 4px solid {ACCENT};
    }}
    .day-card h3 {{
        margin-top: 0;
        color: {ACCENT} !important;
    }}
    .activity-cost {{
        color: {ACCENT};
        font-weight: 600;
    }}
    .day-total {{
        color: {TEXT_MUTED_ON_CARD};
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# ---------- API key ----------
if "GROQ_API_KEY" not in os.environ:
    key = st.sidebar.text_input("Groq API Key", type="password")
    if key:
        os.environ["GROQ_API_KEY"] = key

st.title("🧭 Travel Itinerary Planner")
st.caption("RAG + LangChain itinerary builder — tell it your trip, get a day-by-day plan.")

# ---------- Session state ("memory") ----------
if "prefs" not in st.session_state:
    st.session_state.prefs = None
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# ---------- Step 1: Input ----------
st.subheader("1. Describe your trip")
user_message = st.text_area(
    "e.g. '3 days in Lisbon, budget traveler, around 40 USD a day, love food and history, keep it relaxed'",
    height=80,
)

col1, col2 = st.columns([1, 4])
with col1:
    generate_clicked = st.button("Generate Itinerary", type="primary")

if generate_clicked and user_message.strip():
    if "GROQ_API_KEY" not in os.environ:
        st.error("Please add your Groq API key in the sidebar first.")
    else:
        try:
            with st.spinner("Reading your preferences..."):
                prefs = extract_preferences(user_message)
                st.session_state.prefs = prefs

            with st.spinner(f"Building your {prefs.days}-day {prefs.city} itinerary..."):
                itinerary = build_itinerary(prefs)
                st.session_state.itinerary = itinerary
        except Exception as e:
            st.error(
                "Something went wrong generating your itinerary. This can happen if "
                "the API key is invalid, a rate limit was hit, or the destination "
                "city isn't in the supported list (Cairo, Paris, Bangkok, Lisbon).\n\n"
                f"Details: {e}"
            )

# ---------- Show extracted preferences (transparency) ----------
if st.session_state.prefs:
    prefs = st.session_state.prefs
    with st.expander("Extracted preferences (this is what the app 'remembers')"):
        st.json(prefs.model_dump())

# ---------- Step 2: Show itinerary ----------
if st.session_state.itinerary:
    itinerary = st.session_state.itinerary
    prefs = st.session_state.prefs

    st.subheader(f"2. Your {itinerary.city} Itinerary")
    st.metric("Trip total", f"${itinerary.trip_total:.0f}")

    for day in itinerary.days:
        activities_html = ""
        for act in day.activities:
            activities_html += f"""
            <p style='margin-bottom:0.2rem;'><strong>{act.name}</strong> —
            <span class='activity-cost'>${act.cost_est:.0f}</span> · <em>{act.type}</em></p>
            <p style='color:{TEXT_MUTED_ON_CARD}; font-size:0.85rem; margin-top:0; margin-bottom:0.8rem;'>{act.reason}</p>
            """

        card_html = f"""
        <div class="day-card">
            <h3>Day {day.day}</h3>
            {activities_html}
            <div class='day-total'>Daily total: <b>${day.daily_total:.0f}</b></div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        avoid_note = st.text_input(
            "Anything to avoid or change for this day? (optional)",
            key=f"avoid_{day.day}",
            placeholder="e.g. 'less walking' or 'no museums'",
        )

        regen_col, _ = st.columns([1, 4])
        with regen_col:
            if st.button(f"🔁 Regenerate Day {day.day}", key=f"regen_{day.day}"):
                try:
                    with st.spinner(f"Regenerating Day {day.day}..."):
                        # Pass the current itinerary so the regenerated day avoids
                        # repeating activities already used on other days, and the
                        # optional note so the user can steer what changes.
                        new_day = regenerate_day(prefs, day.day, itinerary, avoid_note=avoid_note)
                        # Replace this day in the itinerary
                        for i, d in enumerate(itinerary.days):
                            if d.day == day.day:
                                itinerary.days[i] = new_day
                        itinerary.trip_total = sum(d.daily_total for d in itinerary.days)
                        st.session_state.itinerary = itinerary
                        st.rerun()
                except Exception as e:
                    st.error(f"Couldn't regenerate Day {day.day} — try again. Details: {e}")
        st.divider()

    # ---------- Cost breakdown chart ----------
    st.subheader("3. Cost Breakdown")
    day_labels = [f"Day {d.day}" for d in itinerary.days]
    day_totals = [d.daily_total for d in itinerary.days]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#152238")
    ax.set_facecolor("#152238")
    ax.bar(day_labels, day_totals, color=ACCENT)
    ax.set_ylabel("Cost (USD)", color="#E7F0F8")
    ax.set_title(f"Daily cost — {itinerary.city}", color="#E7F0F8")
    ax.tick_params(colors="#E7F0F8")
    for spine in ax.spines.values():
        spine.set_color("#9FB3C8")
    for i, v in enumerate(day_totals):
        ax.text(i, v + 0.5, f"${v:.0f}", ha="center", color="#E7F0F8")
    st.pyplot(fig)

    # ---------- PDF export ----------
    st.subheader("4. Export")

    def _clean(text: str) -> str:
        """FPDF's built-in Helvetica font only supports latin-1. Strip accents
        (São -> Sao, Pastéis -> Pasteis) instead of crashing on unicode chars."""
        import unicodedata
        normalized = unicodedata.normalize("NFKD", text)
        return normalized.encode("ascii", "ignore").decode("ascii")

    def build_pdf(itinerary) -> bytes:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _clean(f"{itinerary.city} Itinerary"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Trip total: ${itinerary.trip_total:.0f}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        for day in itinerary.days:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, f"Day {day.day}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            for act in day.activities:
                line = _clean(f"- {act.name} (${act.cost_est:.0f}, {act.type})")
                pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 5, _clean(f"  {act.reason}"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 6, f"Daily total: ${day.daily_total:.0f}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

        return bytes(pdf.output())

    pdf_bytes = build_pdf(itinerary)
    st.download_button(
        "📄 Download Itinerary as PDF",
        data=pdf_bytes,
        file_name=f"{itinerary.city}_itinerary.pdf",
        mime="application/pdf",
    )
