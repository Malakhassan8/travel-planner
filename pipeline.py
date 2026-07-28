"""
pipeline.py - all the RAG + chain logic from Steps 2-5, consolidated for Streamlit.
Uses st.cache_resource so the embedding model, FAISS index, and LLM are only
loaded/built once per session, not on every rerun.
"""

import re
import os
from typing import List, Optional

import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Resolve relative to this file's location, so it works no matter what
# directory Streamlit is launched from (e.g. /content vs /content/repo).
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CITY_FILES = {
    "Cairo": "cairo.txt",
    "Paris": "paris.txt",
    "Bangkok": "bangkok.txt",
    "Lisbon": "lisbon.txt",
}


# ---------- Schemas ----------

class UserPreferences(BaseModel):
    city: str = Field(description="Destination city")
    days: int = Field(description="Number of days for the trip")
    budget_level: str = Field(description="One of: 'budget', 'mid-range', 'luxury'")
    daily_budget_usd: Optional[float] = Field(default=None, description="Approx daily budget in USD, or null")
    interests: List[str] = Field(description="List of interest tags")
    pace: str = Field(default="moderate", description="One of: 'relaxed', 'moderate', 'packed'")


class Activity(BaseModel):
    name: str
    cost_est: float
    type: str
    reason: str


class DayPlan(BaseModel):
    day: int
    city: str
    activities: List[Activity]
    daily_total: float


class Itinerary(BaseModel):
    city: str
    days: List[DayPlan]
    trip_total: float


# ---------- Step 2: Chunking ----------

def chunk_city_file(filepath: str, city: str) -> List[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    sections = re.split(r"\n(?=## )", text)
    docs = []
    for section in sections:
        section = section.strip()
        if not section or not section.startswith("##"):
            continue
        header_line, _, body = section.partition("\n")
        section_name = header_line.replace("##", "").strip()
        docs.append(Document(
            page_content=f"[{city} - {section_name}]\n{body.strip()}",
            metadata={"city": city, "section": section_name},
        ))
    return docs


@st.cache_resource
def load_all_docs():
    all_docs = []
    for city, filename in CITY_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        all_docs.extend(chunk_city_file(filepath, city))
    return all_docs


# ---------- Step 3: Embeddings + FAISS ----------

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_index():
    all_docs = load_all_docs()
    embed_model = load_embed_model()
    texts = [doc.page_content for doc in all_docs]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    return index


def search(query: str, k: int = 4, city_filter: str = None):
    all_docs = load_all_docs()
    embed_model = load_embed_model()
    index = build_index()

    query_embedding = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    search_k = k * 3 if city_filter else k
    distances, indices = index.search(query_embedding, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc = all_docs[idx]
        if city_filter and doc.metadata["city"].lower() != city_filter.lower():
            continue
        results.append((doc, float(dist)))
        if len(results) >= k:
            break
    return results


# ---------- LLM ----------

@st.cache_resource
def get_llm():
    """Main LLM for preference extraction and full itinerary generation.
    temperature=0 keeps these consistent and less prone to inventing things."""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


@st.cache_resource
def get_regen_llm():
    """Separate instance with some temperature, used only for regenerating a
    single day. A little randomness helps it actually explore the available
    options instead of converging on the same pick every time."""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.6)


# ---------- Step 4: Preference extraction ----------

pref_parser = PydanticOutputParser(pydantic_object=UserPreferences)

pref_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract structured travel preferences from a user's message. "
     "Only use information the user actually stated or clearly implied. "
     "If budget_level isn't explicit, infer it from context. "
     "Default pace to 'moderate' if not mentioned.\n\n"
     "IMPORTANT — budget normalization:\n"
     "The 'daily_budget_usd' field must always be a PER-DAY amount, in USD.\n"
     "- If the user gives a per-day figure (phrases like 'a day', 'per day', 'daily', "
     "'each day'), use that number directly.\n"
     "- If the user gives a WHOLE-TRIP or TOTAL figure (phrases like 'for the whole trip', "
     "'total budget', 'overall', 'for the trip', or just a lump sum with no 'per day' "
     "qualifier alongside a stated number of days), divide it by the number of days "
     "to get the per-day amount. Example: '$150 for the whole 5-day trip' -> "
     "daily_budget_usd = 30.\n"
     "- If it's genuinely ambiguous whether a figure is per-day or total, prefer treating "
     "it as per-day only when 'a day'/'per day' is explicitly present; otherwise treat it "
     "as a total and divide by days.\n\n"
     "{format_instructions}"),
    ("user", "{user_message}")
]).partial(format_instructions=pref_parser.get_format_instructions())


def extract_preferences(user_message: str) -> UserPreferences:
    chain = pref_prompt | get_llm() | pref_parser
    return chain.invoke({"user_message": user_message})


# ---------- Step 5: Itinerary builder ----------

itinerary_parser = PydanticOutputParser(pydantic_object=Itinerary)

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a travel itinerary planner. Using ONLY the destination context provided "
     "below, build a day-by-day itinerary matching the user's preferences exactly.\n\n"
     "Rules:\n"
     "- Number of days MUST equal the user's requested trip length.\n"
     "- Activities per day scale with pace: relaxed=2-3, moderate=4, packed=5-6.\n"
     "- Every activity needs a short 'reason' tied to interests/budget/pace.\n"
     "- Keep costs realistic and consistent with the budget level.\n"
     "- Only use activities/places mentioned in the context — don't invent unlisted attractions.\n\n"
     "{format_instructions}"),
    ("user",
     "User preferences:\nCity: {city}\nDays: {days}\nBudget level: {budget_level}\n"
     "Daily budget (USD): {daily_budget_usd}\nInterests: {interests}\nPace: {pace}\n\n"
     "Destination context:\n{context}")
]).partial(format_instructions=itinerary_parser.get_format_instructions())


def get_context_for_trip(prefs: UserPreferences, k_per_query: int = 3) -> str:
    queries = list(prefs.interests) + ["budget and cost", "top things to see and do"]
    seen, combined = set(), []
    for q in queries:
        for doc, dist in search(q, k=k_per_query, city_filter=prefs.city):
            key = (doc.metadata["city"], doc.metadata["section"])
            if key not in seen:
                seen.add(key)
                combined.append(doc)
    return "\n\n---\n\n".join(d.page_content for d in combined)


def build_itinerary(prefs: UserPreferences) -> Itinerary:
    context = get_context_for_trip(prefs)
    chain = itinerary_prompt | get_llm() | itinerary_parser
    return chain.invoke({
        "city": prefs.city, "days": prefs.days, "budget_level": prefs.budget_level,
        "daily_budget_usd": prefs.daily_budget_usd, "interests": ", ".join(prefs.interests),
        "pace": prefs.pace, "context": context,
    })


def regenerate_day(prefs: UserPreferences, day_number: int, itinerary: Itinerary, avoid_note: str = "") -> DayPlan:
    """Regenerate a single day, avoiding activities already used on other days
    and keeping the same number of activities as the original day."""
    context = get_context_for_trip(prefs)

    # Collect activities already used on OTHER days, so we don't repeat them
    used_elsewhere = [
        act.name
        for d in itinerary.days
        if d.day != day_number
        for act in d.activities
    ]
    used_note = (
        f"- Do NOT repeat these activities, already used on other days: {', '.join(used_elsewhere)}."
        if used_elsewhere else ""
    )

    # Anchor the activity count to what this day originally had
    original_day = next((d for d in itinerary.days if d.day == day_number), None)
    target_count = len(original_day.activities) if original_day else None
    count_note = (
        f"- This day must have exactly {target_count} activities, matching the original day being replaced."
        if target_count else ""
    )

    day_parser = PydanticOutputParser(pydantic_object=DayPlan)
    day_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a travel itinerary planner. Using ONLY the context below, build ONE day "
         "of a trip matching the user's preferences.\n"
         "Rules:\n"
         "- This is day {day_number} of the trip.\n"
         "- Activities scale with pace: relaxed=2-3, moderate=4, packed=5-6.\n"
         "- Every activity needs a 'reason' tied to interests/budget/pace.\n"
         "- Only use activities/places mentioned in the context.\n"
         f"{{count_note}}\n{{used_note}}\n{{extra_note}}\n\n{{format_instructions}}"),
        ("user",
         "City: {city}\nBudget level: {budget_level}\nDaily budget (USD): {daily_budget_usd}\n"
         "Interests: {interests}\nPace: {pace}\n\nContext:\n{context}")
    ]).partial(format_instructions=day_parser.get_format_instructions())

    chain = day_prompt | get_regen_llm() | day_parser
    return chain.invoke({
        "day_number": day_number, "city": prefs.city, "budget_level": prefs.budget_level,
        "daily_budget_usd": prefs.daily_budget_usd, "interests": ", ".join(prefs.interests),
        "pace": prefs.pace, "context": context,
        "extra_note": f"- Note: user specifically asked to avoid/change this: {avoid_note}" if avoid_note else "",
        "used_note": used_note,
        "count_note": count_note,
    })
