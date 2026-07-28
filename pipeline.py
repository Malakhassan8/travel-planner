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

DATA_DIR = "data"  # folder containing cairo.txt, paris.txt, bangkok.txt, lisbon.txt

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
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ---------- Step 4: Preference extraction ----------

pref_parser = PydanticOutputParser(pydantic_object=UserPreferences)

pref_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract structured travel preferences from a user's message. "
     "Only use information the user actually stated or clearly implied. "
     "If budget_level isn't explicit, infer it from context. "
     "Default pace to 'moderate' if not mentioned.\n\n{format_instructions}"),
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
     "OUTPUT FORMAT — read carefully, this is strictly enforced:\n"
     "- Respond with ONLY the raw JSON object. No preamble like 'Here's your itinerary'. "
     "No explanation before or after. No markdown code fences (no ```).\n"
     "- Valid JSON only: no trailing commas after the last item in any list or object.\n\n"
     "{format_instructions}"),
    ("user",
     "User preferences:\nCity: {city}\nDays: {days}\nBudget level: {budget_level}\n"
     "Daily budget (USD): {daily_budget_usd}\nInterests: {interests}\nPace: {pace}\n\n"
     "Destination context:\n{context}")
]).partial(format_instructions=itinerary_parser.get_format_instructions())


def _clean_json_text(text: str) -> str:
    """LLMs sometimes wrap JSON in prose or markdown fences, or leave a trailing
    comma before a closing bracket (invalid JSON). Strip both before parsing."""
    # Drop markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    # Keep only the outermost { ... } block, dropping any preamble/explanation text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    # Remove trailing commas before a closing bracket/brace, e.g. "...},\n]" -> "...}\n]"
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _invoke_and_parse(prompt, parser, llm, inputs: dict):
    """Run prompt|llm, clean the raw text, then parse. If parsing still fails,
    ask the LLM once to fix its own output before giving up."""
    raw_chain = prompt | llm
    response = raw_chain.invoke(inputs)
    raw_text = response.content if hasattr(response, "content") else str(response)

    cleaned = _clean_json_text(raw_text)
    try:
        return parser.parse(cleaned)
    except Exception:
        # One repair attempt: show the LLM its own broken output and ask it to fix the JSON only
        fix_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "The following text was supposed to be valid JSON matching a schema, but failed "
             "to parse. Fix it and return ONLY the corrected raw JSON object — no preamble, "
             "no markdown fences, no trailing commas.\n\n{format_instructions}"),
            ("user", "Broken output:\n{broken}")
        ]).partial(format_instructions=parser.get_format_instructions())
        fix_chain = fix_prompt | llm
        fixed_response = fix_chain.invoke({"broken": cleaned})
        fixed_text = fixed_response.content if hasattr(fixed_response, "content") else str(fixed_response)
        fixed_cleaned = _clean_json_text(fixed_text)
        return parser.parse(fixed_cleaned)  # if this still fails, the exception surfaces to the UI


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
    return _invoke_and_parse(itinerary_prompt, itinerary_parser, get_llm(), {
        "city": prefs.city, "days": prefs.days, "budget_level": prefs.budget_level,
        "daily_budget_usd": prefs.daily_budget_usd, "interests": ", ".join(prefs.interests),
        "pace": prefs.pace, "context": context,
    })


def regenerate_day(prefs: UserPreferences, day_number: int, itinerary: "Itinerary" = None, avoid_note: str = "") -> DayPlan:
    """Regenerate a single day, optionally avoiding something the user didn't like.
    If `itinerary` is passed, activities already used on other days are listed so
    the regenerated day doesn't just repeat them."""
    context = get_context_for_trip(prefs)

    used_elsewhere = []
    if itinerary is not None:
        for d in itinerary.days:
            if d.day != day_number:
                used_elsewhere.extend(a.name for a in d.activities)

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
         "- Do not repeat any activity already used on another day of this trip (listed below), "
         "pick different ones from the context instead.\n"
         "- If the user gives a note about what to avoid or change, actually swap out the "
         "relevant activity for a different one — don't just mention the conflict in the reason text.\n"
         f"{{extra_note}}\n\n"
         "Already used on other days (avoid repeating): {used_elsewhere}\n\n"
         "OUTPUT FORMAT — strictly enforced: respond with ONLY the raw JSON object. "
         "No preamble, no markdown fences, no trailing commas.\n\n{format_instructions}"),
        ("user",
         "City: {city}\nBudget level: {budget_level}\nDaily budget (USD): {daily_budget_usd}\n"
         "Interests: {interests}\nPace: {pace}\n\nContext:\n{context}")
    ]).partial(format_instructions=day_parser.get_format_instructions())

    return _invoke_and_parse(day_prompt, day_parser, get_llm(), {
        "day_number": day_number, "city": prefs.city, "budget_level": prefs.budget_level,
        "daily_budget_usd": prefs.daily_budget_usd, "interests": ", ".join(prefs.interests),
        "pace": prefs.pace, "context": context,
        "extra_note": f"- Note: {avoid_note}" if avoid_note else "",
        "used_elsewhere": ", ".join(used_elsewhere) if used_elsewhere else "none",
    })
