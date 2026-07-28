# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Malak Hassan Ali Mahmoud             |
| Project Name     | Travel Itinerary Planner             |
| GitHub Username  | Malakhassan8                         |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

Travel Itinerary Planner is a RAG (Retrieval-Augmented Generation) powered app that builds custom day-by-day travel itineraries. Users describe their trip in plain language (destination, budget, interests, and pace), and the app retrieves relevant destination info from a curated knowledge base and generates a structured, budget-aware itinerary using an LLM. Users can regenerate individual days — optionally steering the regeneration with a note (e.g. "less walking") — view a cost breakdown chart, and export the full itinerary as a PDF.

Currently supports four destinations: **Cairo, Paris, Bangkok, and Lisbon.**

---

# ✨ Features

* Natural language trip input — describe your trip in one sentence and the app extracts structured preferences (city, days, budget, interests, pace)
* RAG-based itinerary generation using semantic search over per-city destination knowledge, so recommendations are grounded in real content instead of invented
* Day-by-day itinerary with cost, category, and a reasoning note for each activity
* Per-day regeneration with an optional steering note (e.g. "no museums", "less walking") — avoids repeating activities already used elsewhere in the itinerary
* Cost breakdown bar chart across all days
* One-click PDF export of the full itinerary
* Clean error handling — invalid API keys or model hiccups show a readable message instead of a crash

---

# 🛠️ Technologies Used

* **Python**
* **Streamlit** — interactive web UI
* **LangChain** — prompt orchestration and structured output parsing
* **Groq (Llama 3.3 70B)** — LLM inference
* **Sentence-Transformers** (`all-MiniLM-L6-v2`) — text embeddings
* **FAISS** — vector similarity search
* **Pydantic** — structured schema validation for LLM outputs
* **Matplotlib** — cost breakdown visualization
* **fpdf2** — PDF generation
* **pyngrok** — tunneling for running/demoing the app from Google Colab

---

# ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Malakhassan8/travel-planner.git
   cd travel-planner
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the `data/` folder (containing `cairo.txt`, `paris.txt`, `bangkok.txt`, `lisbon.txt`) sits alongside `app.py` and `pipeline.py`.
4. Set your Groq API key as an environment variable, or paste it into the sidebar when the app opens.

### Running in Google Colab
```python
!git clone https://github.com/Malakhassan8/travel-planner.git /content/repo
!pip install streamlit langchain-core langchain-groq sentence-transformers faiss-cpu pydantic matplotlib fpdf2 pyngrok --quiet

from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(8501)
print(public_url)

!streamlit run /content/repo/app.py &>/content/log.txt &
```

---

# 🚀 Usage

1. Launch the app (`streamlit run app.py` locally, or via the Colab + ngrok steps above)
2. Enter your Groq API key in the sidebar if prompted
3. Describe your trip, e.g.:
   > *"3 days in Cairo, budget traveler, around 30 USD a day, love food and history, keep it relaxed"*
4. Click **Generate Itinerary**
5. Review the extracted preferences, day-by-day plan, and cost chart
6. Optionally type a note under any day (e.g. "less walking") and click 🔁 to regenerate just that day
7. Click **📄 Download Itinerary as PDF** to export

---

# 📸 Demo

*Add screenshots, GIFs, or a demo video here.*

---

# 📈 Results

* Successfully generates coherent, budget-consistent itineraries across all four supported cities, with trip totals that correctly sum daily costs
* Per-day regeneration was tested and fixed to avoid duplicating activities already used elsewhere in the itinerary, and tuned to produce real variety across repeated regenerations
* PDF export handles non-ASCII destination names and accented characters (e.g. "São", "Pastéis") without crashing
* Destination knowledge base expanded per city to give the retrieval and regeneration steps enough real options to draw from

---

# 🔮 Future Improvements

* Expand to more destination cities
* Add multi-day drag-and-drop reordering of activities
* Support multi-city trips in a single itinerary
* Persist itineraries so users can save/reload past trips
* Deploy on Streamlit Community Cloud for a permanent public link instead of relying on ngrok tunnels

---

# 📚 About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
