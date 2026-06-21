# Cuppa — Coffee Kiosk

Single-screen Streamlit kiosk with a Groq-powered AI barista. No multi-page nav, no 10-question wizard — pick a mood and the weather, get one coffee recommendation, watch the cup and brew animate live.


<img width="1917" height="966" alt="image" src="https://github.com/user-attachments/assets/8be1b8ad-a08e-42e9-827b-2db2769134b3" />

**Grok powered AI assistance for a cafe called cuppa.
**How to use kiosk
1. Choose ur coffee or
2. Use AI barista to give u recommendations based on ur mood and weather
3. Brew ur ☕ coffee
4. Pick up ur order

## Why Groq, not Ollama

This project folder was originally scoped for Ollama, but Ollama requires a local model server running on the host machine. Streamlit Community Cloud only runs your `app.py` in a shared container — there is no way to run or reach an Ollama server from it. Groq is a hosted, API-key-based LLM service, so it's the only one of the two that actually works once deployed. Locally, swapping back to Ollama would mean replacing the `Groq(...)` client and `chat.completions.create(...)` call in `app.py` with an Ollama client — but it would no longer be deployable to Community Cloud.

## 1. Get a Groq API key

Sign up at [console.groq.com/keys](https://console.groq.com/keys) (free tier available) and create a key.

## 2. Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your real key in
streamlit run app.py
```

For a true kiosk look on a touchscreen, open the local URL in Chrome and launch it in kiosk mode, e.g. on Windows:

```bash
chrome.exe --kiosk http://localhost:8501
```

Press `Alt+F4` (or `Esc` then close) to exit kiosk mode.

## 3. Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo (make sure `.streamlit/secrets.toml` is **not** committed — `.gitignore` already excludes it).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, click **New app**, and point it at your repo and `app.py`.
3. Before (or after) deploying, open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
4. Deploy. The app boots straight into the kiosk screen — no sidebar, no extra pages.

## How it behaves without a key

If `GROQ_API_KEY` is missing or a Groq call fails for any reason, the app silently falls back to a deterministic rule-based recommender (see `fallback_recommend()` in `app.py`) so the kiosk never shows an error on screen — it just runs in "offline demo mode," flagged with a small caption under the AI button.

## Model

Uses `openai/gpt-oss-20b` on Groq — currently the fastest production model on their platform (~1000 tok/s), which keeps the "thinking" animation snappy. `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are NOT used here because Groq announced their deprecation on 2026-06-17 (shutdown 2026-08-16); `openai/gpt-oss-20b` is their recommended replacement and is already production-grade.

## File structure

```
app.py                          single-screen kiosk app (everything lives here)
requirements.txt                streamlit + groq
.streamlit/config.toml          dark/amber theme (committed, no secrets)
.streamlit/secrets.toml.example template — copy to secrets.toml locally, never commit the real one
.gitignore                      keeps secrets.toml and caches out of git
```

## Customizing

- **Brand name / tagline** — edit `APP_NAME` and `TAGLINE` near the top of `app.py`.
- **Menu, prices, default strength/sweetness** — edit the `COFFEES` dict.
- **Mood/weather options** — edit `MOODS` / `WEATHERS` / `AGES`.
- **AI prompt or output fields** — edit `SYSTEM_PROMPT`; if you add/remove a JSON key, update `ai_recommend()` and `fallback_recommend()` to match.
