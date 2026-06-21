"""
Cuppa — single-screen coffee kiosk
Streamlit + Groq (LPU inference). Built for Streamlit Community Cloud.

Design notes (read before editing):
- Everything lives on ONE screen, no st.pages / multipage / sidebar nav.
- AI flow is intentionally short: mood + weather (+ optional age) -> one
  recommendation. No 10-question wizard — that's the whole point of a kiosk.
- All "animation" is CSS keyframes (cheap, GPU-composited, no JS framework)
  plus short time.sleep() staged reveals for the thinking/brew sequences.
  This keeps the app fast on Streamlit Community Cloud's free tier.
- Groq is the only LLM backend. If GROQ_API_KEY is missing or the call
  fails for any reason, the app falls back to a deterministic rule-based
  recommender so the kiosk never looks "broken" on stage.
"""

import json
import os
import random
import re
import time
from datetime import datetime

import streamlit as st

try:
    from groq import Groq
except ImportError:  # groq package not installed yet
    Groq = None


# =============================================================================
# CONFIG
# =============================================================================
APP_NAME = "Cuppa"
TAGLINE = "Your Mood. Your Coffee. Instantly."
CURRENCY = "₹"

# Production, non-deprecated Groq model IDs (checked June 2026).
# openai/gpt-oss-20b is the fastest production model on Groq (~1000 tok/s) —
# ideal for a kiosk that needs sub-second responses.
GROQ_MODEL = "openai/gpt-oss-20b"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COFFEES = {
    "Espresso":   {"icon": "☕", "desc": "Bold and intense",    "liquid": "#3b1f12", "foam": "#5a3420", "price": 99,  "strength": 85, "sweetness": 10, "milk": False},
    "Americano":  {"icon": "☕", "desc": "Simple and strong",   "liquid": "#4a2615", "foam": "#6b3f25", "price": 109, "strength": 65, "sweetness": 10, "milk": False},
    "Cappuccino": {"icon": "🥛", "desc": "Classic and creamy",  "liquid": "#7a4a28", "foam": "#e8d4b8", "price": 149, "strength": 55, "sweetness": 35, "milk": True},
    "Latte":      {"icon": "🥛", "desc": "Smooth and milky",    "liquid": "#9c6b3f", "foam": "#f0e0c8", "price": 149, "strength": 40, "sweetness": 30, "milk": True},
    "Flat White": {"icon": "🥛", "desc": "Rich and velvety",    "liquid": "#8a5a32", "foam": "#e8d8c0", "price": 159, "strength": 60, "sweetness": 25, "milk": True},
    "Mocha":      {"icon": "🍫", "desc": "Chocolatey delight",  "liquid": "#4a2818", "foam": "#7a4a2a", "price": 179, "strength": 60, "sweetness": 55, "milk": True},
}
COFFEE_ORDER = list(COFFEES.keys())

MOODS = [("Happy", "😊"), ("Tired", "😴"), ("Relaxed", "😌"),
         ("Stressed", "😤"), ("Focused", "💻"), ("Studying", "📚")]
WEATHERS = [("Sunny", "☀️"), ("Rainy", "🌧️"), ("Cold", "🌫️"), ("Hot", "🔥")]
AGES = ["<18", "18-25", "26-40", "40+"]

SYSTEM_PROMPT = """You are an expert coffee sommelier working inside a fast coffee kiosk called Cuppa.
Recommend exactly ONE coffee from this list only: Espresso, Americano, Cappuccino, Latte, Flat White, Mocha.
Respond with ONLY a single-line JSON object (no markdown, no code fences, no commentary) with exactly these keys:
"coffee": one of the six names above, written exactly as listed.
"strength": one of "Mild", "Medium", "Medium-High", "High".
"temperature": "Hot" or "Iced".
"reason": max 2 short sentences, under 30 words total, second person, referencing the mood and weather given.
"fun_line": one short fun sentence, under 12 words.
"why": an array of exactly 3 short bullet phrases (3-6 words each) explaining the pick.
Keep the entire response under 80 words total."""

WHY_BULLETS_FALLBACK = {
    "Mocha":      ["Chocolate lifts your mood", "Caffeine boosts alertness", "Comfort in every sip"],
    "Espresso":   ["Maximum caffeine, minimum time", "Sharpens focus fast", "No milk, no distractions"],
    "Americano":  ["Steady, long-lasting caffeine", "Light on the stomach", "Keeps you alert for hours"],
    "Cappuccino": ["Balanced caffeine and milk", "Light foam lifts your mood", "Classic comfort in a cup"],
    "Flat White": ["Smooth, low-acid espresso", "Silky texture, no bitterness", "Easy on a relaxed day"],
    "Latte":      ["Gentle caffeine, mostly milk", "Smooth and easy-going", "Comforting without the punch"],
}

FUN_LINES_FALLBACK = {
    "Mocha": "Like a warm hug in a mug.",
    "Espresso": "Small cup, big wake-up call.",
    "Americano": "Strong, simple, no nonsense.",
    "Cappuccino": "Foam on top, smile on you.",
    "Flat White": "Silky smooth, zero drama.",
    "Latte": "Soft, milky, and easygoing.",
}

DEFAULT_COFFEE = "Latte"
DEFAULTS = {
    "coffee": DEFAULT_COFFEE,
    "strength": COFFEES[DEFAULT_COFFEE]["strength"],
    "sweetness": COFFEES[DEFAULT_COFFEE]["sweetness"],
    "temp": "Hot",
    "mood": None,
    "weather": None,
    "age": None,
    "ai_result": None,
    "brewed": False,
    "order_id": None,
}


def init_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Keys bound to a widget (st.slider(..., key=...)) later in the script.
# Streamlit raises StreamlitAPIException if code writes to a widget-bound
# session_state key AFTER that widget has already rendered earlier in the
# SAME script run. The AI-recommendation handler and the "Start New Order"
# reset both need to change strength/sweetness from code that runs after
# the sliders render, so they stage the new value in a "pending_<key>" slot
# instead of writing the real key directly. Draining pending slots here —
# before the sliders are instantiated — applies the change safely.
WIDGET_BOUND_KEYS = ("strength", "sweetness")


def apply_pending_updates():
    for key in WIDGET_BOUND_KEYS:
        pending_key = f"pending_{key}"
        if pending_key in st.session_state:
            st.session_state[key] = st.session_state.pop(pending_key)


init_state()
apply_pending_updates()


# =============================================================================
# GROQ CLIENT + RECOMMENDATION LOGIC
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_groq_client():
    """Cached Groq client. Returns None if no key is configured anywhere —
    the app degrades gracefully to the offline recommender in that case."""
    if Groq is None:
        return None
    api_key = ""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def fallback_recommend(mood, weather, age):
    """Deterministic rule-based recommender used whenever Groq is
    unavailable or errors out. Keeps the kiosk usable with zero API key."""
    mood = mood or "Relaxed"
    weather = weather or "Sunny"

    primary = {
        "Tired": "Mocha",
        "Stressed": "Mocha",
        "Focused": "Espresso",
        "Studying": "Americano",
        "Happy": "Cappuccino",
        "Relaxed": "Flat White",
    }.get(mood, "Latte")

    temperature = "Iced" if weather in ("Hot", "Sunny") else "Hot"
    strength_label = {
        "Espresso": "High", "Americano": "Medium-High", "Mocha": "Medium-High",
        "Cappuccino": "Medium", "Flat White": "Medium", "Latte": "Mild",
    }.get(primary, "Medium")

    reasons = {
        "Tired": "You seem tired, so a caffeine boost with comforting chocolate notes will help.",
        "Stressed": "You're a bit stressed — chocolate and caffeine together take the edge off.",
        "Focused": "You're locked in, so a sharp, no-nonsense shot keeps that focus going.",
        "Studying": "Long study session ahead — a strong, simple brew keeps you alert.",
        "Happy": "You're in a good mood, so a classic creamy cup matches the vibe.",
        "Relaxed": "You're relaxed, so something smooth and velvety fits perfectly.",
    }
    reason = reasons.get(mood, "Here's a well-rounded pick for how you're feeling today.")
    reason += f" With {weather.lower()} weather outside, this is the right temperature call too."

    return {
        "coffee": primary,
        "strength": strength_label,
        "temperature": temperature,
        "reason": reason,
        "fun_line": FUN_LINES_FALLBACK.get(primary, "Brewed just for your mood."),
        "why": WHY_BULLETS_FALLBACK.get(primary, ["Matched to your mood", "Matched to the weather", "A solid all-round pick"]),
        "source": "offline",
    }


def _extract_json(raw):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def ai_recommend(mood, weather, age):
    """Call Groq for a recommendation; fall back to the rule-based
    recommender on any failure (missing key, network error, bad JSON)."""
    client = get_groq_client()
    if client is None:
        return fallback_recommend(mood, weather, age)

    user_msg = (
        f"Mood: {mood}\n"
        f"Weather: {weather}\n"
        f"Age group: {age or 'not provided'}\n"
        "Recommend one coffee now."
    )

    data = None
    for use_json_mode in (True, False):
        try:
            kwargs = dict(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6,
                max_tokens=250,
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            data = _extract_json(resp.choices[0].message.content)
            break
        except Exception:
            data = None
            continue

    if not data:
        return fallback_recommend(mood, weather, age)

    coffee = str(data.get("coffee", "")).strip()
    if coffee not in COFFEES:
        matched = next(
            (c for c in COFFEES if c.lower() in coffee.lower() or coffee.lower() in c.lower()),
            None,
        )
        if not matched:
            return fallback_recommend(mood, weather, age)
        coffee = matched

    why = data.get("why")
    if not isinstance(why, list) or len(why) == 0:
        why = WHY_BULLETS_FALLBACK.get(coffee, ["Matched to your mood", "Matched to the weather", "A solid all-round pick"])

    return {
        "coffee": coffee,
        "strength": data.get("strength", "Medium"),
        "temperature": data.get("temperature", "Hot") if data.get("temperature") in ("Hot", "Iced") else "Hot",
        "reason": data.get("reason", "A great pick for how you're feeling today."),
        "fun_line": data.get("fun_line", "Brewed just for you."),
        "why": why[:3],
        "source": "groq",
    }


def compute_price(coffee, strength_pct):
    base = COFFEES[coffee]["price"]
    extra_shot = 20 if strength_pct >= 80 else 0
    return base + extra_shot, extra_shot


STRENGTH_LABEL_TO_PCT = {"Mild": 25, "Medium": 50, "Medium-High": 70, "High": 90}


# =============================================================================
# HTML SAFETY HELPER
# =============================================================================
def _html(raw):
    """Flatten a multi-line HTML string to one line with no leading
    whitespace before handing it to st.markdown(unsafe_allow_html=True).

    Streamlit's markdown renderer follows CommonMark block rules: 4+
    leading spaces on a line is parsed as an indented code block, and a
    blank (or whitespace-only) line ends an HTML block early. Multi-line
    f-strings written inside indented Python code (functions, `with`
    blocks) inherit that indentation literally in the resulting string,
    and an interpolated value that happens to be "" can leave a
    whitespace-only line behind. Either one silently turns part of the
    markup into visible plain text instead of rendered HTML — exactly
    what happened to the live cup preview. Collapsing everything to a
    single line with no internal newlines sidesteps both failure modes.
    """
    return " ".join(line.strip() for line in raw.strip().splitlines() if line.strip())


# Reusable visual for the "grinding" brew stage: falling beans -> spinning
# gear -> scattering grounds. Built once (it never changes) and reused by
# the brew sequence below instead of just a single spinning emoji.
GRIND_VISUAL_HTML = _html("""
    <div class="grind-visual">
      <div class="grind-beans"><span></span><span></span><span></span></div>
      <div class="grind-gear">⚙️</div>
      <div class="grind-dust"><span></span><span></span><span></span><span></span></div>
    </div>
""")


# =============================================================================
# CUP PREVIEW (pure HTML/CSS — no JS, no images, cheap to render)
# =============================================================================
def render_cup(coffee, temp, strength_pct, sweetness_pct):
    liquid = COFFEES[coffee]["liquid"]
    foam = COFFEES[coffee]["foam"]
    fill_pct = min(82, 52 + int(strength_pct * 0.28))
    steam = "".join(
        f'<span class="steam" style="left:{28 + i * 16}%; animation-delay:{i * 0.45:.2f}s;"></span>'
        for i in range(3)
    ) if temp == "Hot" else ""
    ice = "".join(
        f'<div class="ice" style="left:{14 + i * 26}%; bottom:{6 + i * 5}px; animation-delay:{i * 0.3:.2f}s;"></div>'
        for i in range(3)
    ) if temp == "Iced" else ""
    foam_band = max(6, min(22, int(sweetness_pct * 0.2) + 6))
    # A few sparkling crystals on the foam, scaled to the sweetness slider —
    # more sweetness, more crystals (capped so it never looks cluttered).
    sugar_count = max(2, min(6, 2 + sweetness_pct // 20))
    sugar = "".join(
        f'<span class="sugar" style="left:{8 + i * 15}%; animation-delay:{i * 0.18:.2f}s;"></span>'
        for i in range(sugar_count)
    )
    liquid_style = (
        f"height:{fill_pct}%; "
        f"background: linear-gradient(180deg, {foam} 0%, {liquid} 18%, {liquid} 100%);"
    )

    return _html(f"""
        <div class="cup-stage">
          <div class="steam-wrap">{steam}</div>
          <div class="mug">
            <div class="liquid" style="{liquid_style}">
              <div class="foam-top" style="height:{foam_band}px; background:{foam};">{sugar}</div>
              {ice}
            </div>
            <div class="mug-glass"></div>
            <div class="handle"></div>
          </div>
          <div class="saucer"></div>
        </div>
    """)


# =============================================================================
# GLOBAL CSS — theme, cards, buttons, animations
# =============================================================================
GLOBAL_CSS = """
<style>
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; max-width: 100% !important; }
div[data-testid="stVerticalBlock"] { gap: 0.45rem; }

html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }

/* ---------- Brand header ---------- */
.brand { display: flex; align-items: center; gap: 10px; }
.brand-icon { font-size: 2.1rem; }
.brand-name { font-size: 1.35rem; font-weight: 800; color: #f5e6d3; letter-spacing: 0.5px; line-height: 1.1; }
.brand-tag { font-size: 0.78rem; color: #c9a876; }

.ai-teaser-row { display: flex; align-items: center; gap: 10px; justify-content: center; }
.ai-teaser {
  background: linear-gradient(90deg, rgba(232,163,61,0.18), rgba(232,163,61,0.05));
  border: 1px solid rgba(232,163,61,0.45);
  border-radius: 999px; padding: 10px 18px; text-align: center;
  color: #f0d9b5; font-weight: 600; font-size: 0.92rem;
  animation: pulseGlow 2.4s ease-in-out infinite;
}
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(232,163,61,0.0); }
  50% { box-shadow: 0 0 14px 2px rgba(232,163,61,0.35); }
}
.robo-badge {
  flex-shrink: 0; width: 42px; height: 42px; border-radius: 50%;
  background: linear-gradient(135deg, #e8a33d, #c97f1f);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; box-shadow: 0 0 0 0 rgba(232,163,61,0.55);
  animation: roboBounce 1.8s ease-in-out infinite;
}
@keyframes roboBounce {
  0%, 100%  { transform: translateY(0) scale(1);     box-shadow: 0 0 0 0 rgba(232,163,61,0.55); }
  50%       { transform: translateY(-3px) scale(1.08); box-shadow: 0 0 12px 4px rgba(232,163,61,0.35); }
}

.clock { text-align: right; color: #f0d9b5; font-weight: 700; font-size: 1.05rem; line-height: 1.25; }
.clock span { font-size: 0.72rem; color: #c9a876; font-weight: 400; }

/* ---------- Panel titles ---------- */
.panel-title { font-size: 1.0rem; font-weight: 700; color: #f0d9b5; margin: 2px 0 8px 0; }
.coffee-desc { font-size: 0.74rem; color: #b89a78; margin: -6px 0 6px 4px; }
.divider { border-top: 1px solid rgba(232,163,61,0.18); margin: 10px 0; }
.divider.thick { border-top: 2px solid rgba(232,163,61,0.25); margin: 14px 0 12px 0; }

/* ---------- Buttons (Streamlit native, restyled) ---------- */
div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
  border-radius: 12px !important;
  font-weight: 600 !important;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
  min-height: 2.6rem;
}
div[data-testid="stButton"] button:hover { transform: translateY(-1px); }
div[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, #e8a33d, #c97f1f) !important;
  border: none !important;
  color: #1b120c !important;
  box-shadow: 0 3px 10px rgba(232,163,61,0.35);
}
div[data-testid="stButton"] button[kind="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(232,163,61,0.25) !important;
  color: #e8d8c0 !important;
}

/* ---------- Live cup preview ---------- */
.cup-stage {
  position: relative; height: 270px; display: flex; flex-direction: column;
  align-items: center; justify-content: flex-end; padding-bottom: 6px;
}
.mug { position: relative; width: 150px; height: 170px; }
.mug-glass {
  position: absolute; inset: 0; border-radius: 0 0 26px 26px;
  border: 4px solid rgba(255,255,255,0.18);
  border-top: none;
  background: rgba(255,255,255,0.03);
  box-shadow: inset 0 0 18px rgba(0,0,0,0.35);
}
.liquid {
  position: absolute; bottom: 4px; left: 4px; right: 4px;
  border-radius: 0 0 22px 22px;
  transition: height 0.5s ease, background 0.5s ease;
  overflow: hidden;
}
.foam-top { position: relative; width: 100%; opacity: 0.9; }
.sugar {
  position: absolute; top: -3px; width: 4px; height: 4px; border-radius: 1px;
  background: #fff8e0; box-shadow: 0 0 3px rgba(255,255,255,0.9);
  animation: sugarSparkle 1.3s ease-in-out infinite;
}
@keyframes sugarSparkle {
  0%, 100% { opacity: 0.35; transform: scale(0.8) rotate(0deg); }
  50%      { opacity: 1;    transform: scale(1.2) rotate(30deg); }
}
.handle {
  position: absolute; right: -28px; top: 38px; width: 26px; height: 46px;
  border: 6px solid rgba(255,255,255,0.18); border-left: none;
  border-radius: 0 16px 16px 0;
}
.saucer { width: 190px; height: 14px; border-radius: 50%; background: rgba(255,255,255,0.08); margin-top: 6px; }

.steam-wrap { position: absolute; top: -10px; width: 150px; height: 80px; }
.steam {
  position: absolute; bottom: 0; width: 10px; height: 46px; border-radius: 50%;
  background: rgba(255,255,255,0.55); filter: blur(4px);
  animation: steamRise 2.4s ease-in-out infinite;
}
@keyframes steamRise {
  0%   { opacity: 0;   transform: translateY(0) scaleX(1); }
  30%  { opacity: 0.7; }
  100% { opacity: 0;   transform: translateY(-55px) scaleX(1.6); }
}
.ice {
  position: absolute; width: 10px; height: 10px; border-radius: 3px;
  background: rgba(220,240,255,0.85); border: 1px solid rgba(255,255,255,0.6);
  animation: iceBob 1.8s ease-in-out infinite;
}
@keyframes iceBob {
  0%, 100% { transform: translateY(0) rotate(-6deg); }
  50%      { transform: translateY(-4px) rotate(6deg); }
}
.live-badge { color: #ff6b6b; font-weight: 700; font-size: 0.78rem; }
.live-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#ff4d4d;
  animation: pulseDot 1.1s ease-in-out infinite; margin-right: 4px; }
@keyframes pulseDot { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

/* ---------- AI panel ---------- */
.ai-panel-header { font-size: 1.0rem; font-weight: 800; color: #f0d9b5; margin-bottom: 2px; }
.ai-sub { font-size: 0.76rem; color: #b89a78; font-weight: 400; margin-bottom: 8px; }
.q-label { font-size: 0.82rem; font-weight: 700; color: #e8d8c0; margin: 8px 0 4px 0; }

.thinking-box, .brew-box {
  display: flex; align-items: center; gap: 10px; justify-content: center;
  background: rgba(232,163,61,0.08); border: 1px dashed rgba(232,163,61,0.4);
  border-radius: 14px; padding: 16px; font-weight: 600; color: #f0d9b5;
  font-size: 0.92rem; margin: 8px 0;
}
.spin-bean { display: inline-block; font-size: 1.4rem; animation: beanSpin 1s linear infinite; }
@keyframes beanSpin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.brew-icon { font-size: 1.6rem; animation: beanSpin 0.9s linear infinite; }

/* ---------- Bean-grinding visual (used for the "Grinding..." brew stage) ---------- */
.grind-stage { flex-direction: column; gap: 2px; }
.grind-visual { position: relative; width: 100%; height: 50px; display: flex; align-items: center; justify-content: center; }
.grind-beans { position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 70px; height: 50px; }
.grind-beans span {
  position: absolute; top: -4px; font-size: 0.85rem;
  animation: beanFall 1.1s ease-in infinite;
}
.grind-beans span::before { content: "🫘"; }
.grind-beans span:nth-child(1) { left: 6px;  animation-delay: 0s; }
.grind-beans span:nth-child(2) { left: 30px; animation-delay: 0.35s; }
.grind-beans span:nth-child(3) { left: 54px; animation-delay: 0.7s; }
@keyframes beanFall {
  0%   { top: -4px; opacity: 1; transform: rotate(0deg); }
  55%  { top: 14px; opacity: 1; }
  100% { top: 18px; opacity: 0; transform: rotate(80deg); }
}
.grind-gear { font-size: 1.7rem; animation: beanSpin 0.5s linear infinite; position: relative; z-index: 1; }
.grind-dust { position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%); width: 60px; height: 20px; }
.grind-dust span {
  position: absolute; bottom: 0; width: 5px; height: 5px; border-radius: 50%;
  background: #caa06b; animation: dustScatter 1s ease-out infinite;
}
.grind-dust span:nth-child(1) { left: 12px; --dx: -16px; animation-delay: 0.1s; }
.grind-dust span:nth-child(2) { left: 24px; --dx: -5px;  animation-delay: 0.35s; }
.grind-dust span:nth-child(3) { left: 36px; --dx: 5px;   animation-delay: 0.6s; }
.grind-dust span:nth-child(4) { left: 48px; --dx: 16px;  animation-delay: 0.85s; }
@keyframes dustScatter {
  0%   { opacity: 0; bottom: 0;   transform: translateX(0) scale(0.5); }
  35%  { opacity: 1; }
  100% { opacity: 0; bottom: 14px; transform: translateX(var(--dx)) scale(0.3); }
}
.grind-label { font-weight: 600; color: #f0d9b5; font-size: 0.92rem; margin-top: 2px; }

/* ---------- Recommendation card ---------- */
.rec-card {
  background: linear-gradient(135deg, rgba(232,163,61,0.10), rgba(232,163,61,0.02));
  border: 1px solid rgba(232,163,61,0.35); border-radius: 16px; padding: 16px 18px;
}
.rec-card.empty { color: #9c8366; text-align: center; font-size: 0.9rem; padding: 28px 18px; }
.rec-header { font-size: 0.82rem; font-weight: 700; color: #c9a876; margin-bottom: 6px; }
.rec-coffee { font-size: 1.3rem; font-weight: 800; color: #f5e6d3; margin-bottom: 4px; }
.rec-badge {
  font-size: 0.68rem; font-weight: 700; background: rgba(232,163,61,0.25);
  color: #f0d9b5; border-radius: 999px; padding: 3px 10px; margin-left: 8px; vertical-align: middle;
}
.rec-reason { font-size: 0.86rem; color: #d9c4a3; margin: 6px 0 8px 0; }
.rec-why { list-style: none; padding: 0; margin: 0 0 8px 0; font-size: 0.82rem; color: #e8d8c0; }
.rec-why li { margin: 3px 0; }
.rec-fun { font-style: italic; color: #c9a876; font-size: 0.84rem; }
.slide-up { animation: slideUp 0.45s ease-out; }
@keyframes slideUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }

/* ---------- Order summary ---------- */
.order-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(232,163,61,0.25); border-radius: 16px; padding: 14px 18px; margin-bottom: 10px; }
.order-row { display: flex; justify-content: space-between; padding: 5px 0; font-size: 0.88rem; color: #e8d8c0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.order-row:last-child { border-bottom: none; }
.order-row.head { font-weight: 800; font-size: 1.0rem; color: #f5e6d3; }
.order-row.head .price { color: #e8a33d; font-size: 1.15rem; }
.order-row.extra { color: #c9a876; font-size: 0.78rem; }

.success-card { text-align: center; padding: 30px 18px; background: linear-gradient(135deg, rgba(76,201,131,0.14), rgba(76,201,131,0.03));
  border: 1px solid rgba(76,201,131,0.4); border-radius: 16px; }
.success-icon { font-size: 2.6rem; }
.success-title { font-size: 1.2rem; font-weight: 800; color: #f5e6d3; margin-top: 4px; }
.success-order { color: #4cc983; font-weight: 700; margin-top: 2px; }
.success-sub { color: #b89a78; font-size: 0.82rem; margin-top: 4px; }
.pop { animation: successPop 0.4s cubic-bezier(0.34,1.56,0.64,1); }
@keyframes successPop { 0% { transform: scale(0.7); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }

.offline-note { font-size: 0.72rem; color: #9c8366; text-align: center; margin-top: 4px; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# =============================================================================
# TOP BAR
# =============================================================================
top_l, top_c, top_r = st.columns([1.3, 2, 1])
with top_l:
    st.markdown(
        f'<div class="brand"><span class="brand-icon">☕</span>'
        f'<div><div class="brand-name">{APP_NAME}</div>'
        f'<div class="brand-tag">{TAGLINE}</div></div></div>',
        unsafe_allow_html=True,
    )
with top_c:
    st.markdown(
        '<div class="ai-teaser-row"><div class="robo-badge">🤖</div>'
        '<div class="ai-teaser">Not sure what to order? Our AI barista can pick for you — just 2 taps! ↓</div></div>',
        unsafe_allow_html=True,
    )
with top_r:
    now = datetime.now()
    st.markdown(
        f'<div class="clock">{now.strftime("%I:%M %p")}<br>'
        f'<span>{now.strftime("%a, %d %b %Y")}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# =============================================================================
# MAIN ROW — coffee builder | live preview | AI barista
# =============================================================================
left, center, right = st.columns([1.05, 1.2, 1.15], gap="medium")

with left:
    st.markdown('<div class="panel-title">☕ Choose Your Coffee</div>', unsafe_allow_html=True)
    for name in COFFEE_ORDER:
        data = COFFEES[name]
        selected = st.session_state.coffee == name
        if st.button(f"{data['icon']}  {name}", key=f"coffee_{name}",
                     use_container_width=True,
                     type="primary" if selected else "secondary"):
            st.session_state.coffee = name
            # Safe as direct writes today (this runs before the sliders
            # below in this same pass), but routed through pending_ anyway
            # so it can't break if the layout above the sliders ever changes.
            st.session_state.pending_strength = data["strength"]
            st.session_state.pending_sweetness = data["sweetness"]
            st.session_state.ai_result = None
            st.rerun()
        st.markdown(f'<div class="coffee-desc">{data["desc"]}</div>', unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.slider("🌰 Bean Strength", 0, 100, key="strength")
    st.slider("🍯 Sweetness", 0, 100, key="sweetness")

    t1, t2 = st.columns(2)
    with t1:
        if st.button("🔥 Hot", use_container_width=True, key="temp_hot",
                      type="primary" if st.session_state.temp == "Hot" else "secondary"):
            st.session_state.temp = "Hot"
            st.rerun()
    with t2:
        if st.button("🧊 Iced", use_container_width=True, key="temp_iced",
                      type="primary" if st.session_state.temp == "Iced" else "secondary"):
            st.session_state.temp = "Iced"
            st.rerun()

with center:
    head_l, head_r = st.columns([2, 1])
    with head_l:
        st.markdown('<div class="panel-title">LIVE COFFEE PREVIEW</div>', unsafe_allow_html=True)
    with head_r:
        st.markdown('<div class="live-badge" style="text-align:right;"><span class="live-dot"></span>LIVE</div>', unsafe_allow_html=True)

    st.markdown(
        render_cup(st.session_state.coffee, st.session_state.temp,
                   st.session_state.strength, st.session_state.sweetness),
        unsafe_allow_html=True,
    )
    st.caption("Tip: adjust the sliders or tap Hot / Iced to see the cup change instantly.")

with right:
    st.markdown(
        '<div class="ai-panel-header">🤖 AI Barista</div>'
        '<div class="ai-sub">Let me suggest the perfect coffee for you. Just answer 2 quick questions.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="q-label">1. How are you feeling today?</div>', unsafe_allow_html=True)
    mood_cols = st.columns(3)
    for i, (mood, emoji) in enumerate(MOODS):
        with mood_cols[i % 3]:
            selected = st.session_state.mood == mood
            if st.button(f"{emoji} {mood}", key=f"mood_{mood}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                st.session_state.mood = mood
                st.rerun()

    st.markdown("<div class='q-label'>2. What's the weather like?</div>", unsafe_allow_html=True)
    weather_cols = st.columns(4)
    for i, (weather, emoji) in enumerate(WEATHERS):
        with weather_cols[i]:
            selected = st.session_state.weather == weather
            if st.button(f"{emoji} {weather}", key=f"weather_{weather}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                st.session_state.weather = weather
                st.rerun()

    st.markdown('<div class="q-label">Your age (optional)</div>', unsafe_allow_html=True)
    age_cols = st.columns(4)
    for i, age in enumerate(AGES):
        with age_cols[i]:
            selected = st.session_state.age == age
            if st.button(age, key=f"age_{age}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                st.session_state.age = None if selected else age
                st.rerun()

    ready = bool(st.session_state.mood and st.session_state.weather)
    ai_panel_box = st.empty()
    cta_clicked = st.button("✨ GET MY AI RECOMMENDATION", use_container_width=True,
                             type="primary", disabled=not ready, key="cta_ai")
    if not ready:
        st.caption("Pick a mood + weather above — two taps, that's it.")
    if get_groq_client() is None:
        st.markdown('<div class="offline-note">⚠️ Offline demo mode — add a GROQ_API_KEY in secrets for live AI.</div>', unsafe_allow_html=True)

    if cta_clicked and ready:
        thinking_stages = [
            ("🫘", "Thinking..."),
            ("🔍", "Analyzing mood..."),
            ("🌦️", "Checking weather..."),
            ("🔥", "Roasting recommendation..."),
        ]
        for icon, label in thinking_stages:
            ai_panel_box.markdown(
                f'<div class="thinking-box"><span class="spin-bean">{icon}</span> {label}</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.45)

        result = ai_recommend(st.session_state.mood, st.session_state.weather, st.session_state.age)
        ai_panel_box.empty()

        st.session_state.ai_result = result
        st.session_state.coffee = result["coffee"]
        st.session_state.temp = result.get("temperature", "Hot")
        # strength/sweetness are slider-bound keys and the sliders already
        # rendered earlier in this run (left column, above) — stage the new
        # values as "pending" and let apply_pending_updates() drain them into
        # the real keys at the top of the NEXT run, before sliders re-render.
        st.session_state.pending_strength = STRENGTH_LABEL_TO_PCT.get(result["strength"], 50)
        st.session_state.pending_sweetness = COFFEES[result["coffee"]]["sweetness"]
        st.session_state.brewed = False
        st.rerun()


st.markdown("<div class='divider thick'></div>", unsafe_allow_html=True)


# =============================================================================
# BOTTOM ROW — AI recommendation | order summary + brew
# =============================================================================
bottom_l, bottom_r = st.columns([1.15, 1], gap="medium")

with bottom_l:
    if st.session_state.ai_result:
        r = st.session_state.ai_result
        why_html = "".join(f"<li>✅ {b}</li>" for b in r["why"])
        st.markdown(
            _html(f"""
                <div class="rec-card slide-up">
                  <div class="rec-header">🤖 AI RECOMMENDATION</div>
                  <div class="rec-coffee">{COFFEES[r['coffee']]['icon']} {r['coffee']}
                    <span class="rec-badge">{r['strength']} Strength</span></div>
                  <div class="rec-reason">{r['reason']}</div>
                  <ul class="rec-why">{why_html}</ul>
                  <div class="rec-fun">"{r['fun_line']}"</div>
                </div>
            """),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rec-card empty">🤖 Your AI recommendation will appear here once you answer the 2 questions.</div>',
            unsafe_allow_html=True,
        )

with bottom_r:
    coffee = st.session_state.coffee
    price, extra = compute_price(coffee, st.session_state.strength)

    if not st.session_state.brewed:
        extra_row = (
            f'<div class="order-row extra"><span>Extra shot</span><span>+{CURRENCY}{extra}</span></div>'
            if extra else ""
        )
        st.markdown(
            _html(f"""
                <div class="order-card">
                  <div class="order-row head"><span>Order Summary</span><span class="price">{CURRENCY}{price}</span></div>
                  <div class="order-row"><span>Coffee</span><span>{COFFEES[coffee]['icon']} {coffee}</span></div>
                  <div class="order-row"><span>Strength</span><span>{st.session_state.strength}%</span></div>
                  <div class="order-row"><span>Sweetness</span><span>{st.session_state.sweetness}%</span></div>
                  <div class="order-row"><span>Temperature</span><span>{'🔥 Hot' if st.session_state.temp == 'Hot' else '🧊 Iced'}</span></div>
                  {extra_row}
                </div>
            """),
            unsafe_allow_html=True,
        )
        brew_box = st.empty()
        if st.button("🔥 BREW MY COFFEE", use_container_width=True, type="primary", key="brew_btn"):
            has_milk = COFFEES[coffee]["milk"]
            is_iced = st.session_state.temp == "Iced"

            # Tailor the brew sequence to what's actually in the cup:
            # grind -> extract -> milk (only if the drink has milk) ->
            # ice (only if Iced) or steam (if Hot). The grind stage gets a
            # dedicated visual (falling beans -> spinning gear -> scattering
            # grounds) instead of a single spinning emoji, and a longer
            # dwell time so that animation actually plays out on screen.
            brew_stages = [
                ("plain", "🫘🫘🫘", "Dropping fresh beans..."),
                ("grind", "⚙️", "Grinding fresh beans..."),
                ("plain", "☕", "Extracting rich espresso..."),
            ]
            if has_milk:
                brew_stages.append(("plain", "🥛", "Steaming & pouring milk..."))
            elif coffee == "Americano":
                brew_stages.append(("plain", "💧", "Topping with hot water..."))
            if is_iced:
                brew_stages.append(("plain", "🧊", "Adding ice cubes..."))
            else:
                brew_stages.append(("plain", "💨", "Steam rising..."))

            for kind, icon, label in brew_stages:
                if kind == "grind":
                    brew_box.markdown(
                        _html(f'<div class="brew-box grind-stage">{GRIND_VISUAL_HTML}'
                              f'<div class="grind-label">{label}</div></div>'),
                        unsafe_allow_html=True,
                    )
                    time.sleep(1.3)
                else:
                    brew_box.markdown(
                        f'<div class="brew-box"><span class="brew-icon">{icon}</span> {label}</div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.4)
            brew_box.empty()
            st.session_state.brewed = True
            st.session_state.order_id = random.randint(1000, 9999)
            st.rerun()
    else:
        st.markdown(
            _html(f"""
                <div class="success-card pop">
                  <div class="success-icon">☕</div>
                  <div class="success-title">Ready for Pickup!</div>
                  <div class="success-order">Order #{st.session_state.order_id}</div>
                  <div class="success-sub">Freshly brewed just for you.</div>
                </div>
            """),
            unsafe_allow_html=True,
        )
        if st.button("🔄 Start New Order", use_container_width=True, key="reset_btn"):
            for k, v in DEFAULTS.items():
                if k in WIDGET_BOUND_KEYS:
                    st.session_state[f"pending_{k}"] = v
                else:
                    st.session_state[k] = v
            st.rerun()

