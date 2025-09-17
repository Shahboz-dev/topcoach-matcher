# app.py — TopCoach Major/Uni Match (Open-ended, no helper answers)
# Free-tier Groq, FastAPI + static web UI
# ----------------------------------------------------------

import os, json, re, traceback
from typing import Dict, List, Any, Optional, Tuple
import os
import re
import json
import traceback
import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from weights import (
    default_weights, apply_scenario_choice, apply_priority_points,
    normalize_weights, WEIGHT_KEYS
)
from score import (
    load_programs_from_df, recommend_programs,
    profile_completeness, confidence_score, SLOT_ORDER,
    detect_conflicts, SCENARIOS
)

# -------------------------
# Config / constants
# -------------------------
MIN_RESULTS = 3
UNIS_PER_MAJOR = 2

# Conversation pacing: encourage ~10–15 turns before recommending

# --- Aggressive fast-track for shortlist ---
HARD_STAGE_MIN_TURNS = 3
MIN_SLOT_COUNT_FOR_SYNTHESIS = 1
MIN_SLOT_COUNT_FOR_RECOMMEND = 1
MIN_SCENARIOS_REQUIRED = 0
PROFILE_TARGET = 0.1  # much lower
CONFIDENCE_TARGET = 0.1

# Groq (free-tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL_NAME", "llama3-8b-8192")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY in your .env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

CSV_PATH = os.getenv("CSV_PATH", "majors.csv")
if not os.path.exists(CSV_PATH):
    alt = "majors.sample.csv"
    if os.path.exists(alt):
        CSV_PATH = alt
    else:
        raise RuntimeError("CSV not found. Add majors.csv or majors.sample.csv next to app.py")

# Load data
df = pd.read_csv(CSV_PATH).fillna("")
programs = load_programs_from_df(df)

# -------------------------
# Session shape
# -------------------------
DEFAULT_SLOTS = {
    "interests_theme": None,        # build / analyze / create / help
    "college_experience": None,     # project-based / theory-heavy / team-collab
    "values_top": [],               # impact / prestige / security / freedom / innovation
    "career_goal": None,            # high-income / top-company / entrepreneurship / research/grad-school / social-impact
    "budget_tier": None,            # low / medium / high
    "language": None,               # en / ru / local language
    "scores": {"gpa": None, "ielts": None, "sat": None, "act": None, "toefl": None},
    "prestige_vs_stress": None,     # prestige / low-stress
    "location_vibe": None,          # big-city / campus
    "geo_pref": None,               # region preference (optional, asked late)
    "family_pressure": None         # present / None
}

SESSIONS: Dict[str, Dict[str, Any]] = {}

# -------------------------
# Schemas
# -------------------------
class UniversityOption(BaseModel):
    name: str
    reason: str

class MajorSuggestion(BaseModel):
    name: str
    fit_score: float
    reasoning: str
    universities: List[UniversityOption] = Field(default_factory=list)
    components: Dict[str, float] = Field(default_factory=dict)
    cautions: List[str] = Field(default_factory=list)
    stability: str = "stable"

class ChatResponse(BaseModel):
    reply: str
    next_question: Optional[str] = None
    stage: str
    suggested_majors: List[MajorSuggestion] = Field(default_factory=list)
    profile_pct: int = 0
    confidence_pct: int = 0
    conflicts: List[str] = Field(default_factory=list)

class ChatRequest(BaseModel):
    user_id: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StartRequest(BaseModel):
    user_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProfileOut(BaseModel):
    tags: List[str]
    values: List[str]
    strengths: List[str]
    holland_codes: List[str]
    slots: Dict[str, Any]
    turns: int
    stage: str
    conflicts: List[str]
    profile_pct: int
    confidence_pct: int
    scenarios: List[Dict[str,str]]
    weights: Dict[str, float]
    asked_topics: List[str]

# -------------------------
# LLM helpers (reflection + next question)
# -------------------------

# --- Make LLM questions punchier and clarify off-topic/unclear answers ---
EXPLORATION_SYS = """
You are a world-class university major consultant.
Tone: warm, human, concise. Never list options or templates. No bullet lists.
Reflect the student's message in 1 short, punchy sentence, acknowledging feelings, themes, or clues.
If the answer is off-topic, unclear, or not related to university/major selection, gently redirect with a clarifying statement.
Also infer compact tags (values/strengths/interests words) and any possible Holland/values/strength clues.
Return STRICT JSON:
{
    "reply": str,
    "holland_codes": [str],
    "values": [str],
    "strengths": [str],
    "tags": [str]
}
"""

NEXTQ_SYS = """
You are a seasoned consultant. Ask ONE very short, direct, punchy follow-up question (max 12 words).
Goal: get the key info in as few questions as possible (5-6 max).
- If the last answer is off-topic or unclear, ask a clarifying question or gently redirect.
- missing_slots: which information is still needed (ordered by priority).
- asked_topics: topics already asked recently—avoid repeating.
- conflicts: important tensions to surface; if non-empty, prefer to ask about the first conflict gracefully.
- Constraints: no multiple questions, no bullets, no options, no checkboxes, no templates, no "choose A/B".
- Delay geography ("geo_pref") until late. Ask budget/scores gently and at most once unless student asks.
- Ask like a human: contextual, brief, and specific to the student's last message.
Return STRICT JSON:
{ "question": str, "topic": str }
Where topic ∈ {"interests_theme","college_experience","values_top","career_goal","budget_tier","language","scores","prestige_vs_stress","location_vibe","geo_pref","family_pressure","conflict","wrapup"}.
"""

def _render_history(hist: List[Dict[str,str]]) -> str:
    return "\n".join(f"{'Student' if h['role']=='user' else 'TopCoach'}: {h['content']}" for h in hist[-12:])

def groq_chat_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    payload = {"model": MODEL, "messages":[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], "temperature": 0.55}
    r = requests.post(GROQ_URL, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.S)
        if m: return json.loads(m.group(0))
        raise RuntimeError(f"LLM returned non-JSON:\n{content}")

def safe_reflect(history: List[Dict[str,str]], latest: str) -> Dict[str, Any]:
    prompt = f"""Conversation so far:
{_render_history(history)}

Latest student message:
"{latest}"

Return JSON with keys: reply, holland_codes, values, strengths, tags."""
    fallback = {"reply":"Thanks — I’m tracking what matters to you and how you like to learn.", "holland_codes":[], "values":[], "strengths":[], "tags":["exploration"]}
    try:
        return groq_chat_json(EXPLORATION_SYS, prompt)
    except Exception as e:
        print("[LLM JSON ERROR - reflect]", e)
        traceback.print_exc()
        return fallback

def next_question_llm(state: Dict[str,Any], last_student_msg: str) -> Tuple[Optional[str], Optional[str]]:
    slots = state["slots"]
    # Compute missing slots in priority order
    missing = []
    for k in SLOT_ORDER:
        if k == "values_top":
            if not (slots["values_top"] and len(slots["values_top"]) >= 2): missing.append(k)
        elif k == "scores":
            if not any(slots["scores"].values()): missing.append(k)
        else:
            if not slots.get(k): missing.append(k)

    asked_topics = list(state.get("asked_topics", []))[-6:]
    conflicts = detect_conflicts(slots)

    # Compose user prompt
    up = {
        "history": _render_history(state["history"]),
        "missing_slots": missing,
        "asked_topics": asked_topics,
        "conflicts": conflicts,
        "last_student_message": last_student_msg
    }
    try:
        out = groq_chat_json(NEXTQ_SYS, json.dumps(up, ensure_ascii=False))
        q = (out.get("question") or "").strip()
        t = (out.get("topic") or "").strip()
        # --- Make questions less deep: if too many turns, stop asking for more details ---
        # If already asked 3 or more questions, stop asking further unless it's a wrapup
        if state.get("turns", 0) >= 3 and t not in ("wrapup",):
            return None, None
        if not q:
            return None, None
        return q, t
    except Exception as e:
        print("[LLM JSON ERROR - nextQ]", e)
        traceback.print_exc()
        return None, None

# -------------------------
# Parsing helpers (free-text extraction)
# -------------------------
def _to_float(x):
    try:
        s = str(x).strip()
        if s == "": return None
        return float(s)
    except Exception:
        return None

def extract_numeric_constraints(text: str) -> Dict[str, Any]:
    t = text.lower()
    out: Dict[str, Any] = {}
    m = re.search(r'\bgpa[:\s]*([0-4](?:\.\d{1,2})?)', t)
    if m: out["gpa"] = float(m.group(1))
    m = re.search(r'\bielts[:\s]*([4-9](?:\.\d)?)', t)
    if m: out["ielts"] = float(m.group(1))
    m = re.search(r'\btoefl[:\s]*(\d{2,3})', t)
    if m: out["toefl"] = int(m.group(1))
    m = re.search(r'\bsat[:\s]*(\d{3,4})', t)
    if m: out["sat"] = int(m.group(1))
    m = re.search(r'\bact[:\s]*(\d{1,2})\b', t)
    if m: out["act"] = int(m.group(1))
    return out

def soft_constraint_hints(text: str) -> Dict[str, Any]:
    t = text.lower()
    out: Dict[str, Any] = {}
    # language hints
    if "english" in t or re.search(r'\ben\b', t): out["language"] = "en"
    if "russian" in t or "rus" in t: out["language"] = "ru"
    if "local language" in t: out["language"] = "local"
    # budget hints
    if any(k in t for k in ["low budget","cheap","arzon","affordable"]): out["tuition_level"] = "low"
    if any(k in t for k in ["mid budget","medium budget","o'rtacha","orta","medium-cost"]): out["tuition_level"] = "medium"
    if any(k in t for k in ["high budget","expensive","qimmat"]): out["tuition_level"] = "high"
    # geography hints (asked late anyway)
    if "western europe" in t: out["geo_pref"] = "Western Europe"
    elif "europe" in t: out["geo_pref"] = "Europe"
    if "usa" in t or "us " in t or "america" in t: out["geo_pref"] = "USA"
    if "uk" in t or "england" in t: out["geo_pref"] = "UK"
    if "singapore" in t: out["geo_pref"] = "Singapore"
    return out

def extract_interests_theme(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["build","make","hands-on","handson","lab","engineer","prototype","hardware","project","practical"]): return "build"
    if any(k in t for k in ["analyze","problem","math","data","logic","algorithm","research"]): return "analyze"
    if any(k in t for k in ["create","design","art","write","visual","creative"]): return "create"
    if any(k in t for k in ["help","people","community","teach","leadership","service","impact"]): return "help"
    return None

def extract_college_experience(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["project","studio","hands-on","handson","practical","prototype","lab","portfolio","co-op","internship","case study","team project","group project"]):
        return "project-based"
    if any(k in t for k in ["theory","exam","lecture","textbook","proof","rigor","nazariya"]):
        return "theory-heavy"
    if any(k in t for k in ["team","collaborative","group","jamoa","pair programming"]):
        return "team-collab"
    return None

def extract_values(text: str) -> List[str]:
    t = text.lower()
    vals=[]
    if any(k in t for k in ["impact","help","jamiyat","society"]): vals.append("impact")
    if any(k in t for k in ["prestige","reputation","nufuz"]): vals.append("prestige")
    if any(k in t for k in ["security","stable","barqaror","stability"]): vals.append("security")
    if any(k in t for k in ["freedom","creative","mustaqil","autonomy"]): vals.append("freedom")
    if any(k in t for k in ["innovation","startup","yangilik","novel"]): vals.append("innovation")
    return list(dict.fromkeys(vals))[:3]

def extract_career_goal(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["phd","grad school","professor"]): return "research/grad-school"
    if any(k in t for k in ["startup","entrepreneur"]): return "entrepreneurship"
    if any(k in t for k in ["big tech","faang","top company"]): return "top-company"
    if any(k in t for k in ["impact","help people","society"]): return "social-impact"
    if any(k in t for k in ["money","salary","high pay","pul"]): return "high-income"
    return None

def extract_prestige_vs_stress(text: str) -> Optional[str]:
    t = text.lower()
    if "prestige" in t or "nufuz" in t: return "prestige"
    if "low-stress" in t or "relaxed" in t or "stress yo'q" in t or "less stress" in t: return "low-stress"
    return None

def extract_location_vibe(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["big city","megapolis","large city","katta shahar"]): return "big-city"
    if any(k in t for k in ["campus","community","quiet town","kichik shahar","small town"]): return "campus"
    return None

def extract_family_pressure(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in ["parents want","ota-onam","dad wants","mom wants","oilam xohlaydi","family wants"]): return "present"
    return None

# Scenario extraction (from open-ended text)
SCENARIO_KEYWORDS = {
    "brand_vs_mentorship": {
        "A big-name brand on your diploma":   ["brand","big-name","reputation","prestige"],
        "Closer mentorship & smaller cohorts":["mentor","mentorship","small class","cohort","close attention"]
    },
    "coop_vs_research": {
        "Paid co-op / internships during studies": ["co-op","coop","internship","placement","industry experience"],
        "Research track with publications":        ["research","publication","paper","lab research"]
    },
    "citynetwork_vs_focus": {
        "Large city with big networking": ["big city","network","business hub","metropolis"],
        "Quieter campus with strong focus":["quiet campus","tight-knit","focused","small town","campus community"]
    },
    "breadth_vs_depth": {
        "Broad interdisciplinary start": ["broad","interdisciplinary","explore many","generalist"],
        "Early depth in a single field": ["depth","specialize early","specialization"]
    }
}

def scenario_extract(text: str) -> List[Tuple[str,str]]:
    t = text.lower()
    hits=[]
    for code, opts in SCENARIO_KEYWORDS.items():
        for opt, kws in opts.items():
            if any(k in t for k in kws):
                hits.append((code, opt))
                break
    return hits

# -------------------------
# Opener
# -------------------------
OPENER_TEXT = (
    "Salom! I’m your TopCoach consultant. To start, tell me about a time learning or building something felt exciting for you — what made it fun?"
)

# -------------------------
# Next-question planner (no helper answers; avoids repetition)
# -------------------------
def plan_next_question_open(state: Dict[str,Any], last_student_msg: str) -> Optional[Tuple[str,str]]:
    # 1) If there is a conflict, surface it gracefully
    conflicts = detect_conflicts(state["slots"])
    if conflicts:
        # Let LLM phrase a single conflict question
        q, topic = next_question_llm(state, last_student_msg)
        if q and topic:
            # ensure we actually touch a conflict
            if topic == "conflict" or "conflict" not in state.get("asked_topics", []):
                return q, topic

    # 2) Otherwise, ask LLM to pick next missing slot in a human way
    q, topic = next_question_llm(state, last_student_msg)
    if not q:
        # Only ask wrapup if not already asked
        if "wrapup" not in state.get("asked_topics", []):
            return ("Before I shortlist, is there any must-have or deal-breaker I should respect (scholarships, co-op, city vs campus, or family expectations)?", "wrapup")
        else:
            return None, None
    return q, topic

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="TopCoach Major Match (Open-ended)", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

if os.path.isdir("web"):
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/profile/{user_id}", response_model=ProfileOut)
def profile(user_id: str):
    s = SESSIONS.get(user_id)
    if not s:
        return ProfileOut(tags=[], values=[], strengths=[], holland_codes=[],
                          slots=DEFAULT_SLOTS, turns=0, stage="none",
                          conflicts=[], profile_pct=0, confidence_pct=0,  # force 0%
                          scenarios=[], weights={k:0.0 for k in WEIGHT_KEYS},
                          asked_topics=[])
    confs = detect_conflicts(s["slots"])
    pc = profile_completeness(s["slots"], len(s.get("scenarios",[])))
    conf = confidence_score(s["slots"], confs)
    return ProfileOut(
        tags=sorted(list(s["tags"])),
        values=sorted(list(s.get("values", set()))),
        strengths=sorted(list(s.get("strengths", set()))),
        holland_codes=sorted(list(s.get("holland_codes", set()))),
        slots=s["slots"],
        turns=s["turns"],
        stage=s["stage"],
        conflicts=confs,
        profile_pct=int(pc*100),
        confidence_pct=int(conf*100),
        scenarios=s.get("scenarios",[]),
        weights=s.get("weights", default_weights()),
        asked_topics=list(s.get("asked_topics", []))
    )

    

@app.post("/start", response_model=ChatResponse)
def start(req: StartRequest):
    SESSIONS[req.user_id] = {
        "history": [],
        "tags": set(),
        "holland_codes": set(),
        "values": set(),
        "strengths": set(),
        "constraints": {},
        "slots": DEFAULT_SLOTS.copy(),
        "turns": 0,
        "stage": "exploration",
        "scenarios": [],
        "weights": default_weights(),
        "asked_topics": [],
        "last_question": None,
        "recommendation_emitted": False
    }
    s = SESSIONS[req.user_id]
    s["history"].append({"role":"assistant","content":OPENER_TEXT})

    # Always start with 0% for both profile and confidence
    return ChatResponse(
        reply=OPENER_TEXT,
        next_question=None,  # we ask next after first user reply
        stage=s["stage"],
        suggested_majors=[],
        profile_pct=0,
        confidence_pct=0,
        conflicts=[]
    )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.user_id not in SESSIONS:
        return start(StartRequest(user_id=req.user_id, metadata={}))
    s = SESSIONS[req.user_id]

    # --- If user just answered the wrapup/dealbreaker question with a direct/negative answer, trigger shortlist ---
    last_topic = s.get("asked_topics", [])[-1] if s.get("asked_topics") else None
    user_msg_lower = req.message.strip().lower()
    if last_topic == "wrapup" and user_msg_lower in {"no", "none", "nothing", "nope", "no dealbreakers", "no dealbreaker", "just shortlist", "shortlist", "no, just shortlist", "no thanks", "no thank you", "no preferences", "no must-have", "no must haves", "no must-have or deal-breaker", "no must-haves or deal-breakers"}:
        slots = s["slots"]
        s["stage"] = "recommend"
        s["recommendation_emitted"] = True
        confs = detect_conflicts(slots)
        pc = profile_completeness(slots, len(s.get("scenarios",[])))
        cf = confidence_score(slots, confs)
        suggestions = recommend_programs(programs, s, top_n=MIN_RESULTS, unis_per_major=UNIS_PER_MAJOR)
        if suggestions:
            summary = (
                f"I’m ready to shortlist. Based on what you shared, I’m confident in these (profile {int(pc*100)}%, confidence {int(cf*100)}%). "
                f"My top match is {suggestions[0]['name']}; I’ll also include strong alternatives."
            )
            payload = [
                MajorSuggestion(
                    name=item["name"],
                    fit_score=round(float(item["fit"]),2),
                    reasoning=item["reasoning"],
                    universities=[UniversityOption(name=u["name"], reason=u["reason"]) for u in item["universities"]],
                    components=item.get("components",{}),
                    cautions=item.get("cautions",[]),
                    stability=item.get("stability","stable")
                ) for item in suggestions
            ]
            return ChatResponse(
                reply=summary,
                next_question=None,
                stage="recommend",
                suggested_majors=payload,
                profile_pct=int(pc*100),
                confidence_pct=int(cf*100),
                conflicts=confs
            )
    if req.user_id not in SESSIONS:
        return start(StartRequest(user_id=req.user_id, metadata={}))

    s = SESSIONS[req.user_id]
    s["turns"] += 1
    s["history"].append({"role":"user","content":req.message})

    # Passive extraction from free text
    slots = s["slots"]

    # numeric scores
    nums = extract_numeric_constraints(req.message) 
    for k,v in nums.items():
        if v is not None:
            slots["scores"][k] = v

    # soft hints (budget/language/geo)
    soft = soft_constraint_hints(req.message)
    if "tuition_level" in soft and not slots["budget_tier"]: slots["budget_tier"] = soft["tuition_level"]
    if "language" in soft and not slots["language"]:        slots["language"] = soft["language"]
    if "geo_pref" in soft and not slots["geo_pref"]:        slots["geo_pref"] = soft["geo_pref"]

    # qualitative preferences
    slots["interests_theme"]     = slots["interests_theme"]     or extract_interests_theme(req.message)
    slots["college_experience"]  = slots["college_experience"]  or extract_college_experience(req.message)
    vals = extract_values(req.message)
    if vals:
        slots["values_top"] = list(dict.fromkeys(slots["values_top"] + vals))[:3]
    cg = extract_career_goal(req.message)
    if cg: slots["career_goal"] = slots["career_goal"] or cg
    pv = extract_prestige_vs_stress(req.message)
    if pv: slots["prestige_vs_stress"] = slots["prestige_vs_stress"] or pv
    lv = extract_location_vibe(req.message)
    if lv: slots["location_vibe"] = slots["location_vibe"] or lv
    fp = extract_family_pressure(req.message)
    if fp: slots["family_pressure"] = slots["family_pressure"] or fp

    # scenario extraction (from free text); update weights if new
    for code, choice in scenario_extract(req.message):
        if not any(scen["code"] == code for scen in s.get("scenarios", [])):
            s["scenarios"].append({"code":code, "choice":choice})
            s["weights"] = apply_scenario_choice(s["weights"], code, choice)

    # empathetic reflection
    reflect = safe_reflect(s["history"], req.message)
    s["tags"].update([t.strip().lower() for t in reflect.get("tags", []) if t.strip()])
    s["holland_codes"].update([t.strip().upper() for t in reflect.get("holland_codes", []) if t.strip()])
    s["values"].update([t.strip().lower() for t in reflect.get("values", []) if t.strip()])
    s["strengths"].update([t.strip().lower() for t in reflect.get("strengths", []) if t.strip()])
    reply_text = reflect.get("reply","Thanks — I’m tracking what matters to you.")

    # Stage progression & readiness
    confs = detect_conflicts(slots)
    pc = profile_completeness(slots, len(s.get("scenarios",[])))
    cf = confidence_score(slots, confs)

    filled = sum(1 for k in SLOT_ORDER if (k!="scores" and bool(slots.get(k))) or (k=="scores" and any(slots["scores"].values())))
    if s["stage"] in ("exploration",) and s["turns"] >= HARD_STAGE_MIN_TURNS and filled >= MIN_SLOT_COUNT_FOR_SYNTHESIS:
        s["stage"] = "synthesis"

    ready = (
        (s["stage"] in {"synthesis","recommend"}) and
        pc >= PROFILE_TARGET and cf >= CONFIDENCE_TARGET and
        len(s.get("scenarios",[])) >= MIN_SCENARIOS_REQUIRED and
        filled >= MIN_SLOT_COUNT_FOR_RECOMMEND
    )


    # --- Make it easier to emit recommendations after 4 questions ---
    if not ready and s["turns"] >= 4 and filled >= 2:
        s["stage"] = "recommend"
        ready = True

    if ready and not s.get("recommendation_emitted"):
        s["stage"] = "recommend"
        suggestions = recommend_programs(programs, s, top_n=MIN_RESULTS, unis_per_major=UNIS_PER_MAJOR)
        s["recommendation_emitted"] = True
        if suggestions:
            summary = (
                f"I’m ready to shortlist. Based on what you shared, I’m confident in these (profile {int(pc*100)}%, confidence {int(cf*100)}%). "
                f"My top match is {suggestions[0]['name']}; I’ll also include strong alternatives."
            )
            payload = [
                MajorSuggestion(
                    name=item["name"],
                    fit_score=round(float(item["fit"]),2),
                    reasoning=item["reasoning"],
                    universities=[UniversityOption(name=u["name"], reason=u["reason"]) for u in item["universities"]],
                    components=item.get("components",{}),
                    cautions=item.get("cautions",[]),
                    stability=item.get("stability","stable")
                ) for item in suggestions
            ]
            return ChatResponse(
                reply=summary,
                next_question=None,
                stage="recommend",
                suggested_majors=payload,
                profile_pct=int(pc*100),
                confidence_pct=int(cf*100),
                conflicts=confs
            )

    # Ask ONE next question naturally (no helper options)
    q, topic = plan_next_question_open(s, req.message)

    # repetition guard
    if q and s.get("last_question") and q.strip().lower() == s["last_question"].strip().lower():
        # slightly alter to avoid exact repeat
        q = q.replace("Tell me", "Could you share").replace("Could you", "I’m curious, could you")


    if topic:
        asked = s.get("asked_topics", [])
        # Only append if not already present (prevents wrapup from being added multiple times)
        if topic not in asked:
            asked.append(topic)
        s["asked_topics"] = asked[-8:]  # keep recent window

    s["last_question"] = q
    full_reply = reply_text + (("\n\n" + q) if q else "")

    return ChatResponse(
        reply=full_reply,
        next_question=None,  # front-end just shows reply bubbles
        stage=s["stage"],
        suggested_majors=[],
        profile_pct=int(pc*100),
        confidence_pct=int(cf*100),
        conflicts=confs
    )
