# score.py — Multi-objective scoring, feasibility, sensitivity, diversification
from typing import Dict, List, Any, Tuple

SCENARIOS = [
    {
        "code": "brand_vs_mentorship",
        "question": "Which matters more for you right now?",
        "options": ["A big-name brand on your diploma", "Closer mentorship & smaller cohorts"]
    },
    {
        "code": "coop_vs_research",
        "question": "If you had to choose, what’s more appealing?",
        "options": ["Paid co-op / internships during studies", "Research track with publications"]
    },
    {
        "code": "citynetwork_vs_focus",
        "question": "Which sounds better?",
        "options": ["Large city with big networking", "Quieter campus with strong focus"]
    },
    {
        "code": "breadth_vs_depth",
        "question": "Which learning approach suits you?",
        "options": ["Broad interdisciplinary start", "Early depth in a single field"]
    }
]

def scenarios_missing(state: Dict[str,Any]) -> List[Dict[str,str]]:
    done = {s["code"] for s in state.get("scenarios", [])}
    return [s for s in SCENARIOS if s["code"] not in done]

SLOT_ORDER = [
    "interests_theme",    # What are you interested in?
    "values_top",         # What do you value most?
    "scores",             # GPA, IELTS, SAT, etc.
    "budget_tier",        # Budget
    "career_goal",        # Career goal
    "language",           # Preferred language
]

HIGH_COST_REGIONS = {"uk","united kingdom","usa","united states","western europe","singapore"}

def slot_filled(slots: Dict[str,Any], key: str) -> bool:
    if key == "values_top": return bool(slots["values_top"]) and len(slots["values_top"]) >= 2
    if key == "scores":     return any(slots["scores"].values())
    return bool(slots.get(key))

def profile_completeness(slots: Dict[str,Any], scenarios_count: int) -> float:
    # --- Make each answer count more for faster progress bar ---
    filled = sum(1 for k in SLOT_ORDER if slot_filled(slots,k))
    base = filled / max(1, len(SLOT_ORDER)-1)  # ignore one slot for faster fill
    scen_bonus = min(0.20, scenarios_count * 0.10)  # bigger bonus
    return min(1.0, base + scen_bonus)

def confidence_score(slots: Dict[str,Any], conflicts: List[str]) -> float:
    # --- Make confidence bar fill faster ---
    core = ["interests_theme","college_experience","values_top","career_goal","budget_tier","language","scores"]
    core_filled = sum(1 for k in core if slot_filled(slots,k)) / max(1, len(core)-1)
    penalty = 0.10 * len(conflicts)  # smaller penalty
    return max(0.0, min(1.0, core_filled - penalty))

def detect_conflicts(slots: Dict[str,Any]) -> List[str]:
    conflicts=[]
    budget = (slots.get("budget_tier") or "").lower()
    geo = (slots.get("geo_pref") or "").lower()
    pv = slots.get("prestige_vs_stress")
    lv = slots.get("location_vibe")
    goal = slots.get("career_goal")
    sc = slots.get("scores",{})
    gpa = sc.get("gpa"); ielts = sc.get("ielts"); sat = sc.get("sat")

    if pv == "prestige" and budget == "low":
        conflicts.append("prestige_low_budget")
    if geo and budget == "low":
        if any(x in geo for x in HIGH_COST_REGIONS):
            conflicts.append("highcost_region_low_budget")
    if lv == "big-city" and pv == "low-stress":
        conflicts.append("bigcity_lowstress")

    lowish = (gpa is not None and gpa < 3.2) or (ielts is not None and ielts < 6.0) or (sat is not None and sat < 1200)
    if lowish and (goal in {"top-company","research/grad-school","high-income"} or pv=="prestige"):
        conflicts.append("ambition_score_gap")

    if slots.get("family_pressure") == "present" and goal:
        conflicts.append("family_vs_interest")
    return conflicts

def _to_float(x):
    try:
        s = str(x).strip()
        if s == "": return None
        return float(s)
    except: return None

def _to_int(x):
    try:
        s = str(x).strip()
        if s == "": return None
        return int(float(s))
    except: return None

def load_programs_from_df(df) -> List[Dict[str,Any]]:
    recs=[]
    for _, row in df.iterrows():
        tags = {t.strip().lower() for t in str(row.get("tags","")).split(",") if t.strip()}
        recs.append({
            "university": row["university"],
            "major": row["major"],
            "region": row.get("region",""),
            "country": row.get("country",""),
            "tuition_level": str(row.get("tuition_level","")).lower(),
            "teaching_style": str(row.get("teaching_style","")).lower(),
            "language": str(row.get("language","")).lower() or "en",
            "min_gpa": _to_float(row.get("min_gpa","")),
            "min_ielts": _to_float(row.get("min_ielts","")),
            "min_sat": _to_int(row.get("min_sat","")),
            "tags": tags,
            "city_size": str(row.get("city_size","")).lower(),
            "class_size": str(row.get("class_size","")).lower(),
            "project_ratio": _to_float(row.get("project_ratio","")),
            "theory_ratio": _to_float(row.get("theory_ratio","")),
            "coop_internships": str(row.get("coop_internships","")).lower(),
            "mentorship_level": _to_float(row.get("mentorship_level","")),
            "competition_intensity": _to_float(row.get("competition_intensity","")),
            "career_outcomes_strength": _to_float(row.get("career_outcomes_strength","")),
            "research_intensity": _to_float(row.get("research_intensity","")),
            "cost_of_living_level": str(row.get("cost_of_living_level","")).lower(),
            "scholarship_friendly": _to_float(row.get("scholarship_friendly","")),
            "duration_years": _to_float(row.get("duration_years","")),
            "brand_prestige": _to_float(row.get("brand_prestige",""))
        })
    return recs

def _norm_0_1(x: float, maxv: float = 5.0) -> float:
    if x is None: return 0.5
    return max(0.0, min(1.0, x / maxv))

def value_fit(student: Dict[str,Any], prog: Dict[str,Any]) -> float:
    st_tags = set(student.get("tags", set()))
    vals = set((student.get("slots",{}).get("values_top") or []))
    it = student["slots"].get("interests_theme")
    ls = student["slots"].get("college_experience")
    pseudo = set()
    if it: pseudo.add(it)
    if ls: pseudo.add(ls)
    intersect = len((st_tags | vals | pseudo).intersection(prog["tags"]))
    base = intersect / max(1, len(prog["tags"]) or 1)
    return max(0.0, min(1.0, base))

def style_fit(student: Dict[str,Any], prog: Dict[str,Any]) -> float:
    pref = student["slots"].get("college_experience")
    p = _norm_0_1(prog.get("project_ratio"))
    t = _norm_0_1(prog.get("theory_ratio"))
    if pref == "project-based": return 0.65*p + 0.35*(1-t)
    if pref == "theory-heavy":  return 0.65*t + 0.35*(1-p)
    if pref == "team-collab":   return _norm_0_1(prog.get("mentorship_level"))
    return 0.5 + 0.25*(p - t)

def outcomes_fit(prog):   return _norm_0_1(prog.get("career_outcomes_strength"))
def prestige_fit(prog):   return _norm_0_1(prog.get("brand_prestige"))

def mentorship_fit(prog):
    m = _norm_0_1(prog.get("mentorship_level"))
    cl = prog.get("class_size","")
    inv_cls = 1.0 if cl=="small" else (0.65 if cl=="medium" else 0.35)
    return 0.6*m + 0.4*inv_cls

def city_vibe_fit(student, prog):
    pref = student["slots"].get("location_vibe")
    c = prog.get("city_size","")
    if not pref or pref == "No preference": return 0.6
    if pref == "big-city":
        return 1.0 if c in {"big","medium"} else 0.4
    if pref == "campus":
        return 1.0 if c in {"campus","small"} else 0.5
    return 0.6

def research_fit(prog):  return _norm_0_1(prog.get("research_intensity"))

def affordability_fit(student, prog) -> float:
    budget = (student["slots"].get("budget_tier") or "").lower()
    tuition = prog.get("tuition_level","")
    col = prog.get("cost_of_living_level","")
    def tier_to_num(x):
        x = (x or "").lower()
        return {"low":1,"medium":2,"med":2,"high":3}.get(x,2)
    b = tier_to_num(budget); t = tier_to_num(tuition); c = tier_to_num(col)
    delta = (b - (t + (c-2)*0.5))
    if delta >= 1: return 1.0
    if delta == 0: return 0.8
    if delta == -1: return 0.55
    if delta <= -2: return 0.25
    return 0.6

def scholarship_fit(prog): return _norm_0_1(prog.get("scholarship_friendly"))

def below_min_scores(student, prog) -> Tuple[bool,List[str]]:
    s = student["slots"]["scores"]
    cautions=[]
    fail=False
    if prog.get("min_gpa") is not None and s.get("gpa") is not None and s["gpa"] < prog["min_gpa"]:
        fail=True; cautions.append(f"GPA below typical ({s['gpa']} < {prog['min_gpa']})")
    if prog.get("min_ielts") is not None and s.get("ielts") is not None and s["ielts"] < prog["min_ielts"]:
        fail=True; cautions.append(f"IELTS below typical ({s['ielts']} < {prog['min_ielts']})")
    if prog.get("min_sat") is not None and s.get("sat") is not None and s["sat"] < prog["min_sat"]:
        fail=True; cautions.append(f"SAT below typical ({s['sat']} < {prog['min_sat']})")
    return fail, cautions

def geo_lang_penalty(student, prog) -> float:
    lang = (student["slots"].get("language") or "").lower()
    prog_lang = (prog.get("language") or "en").lower()
    if lang in {"en","english"}:
        return 1.0 if "en" in prog_lang else 0.8
    if lang in {"ru","russian"}:
        return 1.0 if "ru" in prog_lang else 0.8
    return 1.0

def uncertainty_penalty(student) -> float:
    core = ["interests_theme","college_experience","values_top","career_goal","budget_tier","language","scores"]
    missing = sum(1 for k in core if not (k=="scores" and any(student["slots"]["scores"].values())) and not student["slots"].get(k))
    return min(0.30, 0.06 * missing)

def compute_components(program: Dict[str,Any], student: Dict[str,Any]) -> Dict[str,float]:
    return {
        "fit_values":     value_fit(student, program),
        "learning_style": style_fit(student, program),
        "outcomes":       outcomes_fit(program),
        "prestige":       prestige_fit(program),
        "mentorship":     mentorship_fit(program),
        "city_vibe":      city_vibe_fit(student, program),
        "research":       research_fit(program),
        "cost_total":     affordability_fit(student, program),
        "scholarship":    scholarship_fit(program)
    }

def weighted_sum(components: Dict[str,float], weights: Dict[str,float]) -> float:
    return sum(components.get(k,0.0) * weights.get(k,0.0) for k in weights.keys())

def explain_reasoning(components: Dict[str,float]) -> str:
    top = sorted(components.items(), key=lambda x: x[1], reverse=True)[:3]
    names = {
        "learning_style":"project/learning style",
        "outcomes":"career outcomes",
        "prestige":"brand prestige",
        "mentorship":"mentorship & small classes",
        "city_vibe":"city/campus vibe",
        "research":"research track",
        "cost_total":"affordability",
        "scholarship":"scholarship odds",
        "fit_values":"values & interests"
    }
    bits = [names.get(k,k.replace("_"," ")) for k,_ in top]
    return "Top drivers: " + ", ".join(bits)

def sensitivity_flag(ranked: List[Tuple[str,float]]) -> str:
    if len(ranked) < 2: return "stable"
    top, second = ranked[0][1], ranked[1][1]
    return "close-call" if (top - second) < 0.05 else "stable"

def diversify(top_majors: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not top_majors: return top_majors
    first_dom = max(top_majors[0]["components"], key=lambda k: top_majors[0]["components"][k])
    out=[top_majors[0]]
    for item in top_majors[1:]:
        dom = max(item["components"], key=lambda k: item["components"][k])
        if dom != first_dom or len(out) == 1:
            out.append(item)
        if len(out) >= 3: break
    return out

def recommend_programs(programs: List[Dict[str,Any]], state: Dict[str,Any], top_n=3, unis_per_major=2) -> List[Dict[str,Any]]:
    weights = state.get("weights", {})
    student = {"slots": state["slots"], "tags": state.get("tags", set())}

    scored = []
    for p in programs:
        comps = compute_components(p, student)
        raw = weighted_sum(comps, weights)
        pen = (1.0 - uncertainty_penalty(state)) * geo_lang_penalty(state, p)
        fail, cautions = below_min_scores(state, p)
        if fail:
            fit = raw * 0.15 * pen
            cautions.append("Below typical cutoff — consider foundation year or alternatives.")
        else:
            fit = raw * pen
        scored.append({
            "program": p,
            "components": comps,
            "fit": float(fit),
            "cautions": cautions
        })

    by_major: Dict[str, List[Dict[str,Any]]] = {}
    for item in scored:
        by_major.setdefault(item["program"]["major"], []).append(item)

    major_rank = []
    for major, plist in by_major.items():
        plist.sort(key=lambda x: x["fit"], reverse=True)
        top = plist[:unis_per_major]
        major_fit = top[0]["fit"]
        ranked = [(x["program"]["university"], x["fit"]) for x in plist[:3]]
        stab = sensitivity_flag(ranked)
        reason = explain_reasoning(top[0]["components"])
        major_rank.append({
            "name": major,
            "fit": major_fit,
            "components": top[0]["components"],
            "stability": stab,
            "universities": [{
                "name": t["program"]["university"],
                "reason": f"{t['program']['teaching_style'].replace('-',' ')}; {t['program']['country']} — {reason.lower()}"
            } for t in top],
            "cautions": top[0]["cautions"],
            "reasoning": reason
        })

    major_rank.sort(key=lambda x: x["fit"], reverse=True)
    major_rank = diversify(major_rank)
    return major_rank[:top_n]
