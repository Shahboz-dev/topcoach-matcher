# weights.py — Value elicitation & weight updates (compatible with app.py/score.py)
from typing import Dict, List

WEIGHT_KEYS = [
    "fit_values",     # tags / interests / declared values
    "learning_style", # project vs theory alignment
    "outcomes",       # career outcomes strength
    "prestige",       # brand prestige
    "mentorship",     # mentorship level & small classes
    "city_vibe",      # match city vs campus preference
    "research",       # research intensity
    "cost_total",     # tuition + cost of living affordability
    "scholarship"     # scholarship friendliness
]

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in w.values()) or 1.0
    return {k: max(0.0, v)/total for k,v in w.items()}

def default_weights() -> Dict[str, float]:
    base = {k: 1.0 for k in WEIGHT_KEYS}  # balanced start
    return normalize_weights(base)

# ---------- Scenario tuning ----------
SCENARIO_MAP = {
    "brand_vs_mentorship": {
        "A big-name brand on your diploma":   {"prestige": +0.25, "mentorship": -0.15},
        "Closer mentorship & smaller cohorts":{"prestige": -0.15, "mentorship": +0.25}
    },
    "coop_vs_research": {
        "Paid co-op / internships during studies": {"outcomes": +0.20, "research": -0.10, "learning_style": +0.05},
        "Research track with publications":        {"research": +0.25, "outcomes": -0.10}
    },
    "citynetwork_vs_focus": {
        "Large city with big networking": {"city_vibe": +0.20, "mentorship": -0.05},
        "Quieter campus with strong focus":{"city_vibe": -0.05, "mentorship": +0.10}
    },
    "breadth_vs_depth": {
        "Broad interdisciplinary start": {"learning_style": +0.10, "fit_values": +0.05},
        "Early depth in a single field": {"learning_style": +0.10, "research": +0.05}
    }
}

def apply_scenario_choice(weights: Dict[str,float], code: str, option: str) -> Dict[str,float]:
    delta = SCENARIO_MAP.get(code, {}).get(option)
    if not delta: return weights
    new = {k: weights.get(k,0.0) + delta.get(k,0.0) for k in WEIGHT_KEYS}
    return normalize_weights(new)

# ---------- Priority chips (top-3) ----------
# The user can declare up to 3 top priorities; we gently push weights accordingly.
PRIORITY_POINTS = {
    "Outcomes":      ("outcomes", 50),
    "Prestige":      ("prestige", 30),
    "Mentorship":    ("mentorship", 30),
    "Scholarships":  ("scholarship", 40),
    "Low total cost":("cost_total", 50),
    "City network":  ("city_vibe", 25),
    "Research":      ("research", 35),
    "Learning by projects": ("learning_style", 45)
}

def apply_priority_points(weights: Dict[str,float], top3: List[str]) -> Dict[str,float]:
    points = [50,30,20]
    w = dict(weights)
    for i, p in enumerate(top3[:3]):
        key_pts = PRIORITY_POINTS.get(p)
        if key_pts:
            key, pts = key_pts
            w[key] = w.get(key,0.0) + points[i]/100.0
    # keep values in the picture
    w["fit_values"] = w.get("fit_values",0.0) + 0.15
    return normalize_weights(w)
