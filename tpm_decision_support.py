#!/usr/bin/env python3
"""
Neural Network-Based Decision Support for Trade Promotion Management (TPM)
CSC510: Foundations of Artificial Intelligence - Portfolio Project
Author: Anand Verma

WHAT THIS PROGRAM DOES
    Helps a trade-marketing planner at a Clorox-sized consumer goods company decide
    which promotions to fund. The planner types in (or picks) a promotion and the
    program answers three questions BEFORE any budget is committed:
        1. How much extra volume and profit will it probably create?   (neural network)
        2. Is it allowed under the company's business rules?           (expert system)
        3. What is the best promotion / best calendar to consider?     (search + planning)
    The program never approves or rejects anything. The human planner decides.

COURSE CONCEPTS USED (module topics)
    - Deep learning ........ PyTorch feed-forward network with embedding layers predicts % sales lift
    - Search ............... Genetic algorithm (best single promotion), random search (network tuning),
                             k-nearest-neighbor search (new products with no history)
    - Symbolic planning .... A* search over STRIPS-style operators builds a multi-week promotion calendar
    - Expert system ........ Forward-chaining rule engine with certainty factors and explanations
    - Knowledge representation  First-order-logic style facts, rules with variables, shelf-group hierarchy
    - Classification ....... Positive/negative ROI label, and compliant / needs review / non-compliant label

HOW TO RUN
    pip install numpy torch
    python tpm_decision_support.py --demo       # full walkthrough, no typing needed
    python tpm_decision_support.py              # interactive menu
    python tpm_decision_support.py --selftest   # automatic checks (about 1 minute)
    Optional: --seed N (change random seed), --quick (smaller/faster training)

DATA NOTE
    The Dunnhumby "Complete Journey" file cannot be bundled with this submission, so the program
    generates synthetic promotion history with the same kinds of fields (discount depth, display
    type, feature ad, category, week of year). All prices, margins, caps, and fees are illustrative
    assumptions, not real company numbers.
"""
from __future__ import annotations

import argparse
import copy
import heapq
import itertools
import math
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    sys.exit("This program needs PyTorch and NumPy.  Install them with:  pip install numpy torch")


# =====================================================================================
# SECTION 1 - DOMAIN KNOWLEDGE (the "world" the program reasons about)
# =====================================================================================
DISPLAYS = ["none", "aisle", "endcap"]
DISPLAY_MULT = [1.00, 1.20, 1.45]          # how strongly each display type boosts lift
DISPLAY_FEE = [0.0, 3000.0, 8000.0]        # fixed fee paid to the retailer per promotion event
FEATURE_FEE = 5000.0                       # fixed fee for a feature ad (circular / flyer)

RETAILER_CAPS = {"MassRetailer": 0.35, "GroceryChain": 0.40, "WarehouseClub": 0.25}   # max discount allowed
GROUP_LIMITS = {"Cleaning": 0.35, "Bags": 0.30, "Food & Grill": 0.30, "Home & Personal": 0.35}
PRICE_FLOOR_FRACTION = 0.67                # shelf price may not fall below 67% of list price (MAP-style)
MAX_EVENT_SPEND = 500_000.0                # trade-spend limit for one promotion event
MIN_PROMO_MARGIN = 0.10                    # gross margin left after the discount must stay at least 10% of list price

# name, shelf group, baseline units/week, list price, gross margin, seasonal peak week,
# descriptor = (price tier, purchase frequency, seasonality, stockpile-ability, brand loyalty)
_RAW_CATEGORIES = [
    ("Disinfecting Wipes", "Cleaning", 38000, 5.50, 0.46, 6, (0.5, 0.6, 0.7, 0.7, 0.5)),
    ("Bleach", "Cleaning", 52000, 4.20, 0.40, 28, (0.3, 0.7, 0.3, 0.8, 0.7)),
    ("Laundry Additives", "Cleaning", 30000, 6.80, 0.44, 20, (0.5, 0.5, 0.2, 0.6, 0.5)),
    ("Toilet Bowl Cleaners", "Cleaning", 26000, 4.60, 0.45, 30, (0.4, 0.5, 0.2, 0.5, 0.6)),
    ("Glass & Surface Cleaners", "Cleaning", 24000, 5.20, 0.47, 14, (0.5, 0.5, 0.4, 0.4, 0.6)),
    ("Trash Bags", "Bags", 46000, 7.80, 0.38, 40, (0.6, 0.6, 0.2, 0.8, 0.5)),
    ("Food Storage Bags", "Bags", 34000, 4.90, 0.43, 52, (0.4, 0.6, 0.5, 0.6, 0.5)),
    ("Water Filtration", "Home & Personal", 15000, 18.00, 0.50, 2, (0.9, 0.2, 0.4, 0.3, 0.7)),
    ("Salad Dressing", "Food & Grill", 40000, 3.80, 0.44, 27, (0.3, 0.7, 0.6, 0.5, 0.8)),
    ("Charcoal & Grilling", "Food & Grill", 22000, 9.50, 0.42, 27, (0.6, 0.3, 0.95, 0.7, 0.6)),
    ("Cat Litter", "Home & Personal", 36000, 12.50, 0.41, 30, (0.7, 0.7, 0.1, 0.8, 0.8)),
    ("Natural Personal Care", "Home & Personal", 14000, 6.20, 0.48, 48, (0.6, 0.4, 0.5, 0.3, 0.4)),
]


def _build_categories():
    cats = []
    for i, (name, group, base, price, margin, peak, desc) in enumerate(_RAW_CATEGORIES):
        price_tier, freq, seasonality, stock, loyalty = desc
        # Hidden "true" promotion sensitivity: shoppers who can stockpile and who are less loyal respond more;
        # expensive items and everyday-purchase items respond less.
        sens = 0.60 + 0.75 * stock + 0.55 * (1 - loyalty) - 0.25 * price_tier - 0.20 * freq + 0.04 * math.sin(i * 1.7)
        cats.append(dict(name=name, group=group, base=base, price=price, margin=margin, peak=peak,
                         desc=desc, seas=seasonality, loyal=loyalty, sens=sens))
    return cats


CATS = _build_categories()
CAT_NAMES = [c["name"] for c in CATS]
CAT_INDEX = {c["name"]: i for i, c in enumerate(CATS)}


def season_curve(cat, week, scale):
    """Seasonal multiplier (1.0 = average week); week is 1..52 and wraps around."""
    return 1.0 + scale * cat["seas"] * math.cos(2 * math.pi * (week - cat["peak"]) / 52.0)


def wrap_week(week):
    return ((int(week) - 1) % 52) + 1


def month_of_week(week):
    return min(11, int((wrap_week(week) - 1) * 12 / 52))


# =====================================================================================
# SECTION 2 - DATA: weekly sales history, baseline model, synthetic promotion events
# =====================================================================================
def make_weekly_history(rng, years=2):
    """Two years of non-promoted weekly sales per category (stand-in for retailer sales data)."""
    hist = []
    for cat in CATS:
        rows = []
        for _y in range(years):
            rows.append([cat["base"] * season_curve(cat, w, 0.40) * rng.lognormal(0, 0.05) for w in range(1, 53)])
        hist.append(np.array(rows))
    return hist


class BaselineModel:
    """Step 1 of the design: simple, explainable 'no-promotion' sales forecast (NOT a neural network).
    Average of the same week in prior years, smoothed over three neighboring weeks."""

    def __init__(self, hist):
        self.curve = []
        for series in hist:
            avg = series.mean(axis=0)
            padded = np.concatenate([avg[-1:], avg, avg[:1]])
            self.curve.append(np.convolve(padded, np.ones(3) / 3.0, mode="valid"))

    def predict(self, cat_index, week):
        return float(self.curve[cat_index][wrap_week(week) - 1])


def true_lift_pct(cat_index, week, discount, display, feature):
    """Hidden 'real world' used ONLY to create training data and to grade the program.
    Response is S-shaped in discount depth (little effect when small, flattening when deep), displays and
    feature ads add to it, and peak season amplifies it."""
    cat = CATS[cat_index]
    hill = discount ** 2.2 / (discount ** 2.2 + 0.15 ** 2.2)      # shoppers barely notice small discounts
    lift = cat["sens"] * 150.0 * hill
    lift *= DISPLAY_MULT[display] * (1.0 + 0.25 * feature) * (1.0 + 0.10 * (display == 2) * feature)
    lift *= 1.0 + 0.35 * cat["seas"] * math.cos(2 * math.pi * (week - cat["peak"]) / 52.0)
    return lift


def make_events(rng, baseline, n):
    """Synthetic promotion history shaped like Dunnhumby promotion fields."""
    cols = {k: [] for k in ("cat", "week", "disp", "feat", "disc", "base", "lift")}
    for _ in range(n):
        ci = int(rng.integers(len(CATS)))
        week = int(rng.integers(1, 53))
        disp = int(rng.choice(3, p=[0.50, 0.30, 0.20]))
        feat = int(rng.random() < (0.55 if disp > 0 else 0.25))
        disc = int(rng.integers(5, 41)) / 100.0
        lift = true_lift_pct(ci, week, disc, disp, feat) * rng.lognormal(0, 0.10)
        for k, v in zip(cols, (ci, week, disp, feat, disc, baseline.predict(ci, week), lift)):
            cols[k].append(v)
    out = {k: np.array(v) for k, v in cols.items()}
    out["month"] = np.array([month_of_week(w) for w in out["week"]])
    return out


# =====================================================================================
# SECTION 3 - PROFIT MATH (a deterministic formula, not a neural network)
# =====================================================================================
def profit_arrays(cat_idx, base, lift_pct, disc, disp, feat, weeks):
    """Incremental profit and trade spend for arrays of promotions (vectorized).
    While on promotion ALL units sell at the discounted price, so the discount is paid on baseline units too."""
    price = np.array([CATS[i]["price"] for i in range(len(CATS))])[cat_idx]
    unit_margin = np.array([CATS[i]["margin"] * CATS[i]["price"] for i in range(len(CATS))])[cat_idx]
    lift_units = base * lift_pct / 100.0
    per_week = (base + lift_units) * (unit_margin - disc * price) - base * unit_margin
    fixed = np.array(DISPLAY_FEE)[disp] + FEATURE_FEE * feat
    profit = weeks * per_week - fixed
    spend = weeks * (base + lift_units) * disc * price + fixed
    return profit, spend, weeks * lift_units


# =====================================================================================
# SECTION 4 - DEEP LEARNING: embedding network that predicts % lift
# =====================================================================================
class PromoLiftNet(nn.Module):
    """Entity-embedding MLP. Category, display type and month each get a learned vector, which is
    concatenated with the numeric inputs (discount, feature ad, baseline size) and passed through
    two hidden layers. Output = predicted % lift (divided by 100)."""

    def __init__(self, n_cat, cat_dim, hidden, dropout):
        super().__init__()
        self.cat_emb = nn.Embedding(n_cat, cat_dim)
        self.disp_emb = nn.Embedding(3, 2)
        self.month_emb = nn.Embedding(12, 3)
        layers, prev = [], cat_dim + 2 + 3 + 3
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, disp, month, num):
        x = torch.cat([self.cat_emb(cat), self.disp_emb(disp), self.month_emb(month), num], dim=1)
        return self.mlp(x).squeeze(1)


# =====================================================================================
# SECTION 5 - KNOWLEDGE REPRESENTATION + EXPERT SYSTEM (first-order-logic style rules)
# =====================================================================================
# A FACT is a tuple:  ("Discount", "P1", 0.35)   means   Discount(P1, 0.35)
# A RULE is   conditions -> conclusion.  Strings that start with "?" are VARIABLES, so one rule covers
# every promotion, retailer and week (like "for all p ..." in first-order logic).
# A condition ("test", op, a, b) is a built-in comparison; a and b may be arithmetic like ("-", "?m", 0.03).
ARITH = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
COMPARE = {">": lambda a, b: a > b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b,
           "<=": lambda a, b: a <= b, "!=": lambda a, b: a != b, "==": lambda a, b: a == b}


def is_var(x):
    return isinstance(x, str) and x.startswith("?")


def fact_str(f):
    return f"{f[0]}({', '.join(str(a) for a in f[1:])})"


def evaluate_term(term, binding):
    if is_var(term):
        if term not in binding:
            raise ValueError(f"Rule error: variable {term} used in a test before it was bound")
        return binding[term]
    if isinstance(term, tuple) and term and term[0] in ARITH:
        return round(ARITH[term[0]](evaluate_term(term[1], binding), evaluate_term(term[2], binding)), 6)
    return term


def unify(pattern, fact, binding):
    """Match one rule condition against one fact. Returns the extended variable binding or None."""
    if len(pattern) != len(fact):
        return None
    new = dict(binding)
    for p, f in zip(pattern, fact):
        if is_var(p):
            if p in new:
                if new[p] != f:
                    return None
            else:
                new[p] = f
        elif p != f:
            return None
    return new


@dataclass
class Rule:
    name: str
    desc: str
    conditions: list
    conclusion: tuple
    cf: float = 1.0          # certainty factor (1.0 = hard rule; <1 = soft guideline)

    def text(self):
        parts = []
        for c in self.conditions:
            if c[0] == "test":
                parts.append(f"{c[2]} {c[1]} {c[3]}")
            else:
                parts.append(fact_str(c))
        tail = "" if self.cf >= 1.0 else f"   [certainty {self.cf:.2f}]"
        return f"{self.name}: " + " AND ".join(parts) + f"  ->  {fact_str(self.conclusion)}{tail}"


class Engine:
    """Forward-chaining inference engine: keep applying rules until no new facts appear.
    Every derived fact remembers which rule and which supporting facts produced it (for explanations)."""

    def __init__(self, rules):
        self.rules = rules
        self.facts = set()
        self.index = {}
        self.derivations = {}

    def add_fact(self, fact):
        if fact not in self.facts:
            self.facts.add(fact)
            self.index.setdefault(fact[0], []).append(fact)

    def clone(self):
        e = Engine(self.rules)
        e.facts = set(self.facts)
        e.index = {k: list(v) for k, v in self.index.items()}
        return e

    def _solve(self, conds, binding, support):
        if not conds:
            yield binding, support
            return
        c = conds[0]
        if c[0] == "test":
            if COMPARE[c[1]](evaluate_term(c[2], binding), evaluate_term(c[3], binding)):
                yield from self._solve(conds[1:], binding, support)
            return
        for fact in self.index.get(c[0], ()):
            nb = unify(c, fact, binding)
            if nb is not None:
                yield from self._solve(conds[1:], nb, support + [fact])

    def run(self):
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                for binding, support in list(self._solve(rule.conditions, {}, [])):
                    concl = tuple(binding.get(t, t) if is_var(t) else t for t in rule.conclusion)
                    if concl not in self.facts:
                        self.add_fact(concl)
                        self.derivations[concl] = (rule, binding, support)
                        changed = True

    def why(self, fact, depth=0):
        """Human-readable proof of a derived fact."""
        pad = "    " * depth
        if fact not in self.derivations:
            return [f"{pad}{fact_str(fact)}   (given)"]
        rule, _b, support = self.derivations[fact]
        lines = [f"{pad}{fact_str(fact)}   <- {rule.name}: {rule.desc}"]
        for s in dict.fromkeys(support):
            lines += self.why(s, depth + 1) if s in self.derivations else [f"{pad}    because {fact_str(s)}"]
        return lines


def build_rules():
    t = "test"
    return [
        Rule("R1", "retailer discount cap",
             [("Retailer", "?p", "?r"), ("Discount", "?p", "?d"), ("MaxDiscount", "?r", "?m"), (t, ">", "?d", "?m")],
             ("Violation", "?p", "retailer discount cap", "?m")),
        Rule("R2", "every product inherits its shelf group's discount limit",
             [("Category", "?p", "?c"), ("InGroup", "?c", "?g"), ("GroupLimit", "?g", "?x")],
             ("Limit", "?p", "?x")),
        Rule("R3", "discount above the inherited group limit",
             [("Discount", "?p", "?d"), ("Limit", "?p", "?x"), (t, ">", "?d", "?x")],
             ("Violation", "?p", "shelf-group discount limit", "?x")),
        Rule("R4", "shelf price below the price floor (MAP)",
             [("ShelfPrice", "?p", "?s"), ("Category", "?p", "?c"), ("MinShelfPrice", "?c", "?f"), (t, "<", "?s", "?f")],
             ("Violation", "?p", "price floor", "?f")),
        Rule("R5", "trade spend above the per-event limit",
             [("TradeSpend", "?p", "?s"), ("MaxEventSpend", "?m"), (t, ">", "?s", "?m")],
             ("Violation", "?p", "trade-spend limit", "?m")),
        Rule("R6", "gross margin after the discount below the company minimum",
             [("PromoMargin", "?p", "?x"), ("MinPromoMargin", "?m"), (t, "<", "?x", "?m")],
             ("Violation", "?p", "minimum margin after discount", "?m")),
        Rule("R7", "overlaps a scheduled promotion in the same shelf group at the same retailer",
             [("Promotion", "?p"), ("Scheduled", "?q"), ("Retailer", "?p", "?r"), ("Retailer", "?q", "?r"),
              ("Category", "?p", "?c1"), ("Category", "?q", "?c2"), ("InGroup", "?c1", "?g"), ("InGroup", "?c2", "?g"),
              ("ActiveWeek", "?p", "?w"), ("ActiveWeek", "?q", "?w")],
             ("Review", "?p", "overlaps a scheduled promotion", "?q"), cf=0.70),
        Rule("R8", "discount within 3 points of the retailer cap",
             [("Retailer", "?p", "?r"), ("Discount", "?p", "?d"), ("MaxDiscount", "?r", "?m"),
              (t, ">=", "?d", ("-", "?m", 0.03)), (t, "<=", "?d", "?m")],
             ("Review", "?p", "close to the retailer cap", "?m"), cf=0.50),
        Rule("R9", "deep discounts can train shoppers to wait for sales",
             [("Discount", "?p", "?d"), (t, ">=", "?d", 0.30)],
             ("Review", "?p", "deep discount (30%+)", "?d"), cf=0.35),
        Rule("R10", "big discounts on loyal-customer brands can hurt brand value",
             [("Category", "?p", "?c"), ("HighLoyalty", "?c"), ("Discount", "?p", "?d"), (t, ">", "?d", 0.25)],
             ("Review", "?p", "large discount on a high-loyalty brand", "?d"), cf=0.35),
        Rule("R11", "stacking endcap + feature ad + deep discount is often wasteful",
             [("Display", "?p", "endcap"), ("Feature", "?p", 1), ("Discount", "?p", "?d"), (t, ">=", "?d", 0.25)],
             ("Review", "?p", "stacked tactics", "?d"), cf=0.40),
    ]


def build_static_facts():
    facts = set()
    for cat in CATS:
        facts.add(("InGroup", cat["name"], cat["group"]))
        facts.add(("MinShelfPrice", cat["name"], round(cat["price"] * PRICE_FLOOR_FRACTION, 2)))
        if cat["loyal"] >= 0.7:
            facts.add(("HighLoyalty", cat["name"]))
    for g, x in GROUP_LIMITS.items():
        facts.add(("GroupLimit", g, x))
    for r, x in RETAILER_CAPS.items():
        facts.add(("MaxDiscount", r, x))
    facts.add(("MaxEventSpend", MAX_EVENT_SPEND))
    facts.add(("MinPromoMargin", MIN_PROMO_MARGIN))
    return facts


# Promotions that are already committed on the calendar (they feed rule R7).
COMMITTED_CALENDAR = [
    ("SCH-1", "Trash Bags", "MassRetailer", [27, 28]),
    ("SCH-2", "Bleach", "GroceryChain", [10, 11]),
    ("SCH-3", "Laundry Additives", "MassRetailer", [6]),
    ("SCH-4", "Salad Dressing", "GroceryChain", [27, 28]),
    ("SCH-5", "Water Filtration", "WarehouseClub", [2, 3]),
]


def build_calendar_facts():
    facts = set()
    for pid, cat, retailer, weeks in COMMITTED_CALENDAR:
        facts |= {("Scheduled", pid), ("Category", pid, cat), ("Retailer", pid, retailer)}
        facts |= {("ActiveWeek", pid, wrap_week(w)) for w in weeks}
    return facts


# =====================================================================================
# SECTION 6 - SCENARIOS AND RESULTS
# =====================================================================================
@dataclass
class Scenario:
    category: str
    retailer: str
    start_week: int
    weeks: int
    discount: float          # fraction, e.g. 0.20
    display: str
    feature: int

    def short(self):
        return (f"{self.category} @ {self.retailer}, wk {self.start_week}"
                f"{'' if self.weeks == 1 else '-' + str(wrap_week(self.start_week + self.weeks - 1))}, "
                f"{round(self.discount * 100)}% off, display={self.display}, feature={'yes' if self.feature else 'no'}")


@dataclass
class RuleResult:
    label: str                       # COMPLIANT / NEEDS REVIEW / NON-COMPLIANT
    violations: list = field(default_factory=list)   # (reason, detail)
    reviews: list = field(default_factory=list)      # (reason, detail, certainty)
    review_cf: float = 0.0
    proof: list = field(default_factory=list)


@dataclass
class Evaluation:
    scenario: Scenario
    avg_lift_pct: float
    incr_units: float
    profit: float
    spend: float
    roi: float
    roi_class: str
    rules: RuleResult


def combine_cf(cfs):
    """MYCIN-style certainty-factor combination for several soft guidelines."""
    total = 0.0
    for cf in cfs:
        total = total + cf * (1.0 - total)
    return total


# =====================================================================================
# SECTION 7 - THE SYSTEM: trains the network and offers every decision-support feature
# =====================================================================================
class TPMSystem:
    PENALTY_NONCOMPLIANT = 1e7      # GA fitness penalty (keeps illegal plans out of the answer)
    PENALTY_REVIEW = 2000.0         # small penalty: needs-review plans are allowed but not preferred

    def __init__(self, seed=42, quick=False, verbose=True):
        self.seed, self.quick, self.verbose = seed, quick, verbose
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.say("Building synthetic sales history and baseline model ...")
        self.baseline = BaselineModel(make_weekly_history(self.rng))
        self.data = make_events(self.rng, self.baseline, 5000 if quick else 6000)
        self.rules = build_rules()
        self.base_engine = Engine(self.rules)
        for f in sorted(build_static_facts() | build_calendar_facts(), key=str):    # fixed order = repeatable output
            self.base_engine.add_fact(f)
        self._rule_cache = {}
        self.rule_checks = 0
        self._split_and_normalize()
        self.say("Tuning and training the neural network (random search over settings) ...")
        self._train_with_random_search()
        self._compute_metrics()

    def say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---------------------------------------------------------------- data / network
    def _split_and_normalize(self):
        n = len(self.data["cat"])
        perm = np.random.default_rng(self.seed + 1).permutation(n)
        a, b = int(0.70 * n), int(0.85 * n)
        self.idx = {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}
        logb = np.log(self.data["base"][self.idx["train"]])
        self.lb_mean, self.lb_std = float(logb.mean()), float(logb.std())

    def _tensors(self, cat, disp, month, disc, feat, base):
        num = np.stack([np.asarray(disc) / 0.4, np.asarray(feat, dtype=float),
                        (np.log(np.asarray(base)) - self.lb_mean) / self.lb_std], axis=1)
        return (torch.tensor(np.asarray(cat), dtype=torch.long), torch.tensor(np.asarray(disp), dtype=torch.long),
                torch.tensor(np.asarray(month), dtype=torch.long), torch.tensor(num, dtype=torch.float32))

    def _split_tensors(self, name):
        i, d = self.idx[name], self.data
        x = self._tensors(d["cat"][i], d["disp"][i], d["month"][i], d["disc"][i], d["feat"][i], d["base"][i])
        y = torch.tensor(d["lift"][i] / 100.0, dtype=torch.float32)
        return x, y

    def _fit(self, params, epochs, seed):
        torch.manual_seed(seed)
        model = PromoLiftNet(len(CATS), params["cat_dim"], params["hidden"], params["dropout"])
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
        loss_fn = nn.MSELoss()
        xtr, ytr = self._split_tensors("train")
        xva, yva = self._split_tensors("val")
        n, best_mae, best_state = len(ytr), float("inf"), None
        for _ in range(epochs):
            model.train()
            order = torch.randperm(n)
            for s in range(0, n, 128):
                b = order[s:s + 128]
                opt.zero_grad()
                loss = loss_fn(model(*[t[b] for t in xtr]), ytr[b])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                mae = float((model(*xva) - yva).abs().mean()) * 100.0
            if mae < best_mae:
                best_mae, best_state = mae, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        model.eval()
        return model, best_mae

    def _train_with_random_search(self):
        """Random search (instead of trying every combination) over a few network settings."""
        rs = random.Random(self.seed)
        trials = 3 if self.quick else 5
        epochs = 25 if self.quick else 45
        self.search_log = []
        best = (float("inf"), None, None)
        for t in range(trials):
            params = dict(cat_dim=rs.choice([4, 8, 12]), hidden=rs.choice([(32, 16), (64, 32), (96, 48)]),
                          lr=10 ** rs.uniform(-2.5, -1.7), dropout=rs.choice([0.0, 0.1]), wd=rs.choice([0.0, 1e-5]))
            model, val_mae = self._fit(params, epochs, self.seed + t)
            self.search_log.append((params, val_mae))
            self.say(f"   trial {t + 1}/{trials}: embedding={params['cat_dim']}, hidden={params['hidden']}, "
                     f"lr={params['lr']:.4f}, dropout={params['dropout']}  ->  validation error {val_mae:.2f} points")
            if val_mae < best[0]:
                best = (val_mae, params, model)
        self.val_mae, self.best_params, self.model = best

    def predict_lift(self, cat, disp, month, disc, feat, base):
        """Predicted % lift for arrays of promotions (never below zero)."""
        with torch.no_grad():
            out = self.model(*self._tensors(cat, disp, month, disc, feat, base)).numpy() * 100.0
        return np.maximum(out, 0.0)

    # ---------------------------------------------------------------- model quality
    @staticmethod
    def _features_linear(d, idx):
        cat = np.eye(len(CATS))[d["cat"][idx]]
        disp = np.eye(3)[d["disp"][idx]][:, 1:]
        ang = 2 * np.pi * d["week"][idx] / 52.0
        return np.column_stack([cat, disp, d["feat"][idx], d["disc"][idx], d["disc"][idx] ** 2,
                                np.sin(ang), np.cos(ang)])

    def _compute_metrics(self):
        d, te, tr = self.data, self.idx["test"], self.idx["train"]
        pred_nn = self.predict_lift(d["cat"][te], d["disp"][te], d["month"][te], d["disc"][te], d["feat"][te], d["base"][te])
        cat_mean = np.array([d["lift"][tr][d["cat"][tr] == c].mean() for c in range(len(CATS))])
        pred_mean = cat_mean[d["cat"][te]]
        coef, *_ = np.linalg.lstsq(self._features_linear(d, tr), d["lift"][tr], rcond=None)
        pred_lin = np.maximum(self._features_linear(d, te) @ coef, 0)
        truth = d["lift"][te]
        args = (d["cat"][te], d["base"][te])

        def lift_err(p):
            return float(np.abs(p - truth).mean()), float(np.sqrt(((p - truth) ** 2).mean()))

        def profit_of(lift):
            return profit_arrays(args[0], args[1], lift, d["disc"][te], d["disp"][te], d["feat"][te], 1)[0]

        true_pos = profit_of(truth) > 0

        def roi_stats(p):
            guess_pos = profit_of(p) > 0
            tn = int(((~true_pos) & (~guess_pos)).sum())      # correctly flagged negative-ROI
            fn = int(((true_pos) & (~guess_pos)).sum())       # wrongly flagged negative
            fp = int(((~true_pos) & (guess_pos)).sum())       # negative-ROI missed
            tp = int(((true_pos) & (guess_pos)).sum())
            return dict(acc=(tp + tn) / len(truth), prec_neg=tn / max(tn + fn, 1), rec_neg=tn / max(tn + fp, 1))

        self.metrics = {
            "n_test": len(truth), "pct_positive": float(true_pos.mean()),
            "lift": {"Neural network": lift_err(pred_nn), "Linear regression": lift_err(pred_lin),
                     "Category average (rule of thumb)": lift_err(pred_mean)},
            "roi": {"Neural network": roi_stats(pred_nn), "Linear regression": roi_stats(pred_lin),
                    "Category average (rule of thumb)": roi_stats(pred_mean)},
        }
        self.metrics["cold_start"] = self._cold_start_test()

    # ---------------------------------------------------------------- nearest neighbors (new products)
    @staticmethod
    def _descriptor_matrix():
        m = np.array([c["desc"] for c in CATS], dtype=float)
        return m, m.mean(axis=0), m.std(axis=0) + 1e-9

    def neighbors(self, descriptor, k=5, exclude=None):
        m, mu, sd = self._descriptor_matrix()
        z, q = (m - mu) / sd, (np.asarray(descriptor, dtype=float) - mu) / sd
        dist = np.sqrt(((z - q) ** 2).sum(axis=1))
        order = [i for i in np.argsort(dist) if i != exclude][:k]
        return [(int(i), float(dist[i])) for i in order]

    def _grid_configs(self):
        return [(d, p, f) for d in (0.10, 0.20, 0.30) for p in (0, 1, 2) for f in (0, 1)]

    SEASON_POINTS = (1, 14, 27, 40)       # four weeks spread over the year (cancels out seasonal peaks)

    def _mean_lift_over_grid(self, ci):
        cfg = self._grid_configs()
        n = len(cfg)
        vals = []
        for week in self.SEASON_POINTS:
            base = self.baseline.predict(ci, week)
            vals.append(self.predict_lift([ci] * n, [c[1] for c in cfg], [month_of_week(week)] * n,
                                          [c[0] for c in cfg], [c[2] for c in cfg], [base] * n).mean())
        return float(np.mean(vals))

    def _cold_start_test(self):
        """Leave-one-category-out: pretend each category is new, estimate it from neighbors, compare to the truth."""
        cfg = self._grid_configs()
        truth = [np.mean([true_lift_pct(ci, w, d, p, f) for w in self.SEASON_POINTS for d, p, f in cfg]) for ci in range(len(CATS))]
        learned = [self._mean_lift_over_grid(ci) for ci in range(len(CATS))]
        out = {}
        for k in (1, 3, 5):
            errs = []
            for ci in range(len(CATS)):
                nb = self.neighbors(CATS[ci]["desc"], k=k, exclude=ci)
                errs.append(abs(np.mean([learned[i] for i, _ in nb]) - truth[ci]))
            out[f"{k} neighbor(s)"] = float(np.mean(errs))
        out["Average of all other categories"] = float(np.mean(
            [abs(np.mean([learned[i] for i in range(len(CATS)) if i != ci]) - truth[ci]) for ci in range(len(CATS))]))
        return out

    def estimate_new_product(self, descriptor, price, margin, base_units, k=5, week=26):
        """Cold start: borrow the predicted results of the k most similar existing categories."""
        nb = self.neighbors(descriptor, k=k)
        rows = []
        for cfg_i, (d, p, f) in enumerate(self._grid_configs()):
            lifts = [float(self.predict_lift([ci], [p], [month_of_week(week)], [d], [f], [self.baseline.predict(ci, week)])[0])
                     for ci, _ in nb]
            lift = float(np.mean(lifts))
            unit_margin = margin * price
            units = base_units * lift / 100.0
            profit = (base_units + units) * (unit_margin - d * price) - base_units * unit_margin - DISPLAY_FEE[p] - FEATURE_FEE * f
            rows.append((d, p, f, lift, profit))
        return nb, rows

    def embedding_neighbors(self, ci, k=3):
        w = self.model.cat_emb.weight.detach().numpy()
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
        sims = w @ w[ci]
        order = [i for i in np.argsort(-sims) if i != ci][:k]
        return [(int(i), float(sims[i])) for i in order]

    # ---------------------------------------------------------------- expert system
    def check_rules(self, sc: Scenario, spend: float, pid="NEW"):
        """Run the forward-chaining engine on one promotion and classify it."""
        disc = round(sc.discount, 2)
        key = (sc.category, sc.retailer, sc.start_week, sc.weeks, disc, sc.display, sc.feature, round(spend, -2))
        if key in self._rule_cache and pid == "NEW":
            return self._rule_cache[key]
        self.rule_checks += 1
        cat = CATS[CAT_INDEX[sc.category]]
        e = self.base_engine.clone()
        for f in [("Promotion", pid), ("Category", pid, sc.category), ("Retailer", pid, sc.retailer),
                  ("Discount", pid, disc), ("Display", pid, sc.display), ("Feature", pid, sc.feature),
                  ("ShelfPrice", pid, round(cat["price"] * (1 - disc), 2)), ("TradeSpend", pid, round(spend)),
                  ("PromoMargin", pid, round(cat["margin"] - disc, 2))]:
            e.add_fact(f)
        for w in range(sc.weeks):
            e.add_fact(("ActiveWeek", pid, wrap_week(sc.start_week + w)))
        e.run()
        ordered = sorted(e.facts, key=lambda f: tuple(str(x) for x in f))
        viol = [(f[2], f[3]) for f in ordered if f[0] == "Violation" and f[1] == pid]
        revs = [(f[2], f[3], e.derivations[f][0].cf) for f in ordered if f[0] == "Review" and f[1] == pid]
        cf = combine_cf([r[2] for r in revs])
        label = "NON-COMPLIANT" if viol else ("NEEDS REVIEW" if cf >= 0.5 else "COMPLIANT")
        proof = []
        for f in ordered:
            if f[0] in ("Violation", "Review") and f[1] == pid:
                proof += e.why(f)
        res = RuleResult(label, viol, revs, cf, proof)
        if pid == "NEW":
            self._rule_cache[key] = res
        return res

    # ---------------------------------------------------------------- evaluating promotions
    def score_configs(self, ci, configs):
        """Vectorized neural-network scoring of many (discount_pct, display, feature, weeks, start) configs."""
        rows_cfg, rows_week = [], []
        for k, (dp, disp, feat, weeks, start) in enumerate(configs):
            for w in range(weeks):
                rows_cfg.append(k)
                rows_week.append(wrap_week(start + w))
        rows_cfg = np.array(rows_cfg)
        weeks_arr = np.array(rows_week)
        cfg = np.array(configs, dtype=float)
        disc = cfg[rows_cfg, 0] / 100.0
        disp = cfg[rows_cfg, 1].astype(int)
        feat = cfg[rows_cfg, 2].astype(int)
        base = np.array([self.baseline.predict(ci, w) for w in weeks_arr])
        month = np.array([month_of_week(w) for w in weeks_arr])
        lift = self.predict_lift(np.full(len(rows_cfg), ci), disp, month, disc, feat, base)
        profit_rows, spend_rows, units_rows = profit_arrays(np.full(len(rows_cfg), ci), base, lift, disc, disp, feat, 1)
        n = len(configs)
        fixed = np.array(DISPLAY_FEE)[cfg[:, 1].astype(int)] + FEATURE_FEE * cfg[:, 2]
        # profit_arrays charged the fixed fee on every weekly row; charge it once per event instead
        row_fixed = np.array(DISPLAY_FEE)[disp] + FEATURE_FEE * feat
        profit = np.bincount(rows_cfg, weights=profit_rows + row_fixed, minlength=n) - fixed
        spend = np.bincount(rows_cfg, weights=spend_rows - row_fixed, minlength=n) + fixed
        units = np.bincount(rows_cfg, weights=units_rows, minlength=n)
        lift_avg = np.bincount(rows_cfg, weights=lift, minlength=n) / cfg[:, 3]
        return profit, spend, units, lift_avg

    def scenario_from_config(self, ci, retailer, cfg):
        dp, disp, feat, weeks, start = cfg
        return Scenario(CAT_NAMES[ci], retailer, int(start), int(weeks), dp / 100.0, DISPLAYS[int(disp)], int(feat))

    def fitness(self, ci, retailer, cfg, profit, spend):
        res = self.check_rules(self.scenario_from_config(ci, retailer, cfg), spend)
        pen = self.PENALTY_NONCOMPLIANT if res.label == "NON-COMPLIANT" else (
            self.PENALTY_REVIEW if res.label == "NEEDS REVIEW" else 0.0)
        return profit - pen, res

    def evaluate(self, sc: Scenario, pid="NEW") -> Evaluation:
        ci = CAT_INDEX[sc.category]
        cfg = (round(sc.discount * 100), DISPLAYS.index(sc.display), sc.feature, sc.weeks, sc.start_week)
        profit, spend, units, lift = self.score_configs(ci, [cfg])
        res = self.check_rules(sc, float(spend[0]), pid)
        roi = float(profit[0] / spend[0]) if spend[0] > 0 else 0.0
        return Evaluation(sc, float(lift[0]), float(units[0]), float(profit[0]), float(spend[0]), roi,
                          "POSITIVE ROI" if profit[0] > 0 else "NEGATIVE ROI", res)

    # ---------------------------------------------------------------- SEARCH 1: genetic algorithm
    def all_configs(self, start_weeks, durations):
        return [(dp, disp, feat, wk, st) for dp in range(5, 41) for disp in range(3) for feat in range(2)
                for wk in durations for st in start_weeks]

    def exhaustive_search(self, ci, retailer, start_weeks, durations, top=5):
        """Ground truth for checking the GA: score everything with the network, then check rules
        from the most profitable down, stopping as soon as nothing below can beat the best found."""
        configs = self.all_configs(start_weeks, durations)
        profit, spend, units, lift = self.score_configs(ci, configs)
        order = np.argsort(-profit)
        found, checks = [], 0
        for i in order:
            if len(found) >= top and profit[i] < sorted(f[0] for f in found)[-top]:
                break
            fit, res = self.fitness(ci, retailer, configs[i], profit[i], spend[i])
            checks += 1
            found.append((fit, configs[i], float(profit[i]), float(spend[i]), float(lift[i]), res))
            found.sort(key=lambda x: -x[0])
            found = found[:top]
        return found, len(configs), checks

    def genetic_search(self, ci, retailer, start_weeks, durations, pop=40, gens=25, seed=None):
        """SEARCH METHOD 1 - genetic algorithm. Each 'chromosome' is a promotion plan
        (discount, display, feature ad, duration, start week). Fitness = neural-network profit,
        minus a penalty if the rule base says the plan is not allowed (or needs review)."""
        rnd = random.Random(self.seed if seed is None else seed)
        lo_s, hi_s = min(start_weeks), max(start_weeks)

        def rand_gene():
            return [rnd.randint(5, 40), rnd.randint(0, 2), rnd.randint(0, 1), rnd.choice(durations), rnd.randint(lo_s, hi_s)]

        def score(pop_list):
            cfgs = [tuple(g) for g in pop_list]
            profit, spend, units, lift = self.score_configs(ci, cfgs)
            out = []
            for g, p, s, l in zip(cfgs, profit, spend, lift):
                fit, res = self.fitness(ci, retailer, g, float(p), float(s))
                out.append((fit, g, float(p), float(s), float(l), res))
            return out

        population = [rand_gene() for _ in range(pop)]
        history, evals, seen = [], 0, {}
        for gen in range(gens):
            scored = score(population)
            evals += len(scored)
            for item in scored:
                seen[item[1]] = item
            scored.sort(key=lambda x: -x[0])
            history.append(scored[0][0])
            elite = [list(x[1]) for x in scored[:2]]

            def pick():
                return list(max(rnd.sample(scored, 3), key=lambda x: x[0])[1])

            nxt = elite[:]
            while len(nxt) < pop:
                a, b = pick(), pick()
                child = [a[i] if rnd.random() < 0.5 else b[i] for i in range(5)]
                if rnd.random() < 0.25:
                    child[0] = min(40, max(5, child[0] + rnd.randint(-6, 6)))
                if rnd.random() < 0.15:
                    child[1] = rnd.randint(0, 2)
                if rnd.random() < 0.15:
                    child[2] = 1 - child[2]
                if rnd.random() < 0.15:
                    child[3] = rnd.choice(durations)
                if rnd.random() < 0.20:
                    child[4] = min(hi_s, max(lo_s, child[4] + rnd.randint(-2, 2)))
                nxt.append(child)
            population = nxt
        ranked = sorted(seen.values(), key=lambda x: -x[0])
        return ranked[:5], history, evals

    # ---------------------------------------------------------------- SEARCH 2 + PLANNING: A* calendar
    def build_plan_options(self, retailer, start_week, horizon, per_week=5):
        """For every week and category: the best allowed 1-week promotion (found by scoring all 216
        discount/display/feature combinations and checking rules from the top)."""
        options = {}
        for t in range(horizon):
            week = wrap_week(start_week + t)
            week_opts = []
            for ci in range(len(CATS)):
                best, _n, _c = self.exhaustive_search(ci, retailer, [week], [1], top=1)
                fit, cfg, profit, spend, lift, res = best[0]
                if res.label == "NON-COMPLIANT":
                    continue
                week_opts.append(dict(week=week, ci=ci, cfg=cfg, profit=profit, spend=spend, lift=lift,
                                      base=self.baseline.predict(ci, week), label=res.label))
            week_opts.sort(key=lambda o: -o["profit"])
            options[t] = week_opts[:per_week]
        return options

    @staticmethod
    def _option_value(o, mult):
        """Profit and trade spend of a planned promotion when its lift is scaled by `mult` (repeat-promotion fatigue)."""
        cat = CATS[o["ci"]]
        dp, disp, feat = o["cfg"][0], o["cfg"][1], o["cfg"][2]
        d, base = dp / 100.0, o["base"]
        lift_units = base * (o["lift"] * mult) / 100.0
        unit_margin = cat["margin"] * cat["price"]
        fixed = DISPLAY_FEE[disp] + FEATURE_FEE * feat
        profit = (base + lift_units) * (unit_margin - d * cat["price"]) - base * unit_margin - fixed
        spend = (base + lift_units) * d * cat["price"] + fixed
        return profit, spend

    @staticmethod
    def fatigue(gap):
        """Planning knowledge: a category promoted again soon sells less on promotion (pantry loading)."""
        return 0.70 if gap == 2 else (0.85 if gap == 3 else 1.0)

    def plan_calendar(self, options, horizon, budget, min_events, node_limit=400_000):
        """SYMBOLIC PLANNING with A*.
        State  : (week, week each category was last promoted, trade spend used, events so far)
        Actions: DoNothing(t)  or  Promote(category, t)
        Preconditions of Promote: category not promoted last week (cooldown) AND budget remains
        Effects: records the promotion, uses budget, counts one event (fatigue lowers a repeat's lift)
        Goal   : every week decided, at least `min_events` promotions, budget respected.
        Cost   : negative profit.  Heuristic: the best profit each remaining week could possibly
                 deliver if no rule applied (an optimistic bound, so A* stays optimal)."""
        n_cat = len(CATS)
        best_week = [max([0.0] + [o["profit"] for o in options[t]]) for t in range(horizon)]
        suffix = [0.0] * (horizon + 1)
        for t in range(horizon - 1, -1, -1):
            suffix[t] = suffix[t + 1] + best_week[t]
        start = (0, tuple([-99] * n_cat), 0, 0)
        counter = itertools.count()
        heap = [(-suffix[0], next(counter), 0.0, start)]
        parent, best_g = {start: None}, {start: 0.0}
        expanded = 0
        while heap:
            f, _, g, state = heapq.heappop(heap)
            if g > best_g.get(state, float("inf")) + 1e-9:
                continue
            t, last, spent, events = state
            if t == horizon:
                if events >= min_events:
                    plan = []
                    s = state
                    while parent[s] is not None:
                        s, act = parent[s]
                        plan.append(act)
                    plan.reverse()
                    return plan, -g, expanded
                continue
            expanded += 1
            if expanded > node_limit:
                return None, 0.0, expanded
            if events + (horizon - t) < min_events:
                continue
            acts = [("none", None, 0.0, 0.0)]
            for o in options[t]:
                gap = t - last[o["ci"]]
                if gap == 1:
                    continue                                   # cooldown precondition
                profit, spend = self._option_value(o, self.fatigue(gap))
                if spent + spend > budget:
                    continue                                   # budget precondition
                acts.append(("promote", o, profit, spend))
            for kind, o, profit, spend in acts:
                if kind == "none":
                    nstate = (t + 1, last, spent, events)
                else:
                    nl = list(last)
                    nl[o["ci"]] = t
                    nstate = (t + 1, tuple(nl), spent + int(round(spend)), min(events + 1, min_events))
                ng = g - profit
                if ng < best_g.get(nstate, float("inf")) - 1e-9:
                    best_g[nstate] = ng
                    parent[nstate] = (state, (t, kind, o, profit, spend))
                    heapq.heappush(heap, (ng - suffix[t + 1], next(counter), ng, nstate))
        return None, 0.0, expanded

    def greedy_plan(self, options, horizon, budget, min_events):
        """Simple baseline for comparison: each week take the best allowed promotion right now."""
        last, spent, total, events, plan = [-99] * len(CATS), 0.0, 0.0, 0, []
        for t in range(horizon):
            cands = []
            for o in options[t]:
                gap = t - last[o["ci"]]
                if gap == 1:
                    continue
                profit, spend = self._option_value(o, self.fatigue(gap))
                if spent + spend <= budget and profit > 0:
                    cands.append((profit, spend, o))
            if cands:
                profit, spend, o = max(cands, key=lambda x: x[0])
                last[o["ci"]], spent, total, events = t, spent + spend, total + profit, events + 1
                plan.append((t, "promote", o, profit, spend))
        return (plan, total) if events >= min_events else (None, 0.0)


# =====================================================================================
# SECTION 8 - WORKFLOW LAYER: route each result to the right person (the planner decides)
# =====================================================================================
def route(ev: Evaluation):
    """Simple workflow rules that decide who should look at a promotion next.
    In a production system this hand-off would live in a workflow platform (for example Salesforce Agentforce)."""
    if ev.rules.label == "NON-COMPLIANT":
        return "Return to proposer", "fix the rule violations before anyone reviews it"
    if ev.rules.label == "NEEDS REVIEW":
        return "Trade marketing manager", "check the flagged guidelines, then pass to the planner"
    if ev.profit <= 0:
        return "Category planner", "allowed but negative ROI - fund only for a strategic reason"
    return "Category planner", "allowed and positive ROI - ready for a funding decision"


def print_routing(evals):
    print("\n  Workflow routing (who sees each promotion next):")
    for i, ev in enumerate(evals, 1):
        who, why = route(ev)
        print(f"   #{i}: {who:<25} {why}")


def save_review_packet(evals, path="review_packet.txt"):
    """Write the comparison and routing to a text file the planner can open or forward."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("TRADE PROMOTION REVIEW PACKET\n\n")
            for i, ev in enumerate(evals, 1):
                who, why = route(ev)
                fh.write(f"#{i} {ev.scenario.short()}\n   lift {ev.avg_lift_pct:.0f}%, profit {money(ev.profit)}, "
                         f"spend {money(ev.spend)}, {ev.roi_class}, rules: {ev.rules.label}\n   route to: {who} ({why})\n")
                for reason, detail in ev.rules.violations:
                    fh.write(f"   violation: {reason}\n")
                for reason, detail, cf in ev.rules.reviews:
                    fh.write(f"   guideline: {reason} (certainty {cf:.2f})\n")
                fh.write("\n")
        return path
    except OSError as err:
        print(f"  Could not save the review packet: {err}")
        return None


# =====================================================================================
# SECTION 9 - OUTPUT (decision-support tables)
# =====================================================================================
def money(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def fmt_detail(reason, v):
    if isinstance(v, float) and 0 < v < 1:
        return f"{v * 100:.0f}%"
    if "price floor" in reason:
        return f"${v:.2f}"
    if isinstance(v, float) and v >= 1000:
        return money(v)
    return str(v)


def print_evaluation(ev: Evaluation, explain=True):
    sc = ev.scenario
    print(f"\n  Promotion : {sc.short()}  ({sc.weeks} week{'s' if sc.weeks > 1 else ''})")
    print(f"  Neural network : average lift {ev.avg_lift_pct:5.1f}%   extra units {ev.incr_units:,.0f}")
    print(f"  Money          : incremental profit {money(ev.profit)}   trade spend {money(ev.spend)}   ROI {ev.roi * 100:+.0f}%")
    print(f"  ROI class      : {ev.roi_class}")
    print(f"  Rule check     : {ev.rules.label}")
    for reason, detail in ev.rules.violations:
        print(f"      VIOLATION  {reason} (limit {fmt_detail(reason, detail)})")
    for reason, detail, cf in ev.rules.reviews:
        print(f"      guideline  {reason} ({fmt_detail(reason, detail)}), certainty {cf:.2f}")
    if ev.rules.reviews:
        print(f"      combined certainty of guidelines: {ev.rules.review_cf:.2f}  (0.50 or more = needs review)")
    if explain and ev.rules.proof:
        print("  Why (inference chain):")
        for line in ev.rules.proof:
            print("      " + line)


def print_comparison(evals):
    print("\n  " + "-" * 118)
    print(f"  {'#':<3}{'Promotion':<62}{'Lift':>6}{'Profit':>12}{'Spend':>11}  {'ROI class':<13}{'Rules'}")
    print("  " + "-" * 118)
    for i, ev in enumerate(evals, 1):
        print(f"  {i:<3}{ev.scenario.short()[:60]:<62}{ev.avg_lift_pct:5.0f}%{money(ev.profit):>12}{money(ev.spend):>11}  "
              f"{ev.roi_class:<13}{ev.rules.label}")
    print("  " + "-" * 118)
    ok = [e for e in evals if e.rules.label != "NON-COMPLIANT"]
    if ok:
        best = max(ok, key=lambda e: e.profit)
        print(f"  Highest predicted profit among allowed options: #{evals.index(best) + 1}. "
              "The planner makes the final call.")


def print_config_table(system, ci, retailer, ranked, title):
    print(f"\n  {title}")
    print(f"  {'Rank':<5}{'Plan':<56}{'Profit':>11}{'ROI class':>14}{'Rules':>16}")
    for i, (fit, cfg, profit, spend, lift, res) in enumerate(ranked, 1):
        print(f"  {i:<5}{system.scenario_from_config(ci, retailer, cfg).short()[len(CAT_NAMES[ci]) + len(retailer) + 4:][:54]:<56}"
              f"{money(profit):>11}{('POSITIVE' if profit > 0 else 'NEGATIVE'):>14}{res.label:>16}")


def print_plan(plan, total, nodes, options, horizon, budget, start_week):
    print(f"\n  {'Week':<6}{'Action':<66}{'Profit':>11}{'Spend':>11}")
    spent = 0.0
    for t, kind, o, profit, spend in plan:
        wk = wrap_week(start_week + t)
        if kind == "none":
            print(f"  {wk:<6}{'DoNothing':<66}{'':>11}{'':>11}")
        else:
            d, disp, feat = o["cfg"][0], DISPLAYS[o["cfg"][1]], "yes" if o["cfg"][2] else "no"
            txt = f"Promote({CAT_NAMES[o['ci']]}) {d}% off, {disp}, feature={feat}"
            spent += spend
            print(f"  {wk:<6}{txt:<66}{money(profit):>11}{money(spend):>11}")
    print(f"  Total expected profit {money(total)}   trade spend {money(spent)} of {money(budget)} budget")
    print(f"  A* examined {nodes:,} states (a brute-force search of every calendar would face about "
          f"{(len(CATS) + 1) ** horizon:,}).")


# =====================================================================================
# SECTION 10 - USER INTERFACE (menu) AND DEMO
# =====================================================================================
def ask(prompt, default=None, cast=str, valid=None):
    while True:
        raw = input(f"  {prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if raw == "" and default is not None:
            return default
        try:
            value = cast(raw)
        except ValueError:
            print("    Please enter a valid value.")
            continue
        if valid is not None and not valid(value):
            print("    That value is not allowed here.")
            continue
        return value


def ask_choice(prompt, options, default_index=0):
    print(f"  {prompt}")
    for i, o in enumerate(options, 1):
        print(f"     {i:>2}) {o}")
    return ask("Choose a number", default_index + 1, int, lambda v: 1 <= v <= len(options)) - 1


def ask_scenario(system, defaults=None):
    d = defaults or PRESETS[0]
    ci = ask_choice("Category:", CAT_NAMES, CAT_INDEX[d.category])
    retailer = list(RETAILER_CAPS)[ask_choice("Retailer:", list(RETAILER_CAPS), list(RETAILER_CAPS).index(d.retailer))]
    week = ask("Start week (1-52)", d.start_week, int, lambda v: 1 <= v <= 52)
    weeks = ask("Number of weeks (1-4)", d.weeks, int, lambda v: 1 <= v <= 4)
    disc = ask("Discount percent (5-40)", round(d.discount * 100), int, lambda v: 5 <= v <= 40)
    disp = DISPLAYS[ask_choice("Display type:", DISPLAYS, DISPLAYS.index(d.display))]
    feat = ask("Feature ad? 1 = yes, 0 = no", d.feature, int, lambda v: v in (0, 1))
    return Scenario(CAT_NAMES[ci], retailer, week, weeks, disc / 100.0, disp, feat)


PRESETS = [
    Scenario("Disinfecting Wipes", "MassRetailer", 8, 2, 0.20, "endcap", 1),
    Scenario("Trash Bags", "MassRetailer", 27, 1, 0.38, "endcap", 0),
    Scenario("Charcoal & Grilling", "GroceryChain", 27, 2, 0.22, "aisle", 0),
    Scenario("Cat Litter", "WarehouseClub", 30, 1, 0.25, "none", 0),
]


def menu_evaluate(system):
    print("\n  === Evaluate one promotion ===")
    print_evaluation(system.evaluate(ask_scenario(system)))


def menu_compare(system):
    print("\n  === Compare promotions side by side ===")
    use_presets = ask("Use the 4 built-in example promotions? (y/n)", "y", str, lambda v: v.lower() in ("y", "n")).lower() == "y"
    scenarios = list(PRESETS)
    if not use_presets:
        n = ask("How many promotions to compare (2-4)", 2, int, lambda v: 2 <= v <= 4)
        scenarios = [ask_scenario(system, PRESETS[i % len(PRESETS)]) for i in range(n)]
    evals = [system.evaluate(s, pid=f"S{i}") for i, s in enumerate(scenarios, 1)]
    print_comparison(evals)
    print_routing(evals)
    if ask("Save a review packet file for the planner? (y/n)", "n", str, lambda v: v.lower() in ("y", "n")).lower() == "y":
        path = save_review_packet(evals)
        if path:
            print(f"  Saved {path}")


def run_ga(system, ci, retailer, start, window=8, verify=True):
    starts, durations = list(range(start, start + window)), [1, 2, 3, 4]
    space = len(system.all_configs(starts, durations))
    t0 = time.time()
    ranked, history, evals = system.genetic_search(ci, retailer, starts, durations)
    print(f"\n  Genetic algorithm: searched {space:,} possible plans for {CAT_NAMES[ci]} at {retailer} "
          f"(weeks {start}-{start + window - 1}) using {evals} evaluations in {time.time() - t0:.1f}s.")
    print(f"  Best fitness by generation: {', '.join(money(h) for h in history[::6] + [history[-1]])}")
    print_config_table(system, ci, retailer, ranked, "Top plans found by the genetic algorithm (fitness = profit minus rule penalties):")
    if verify:
        exact, _n, checks = system.exhaustive_search(ci, retailer, starts, durations, top=1)
        gap = ranked[0][0] / exact[0][0] * 100 if exact[0][0] > 0 else 100.0
        print(f"  Check against exhaustive scoring: best possible fitness {money(exact[0][0])}; "
              f"the genetic algorithm reached {gap:.1f}% of it.")
    return ranked


def menu_ga(system):
    print("\n  === Find the best promotion for a category (genetic algorithm) ===")
    ci = ask_choice("Category:", CAT_NAMES, 0)
    retailer = list(RETAILER_CAPS)[ask_choice("Retailer:", list(RETAILER_CAPS), 0)]
    start = ask("First week to consider (1-45)", 6, int, lambda v: 1 <= v <= 45)
    ranked = run_ga(system, ci, retailer, start)
    best = system.scenario_from_config(ci, retailer, ranked[0][1])
    if ask("Show full evaluation and rule explanation of the best plan? (y/n)", "y", str, lambda v: v.lower() in ("y", "n")).lower() == "y":
        print_evaluation(system.evaluate(best))


def run_planner(system, retailer, start_week, horizon, budget, min_events):
    t0 = time.time()
    options = system.build_plan_options(retailer, start_week, horizon)
    plan, total, nodes = system.plan_calendar(options, horizon, budget, min_events)
    if plan is None:
        print("\n  No calendar satisfies all constraints (try a larger budget or fewer required events).")
        return None
    print(f"\n  A* planner for {retailer}, weeks {start_week}-{wrap_week(start_week + horizon - 1)}: "
          f"budget {money(budget)}, at least {min_events} promotions, no repeat in consecutive weeks. ({time.time() - t0:.1f}s)")
    print_plan(plan, total, nodes, options, horizon, budget, start_week)
    gplan, gtotal = system.greedy_plan(options, horizon, budget, min_events)
    if gplan is None:
        print("  Simple greedy calendar: could not meet the constraints.")
    else:
        print(f"  Simple greedy calendar (best promotion each week, no look-ahead): {money(gtotal)}  "
              f"-> A* is {money(total - gtotal)} better.")
    return plan, total


def menu_plan(system):
    print("\n  === Plan a multi-week promotion calendar (A* search) ===")
    retailer = list(RETAILER_CAPS)[ask_choice("Retailer:", list(RETAILER_CAPS), 0)]
    start = ask("Start week (1-52)", 24, int, lambda v: 1 <= v <= 52)
    horizon = ask("Weeks to plan (3-8)", 6, int, lambda v: 3 <= v <= 8)
    budget = ask("Total trade-spend budget in dollars", 600000, int, lambda v: v > 0)
    min_events = ask("Minimum number of promotions required", 2, int, lambda v: 0 <= v <= horizon)
    run_planner(system, retailer, start, horizon, float(budget), min_events)


NEW_PRODUCT_PRESETS = [
    ("Antibacterial Hand Soap (similar to cleaning products)", (0.4, 0.7, 0.3, 0.6, 0.5), 4.20, 0.45, 20000),
    ("Premium Reusable Water Bottle (similar to water filtration)", (0.9, 0.2, 0.4, 0.3, 0.7), 14.00, 0.50, 9000),
    ("Custom product (enter your own traits)", None, None, None, None),
]


def run_cold_start(system, name, desc, price, margin, units, k=5):
    nb, rows = system.estimate_new_product(desc, price, margin, units, k=k)
    print(f"\n  New product: {name}")
    print(f"  Nearest existing categories (k={k}), borrowed as the starting estimate:")
    for ci, dist in nb:
        print(f"      {CAT_NAMES[ci]:<26} distance {dist:.2f}")
    print(f"  {'Discount':<10}{'Display':<9}{'Feature':<9}{'Est. lift':>10}{'Est. profit/wk':>16}")
    for d, p, f, lift, profit in sorted(rows, key=lambda r: -r[4])[:5]:
        print(f"  {round(d * 100):>3}%      {DISPLAYS[p]:<9}{'yes' if f else 'no':<9}{lift:>9.0f}%{money(profit):>16}")
    print("  (Top 5 of 18 test promotions. These are starting estimates only; real results should replace them.)")


def menu_new_product(system):
    print("\n  === Estimate a new product with no promotion history (nearest neighbors) ===")
    i = ask_choice("Which product?", [p[0] for p in NEW_PRODUCT_PRESETS], 0)
    name, desc, price, margin, units = NEW_PRODUCT_PRESETS[i]
    if desc is None:
        name = ask("Product name", "My New Product")
        print("  Rate each trait from 0.0 (low) to 1.0 (high):")
        desc = tuple(ask(t, 0.5, float, lambda v: 0 <= v <= 1) for t in
                     ("Price tier", "Purchase frequency", "Seasonality", "Stockpile-ability", "Brand loyalty"))
        price = ask("List price in dollars", 6.00, float, lambda v: v > 0)
        margin = ask("Gross margin (0.05-0.80)", 0.45, float, lambda v: 0.05 <= v <= 0.8)
        units = ask("Expected baseline units per week", 20000, int, lambda v: v > 0)
    run_cold_start(system, name, desc, price, margin, units)


def show_metrics(system):
    m = system.metrics
    print("\n  === How good is the model? (held-out test promotions, never used in training) ===")
    print(f"  Test promotions: {m['n_test']}   share that truly earn a profit: {m['pct_positive'] * 100:.0f}%")
    print(f"  Best network settings found by random search: embedding={system.best_params['cat_dim']}, "
          f"hidden={system.best_params['hidden']}, lr={system.best_params['lr']:.4f}")
    print(f"\n  {'Lift prediction error (percentage points)':<44}{'MAE':>8}{'RMSE':>9}")
    for k, (mae, rmse) in m["lift"].items():
        print(f"  {k:<44}{mae:>8.1f}{rmse:>9.1f}")
    print(f"\n  {'ROI classification (positive vs negative)':<44}{'Accuracy':>9}{'Catch neg.':>12}{'Neg. precision':>16}")
    for k, s in m["roi"].items():
        print(f"  {k:<44}{s['acc'] * 100:>8.0f}%{s['rec_neg'] * 100:>11.0f}%{s['prec_neg'] * 100:>15.0f}%")
    print("  ('Catch neg.' = share of truly unprofitable promotions the model flagged as negative ROI.)")
    print(f"\n  {'New-product (cold start) test, mean error in lift points':<58}{'Error':>8}")
    for k, v in m["cold_start"].items():
        print(f"  {k:<58}{v:>8.1f}")
    print("\n  Categories the network treats alike (closest learned embeddings; reflects promotion response, not product type):")
    for ci in (0, 5, 9):
        close = ", ".join(f"{CAT_NAMES[i]} ({s:.2f})" for i, s in system.embedding_neighbors(ci))
        print(f"      {CAT_NAMES[ci]:<22} ~ {close}")


def show_rules(system):
    print("\n  === Knowledge base: rules the expert system uses (variables start with ?) ===")
    for r in system.rules:
        print("  " + r.text())
    print("\n  Facts include InGroup(category, group), GroupLimit(group, x), MaxDiscount(retailer, x),")
    print("  MinShelfPrice(category, price), MinPromoMargin(x), and the committed calendar:")
    for pid, cat, retailer, weeks in COMMITTED_CALENDAR:
        print(f"      Scheduled({pid}): {cat} at {retailer}, weeks {weeks}")
    print("\n  Planner operators (STRIPS style):")
    print("      Promote(category, week)  PRE: not promoted last week, budget remains, rule-compliant")
    print("                               EFFECT: category marked promoted, budget reduced, event count + 1")
    print("      DoNothing(week)          PRE: none          EFFECT: week passes")


def interactive(system):
    actions = {"1": menu_evaluate, "2": menu_compare, "3": menu_ga, "4": menu_plan,
               "5": menu_new_product, "6": show_metrics, "7": show_rules}
    while True:
        print("\n" + "=" * 78)
        print(" TRADE PROMOTION DECISION SUPPORT  (the planner always makes the final decision)")
        print("=" * 78)
        print("  1) Evaluate one promotion        (lift, profit, ROI class, rule check, explanation)")
        print("  2) Compare promotions side by side")
        print("  3) Find the best promotion for a category      (genetic algorithm)")
        print("  4) Plan a multi-week calendar                   (A* symbolic planning)")
        print("  5) Estimate a new product with no history       (nearest neighbors)")
        print("  6) Show model quality")
        print("  7) Show the business rules (knowledge base)")
        print("  0) Exit")
        choice = input("  Your choice: ").strip()
        if choice == "0":
            print("  Goodbye.")
            return
        if choice in actions:
            actions[choice](system)
        else:
            print("  Please choose 0-7.")


def demo(system):
    bar = "=" * 78
    print(f"\n{bar}\n DEMO 1 of 6 - Model quality\n{bar}")
    show_metrics(system)
    print(f"\n{bar}\n DEMO 2 of 6 - Evaluate one promotion (neural network + expert system)\n{bar}")
    print_evaluation(system.evaluate(PRESETS[0]), explain=False)
    print(f"\n{bar}\n DEMO 3 of 6 - Compare four promotions; one breaks the rules\n{bar}")
    evals = [system.evaluate(s, pid=f"S{i}") for i, s in enumerate(PRESETS, 1)]
    print_comparison(evals)
    print_routing(evals)
    print("\n  Explanation for promotion #2 (why the expert system flagged it):")
    print_evaluation(evals[1])
    print(f"\n{bar}\n DEMO 4 of 6 - Best promotion for a category (genetic algorithm)\n{bar}")
    run_ga(system, CAT_INDEX["Disinfecting Wipes"], "MassRetailer", 4)
    print(f"\n{bar}\n DEMO 5 of 6 - Six-week calendar (A* symbolic planning)\n{bar}")
    run_planner(system, "MassRetailer", 24, 6, 600000.0, 2)
    print(f"\n{bar}\n DEMO 6 of 6 - New product with no history (nearest neighbors)\n{bar}")
    name, desc, price, margin, units = NEW_PRODUCT_PRESETS[0]
    run_cold_start(system, name, desc, price, margin, units)
    print("\n  Demo finished. Run without --demo to use the interactive menu.")


# =====================================================================================
# SECTION 11 - SELF-TEST (python tpm_decision_support.py --selftest)
# =====================================================================================
def selftest(system):
    print("\nRunning self-tests ...")
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # 1. Expert system behaves as the rules say
    good = system.evaluate(PRESETS[0], pid="T1")
    over_cap = system.evaluate(Scenario("Trash Bags", "MassRetailer", 27, 1, 0.38, "endcap", 0), pid="T2")
    near = system.evaluate(PRESETS[3], pid="T3")
    check("20% promotion with no rule problems is COMPLIANT", good.rules.label == "COMPLIANT")
    over_overlap = system.evaluate(PRESETS[2], pid="T4")
    check("promotion overlapping a scheduled one in the same shelf group NEEDS REVIEW (rule R7)", over_overlap.rules.label == "NEEDS REVIEW")
    check("38% discount above retailer cap is NON-COMPLIANT", over_cap.rules.label == "NON-COMPLIANT")
    check("cap violation has an explanation chain", len(over_cap.rules.proof) > 0)
    check("discount exactly at the retailer cap NEEDS REVIEW (rule R8)", near.rules.label == "NEEDS REVIEW")
    check("inheritance rule R2 derives a group limit for a new promotion",
          any("Limit" in line for line in over_cap.rules.proof))
    check("certainty factors combine (0.35 + 0.35 -> 0.5775)", abs(combine_cf([0.35, 0.35]) - 0.5775) < 1e-9)
    low_margin = system.evaluate(Scenario("Bleach", "GroceryChain", 40, 1, 0.32, "none", 0), pid="T5")
    check("32% off a 40%-margin product breaks only the minimum-margin rule (R6)",
          [v[0] for v in low_margin.rules.violations] == ["minimum margin after discount"])
    check("workflow routes a NON-COMPLIANT promotion back to the proposer", route(over_cap)[0] == "Return to proposer")
    check("workflow routes a compliant, profitable promotion to the category planner",
          good.profit > 0 and route(good)[0] == "Category planner")

    # 2. Neural network beats the spreadsheet-style baseline
    m = system.metrics["lift"]
    check("neural network beats the category-average baseline on lift error",
          m["Neural network"][0] < m["Category average (rule of thumb)"][0])
    check("neural network is at least as good as linear regression",
          m["Neural network"][0] <= m["Linear regression"][0] * 1.10)

    # 3. A* returns the true optimum (compared with brute force on a small problem)
    horizon, min_events = 4, 2
    options = system.build_plan_options("MassRetailer", 24, horizon, per_week=3)
    spends = [o["spend"] for t in range(horizon) for o in options[t]]
    budget = round(2.4 * float(np.median(spends)), -3)       # room for about two promotions, so the budget really binds
    plan, total, _nodes = system.plan_calendar(options, horizon, budget, min_events)
    best = -1e18
    choices = [[None] + options[t] for t in range(horizon)]
    for combo in itertools.product(*choices):
        last, spent, tot, ev, valid = [-99] * len(CATS), 0.0, 0.0, 0, True
        for t, o in enumerate(combo):
            if o is None:
                continue
            gap = t - last[o["ci"]]
            if gap == 1:
                valid = False
                break
            p, s = system._option_value(o, system.fatigue(gap))
            spent, tot, ev, last[o["ci"]] = spent + s, tot + p, ev + 1, t
            if spent > budget:
                valid = False
                break
        if valid and ev >= min_events:
            best = max(best, tot)
    check("A* plan profit equals brute-force optimum", plan is not None and abs(total - best) < 1.0)
    none_plan, _t, _n = system.plan_calendar(options, horizon, 50000.0, min_events)
    check("A* reports 'no plan' when the budget makes the goal impossible", none_plan is None)

    # 4. Genetic algorithm gets close to the exhaustive optimum
    starts, durations = list(range(4, 12)), [1, 2, 3, 4]
    ranked, _h, _e = system.genetic_search(CAT_INDEX["Disinfecting Wipes"], "MassRetailer", starts, durations)
    exact, _n, _c = system.exhaustive_search(CAT_INDEX["Disinfecting Wipes"], "MassRetailer", starts, durations, top=1)
    check("genetic algorithm reaches at least 90% of the best possible fitness", ranked[0][0] >= 0.90 * exact[0][0])
    check("genetic algorithm's best plan is not NON-COMPLIANT", ranked[0][5].label != "NON-COMPLIANT")

    # 5. Nearest neighbors
    nb, _rows = system.estimate_new_product((0.4, 0.7, 0.3, 0.6, 0.5), 4.2, 0.45, 20000, k=5)
    check("nearest-neighbor search returns 5 neighbors", len(nb) == 5)
    cs = system.metrics["cold_start"]
    check("averaging neighbors beats using all other categories", cs["5 neighbor(s)"] < cs["Average of all other categories"])

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return ok


# =====================================================================================
# MAIN
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description="TPM decision-support program (CSC510 portfolio project)")
    parser.add_argument("--demo", action="store_true", help="run a complete non-interactive walkthrough")
    parser.add_argument("--selftest", action="store_true", help="run automatic checks")
    parser.add_argument("--quick", action="store_true", help="smaller data and shorter training")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    args = parser.parse_args()

    print("=" * 78)
    print(" Neural Network-Based Decision Support for Trade Promotion Management")
    print(" CSC510 Portfolio Project - Anand Verma")
    print("=" * 78)
    t0 = time.time()
    system = TPMSystem(seed=args.seed, quick=args.quick)
    print(f"Ready in {time.time() - t0:.1f} seconds.")
    try:
        if args.selftest:
            sys.exit(0 if selftest(system) else 1)
        elif args.demo:
            demo(system)
        else:
            interactive(system)
    except (EOFError, KeyboardInterrupt):
        print("\n  Input closed - exiting.")


if __name__ == "__main__":
    main()
