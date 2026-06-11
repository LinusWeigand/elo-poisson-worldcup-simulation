#!/usr/bin/env python3
"""
FIFA World Cup 2026 Monte Carlo Simulation
==========================================
- Elo ratings from eloratings.net (data/teams.csv, snapshot June 11, 2026)
- Official 2026 format: 12 groups of 4, top 2 + 8 best third-placed teams
  advance to a Round of 32; FIFA Annex C governs third-place allocation.
- Hosts (USA, Mexico, Canada) receive a home-advantage Elo bonus.
- Goals are sampled from a pair of Poisson distributions whose rates are
  fitted so that the implied match expectancy matches the Elo expectancy
  as closely as possible (draws fall out of the model naturally).
- Elo ratings evolve *inside* each simulation after every match
  (eloratings.net update rule: K=60, goal-difference multiplier).
- Real results can be entered in data/results.json and are then treated
  as fixed in every simulation (including their Elo effect).

Usage:
    python3 simulate.py [n_sims]          (default 10000)

Outputs (output/):
    team_stats.csv          per-team tournament probabilities
    group_match_probs.csv   per-fixture win/draw/loss probabilities
    knockout_stats.csv      knockout draw/penalty statistics per team
    console summary
"""

import argparse
import csv
import io
import json
import math
import os
import random
import re
import sys
import unicodedata
import urllib.request
from collections import defaultdict

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")

N_SIMS_DEFAULT = 10000
RNG_SEED = None          # set an int for reproducible runs

# Expected total goals in a match between equal teams (FIFA WC average ~2.6-2.8)
AVG_TOTAL_GOALS = 2.70
MAX_GOALS = 12           # truncation of the Poisson when sampling/fitting

# eloratings.net update parameters
K_FACTOR = 60            # World Cup finals weight

# Host home-advantage bonus, in Elo points, by tournament stage.
# eloratings.net itself uses +100 for home advantage; hosts historically
# overperform (your slide: 66.0% host wins vs 39.9% non-host, +26.1 pp).
# From the QF onward every match is played in the USA.
HOST_BONUS = {
    "USA": {"group": 100, "r32": 100, "r16": 100, "qf": 100, "sf": 100, "final": 100},
    "MEX": {"group": 100, "r32": 100, "r16": 100, "qf": 0,   "sf": 0,   "final": 0},
    "CAN": {"group": 100, "r32": 100, "r16": 100, "qf": 0,   "sf": 0,   "final": 0},
}
# (Mexico City hosts knockout games through the R16; Vancouver through the R16;
#  set entries to taste if you want a different assumption.)

# Penalty shootouts: p(team A wins) = 0.5 + PENALTY_ELO_WEIGHT * (E_elo - 0.5)
# 0.0 = pure coin flip; 1.0 = full Elo expectancy. Shootouts are mostly luck.
PENALTY_ELO_WEIGHT = 0.25

# ----------------------------------------------------------------------------
# Static tournament structure
# ----------------------------------------------------------------------------
GROUPS = "ABCDEFGHIJKL"

# Group fixtures by draw position (1-4). Every pair plays exactly once, so
# you can always enter a real result for any pairing; ordering only affects
# the sequence of in-simulation Elo updates.
GROUP_FIXTURE_PATTERN = [(1, 2), (3, 4), (1, 3), (4, 2), (4, 1), (2, 3)]

# Round of 32 (FIFA matches 73-88). "W"=group winner, "R"=runner-up,
# "T"=best third-placed team (allowed source groups listed).
R32 = {
    73: ("R", "A", "R", "B"),
    74: ("W", "E", "T", "ABCDF"),
    75: ("W", "F", "R", "C"),
    76: ("W", "C", "R", "F"),
    77: ("W", "I", "T", "CDFGH"),
    78: ("R", "E", "R", "I"),
    79: ("W", "A", "T", "CEFHI"),
    80: ("W", "L", "T", "EHIJK"),
    81: ("W", "D", "T", "BEFIJ"),
    82: ("W", "G", "T", "AEHIJ"),
    83: ("R", "K", "R", "L"),
    84: ("W", "H", "R", "J"),
    85: ("W", "B", "T", "EFGIJ"),
    86: ("W", "J", "R", "H"),
    87: ("W", "K", "T", "DEIJL"),
    88: ("R", "D", "R", "G"),
}
# Which R32 match each group winner with a third-place opponent plays,
# in Annex C column order (1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L):
ANNEX_C_COLUMNS = [("A", 79), ("B", 85), ("D", 81), ("E", 74),
                   ("G", 82), ("I", 77), ("K", 87), ("L", 80)]

R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = 104  # winners of 101 and 102 (match 103 = third-place play-off)

KO_STAGE_OF_MATCH = {}
for m in R32: KO_STAGE_OF_MATCH[m] = "r32"
for m in R16: KO_STAGE_OF_MATCH[m] = "r16"
for m in QF:  KO_STAGE_OF_MATCH[m] = "qf"
for m in SF:  KO_STAGE_OF_MATCH[m] = "sf"
KO_STAGE_OF_MATCH[103] = "sf"      # 3rd-place play-off (same venue tier)
KO_STAGE_OF_MATCH[104] = "final"

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
HOST_OF = {"Mexico": "MEX", "Canada": "CAN", "United States": "USA"}


def load_teams():
    """
    Robust loader for data/teams.csv.
    Tolerates: UTF-8 BOM, semicolon delimiters (Excel in many locales),
    extra whitespace, missing 'pos' column (assigned by row order within
    each group), missing 'host' column (inferred), and comma decimals.
    """
    path = os.path.join(DATA_DIR, "teams.csv")
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    header = text.splitlines()[0] if text else ""
    delim = ";" if header.count(";") > header.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    if not reader.fieldnames:
        raise ValueError(f"{path} appears to be empty")
    colmap = {fn.strip().lower(): fn for fn in reader.fieldnames}
    for required in ("team", "group", "elo"):
        if required not in colmap:
            raise ValueError(
                f"{path}: column '{required}' not found "
                f"(got {reader.fieldnames}). If you edited the file in a "
                f"spreadsheet app, re-save it as plain CSV.")

    teams = {}
    by_group = defaultdict(list)
    pos_counter = defaultdict(int)
    for row in reader:
        name = row[colmap["team"]].strip()
        if not name:
            continue
        g = row[colmap["group"]].strip().upper()
        if "pos" in colmap and (row.get(colmap["pos"]) or "").strip():
            pos = int(row[colmap["pos"]].strip())
        else:
            pos_counter[g] += 1
            pos = pos_counter[g]
        elo = float(row[colmap["elo"]].strip().replace(",", "."))
        if "host" in colmap:
            host = (row.get(colmap["host"]) or "").strip() or None
        else:
            host = HOST_OF.get(name)
        teams[name] = {"group": g, "pos": pos, "elo": elo, "host": host}
        by_group[g].append(name)
    if len(teams) != 48:
        raise ValueError(f"{path}: expected 48 teams, found {len(teams)}")
    for g in by_group:
        by_group[g].sort(key=lambda t: teams[t]["pos"])
    return teams, by_group


# ----------------------------------------------------------------------------
# Optional: refetch current Elo ratings at run time (--refetch-elo)
# ----------------------------------------------------------------------------
# Name spellings differ between sources; normalize + alias to our names.
_ALIASES = {
    "turkiye": "Turkey", "czechia": "Czech Republic",
    "korearepublic": "South Korea", "usa": "United States",
    "cotedivoire": "Ivory Coast", "caboverde": "Cape Verde",
    "demrepofcongo": "DR Congo", "congodr": "DR Congo",
    "drcongo": "DR Congo", "congokinshasa": "DR Congo",
    "bosniaherzegovina": "Bosnia and Herzegovina",
    "bosnia": "Bosnia and Herzegovina", "ksa": "Saudi Arabia",
    "iriran": "Iran", "uae": "United Arab Emirates",
}


def _norm(name):
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", s.lower())


def _http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", "replace")


def _fetch_from_eloratings():
    """eloratings.net serves its data as TSV files behind the JS frontend."""
    code_to_name = {}
    for line in _http_get("https://eloratings.net/en.teams.tsv").splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) >= 2 and re.fullmatch(r"[A-Z]{2,3}", parts[0]):
            for p in parts[1:]:
                if p and re.search(r"[A-Za-z]", p) and not p.isdigit():
                    code_to_name[parts[0]] = p
                    break
    ratings = {}
    for line in _http_get("https://eloratings.net/World.tsv").splitlines():
        parts = [p.strip() for p in line.split("\t")]
        code = next((p for p in parts if re.fullmatch(r"[A-Z]{2,3}", p)), None)
        if not code or code not in code_to_name:
            continue
        after = parts[parts.index(code) + 1:]
        rating = next((int(p) for p in after
                       if re.fullmatch(r"\d{3,4}", p) and 500 <= int(p) <= 2400),
                      None)
        if rating is not None:
            ratings[code_to_name[code]] = float(rating)
    return ratings


def _fetch_from_mirror():
    """Fallback: international-football.net mirrors eloratings.net daily."""
    html = _http_get("https://www.international-football.net/elo-ratings-table")
    ratings = {}
    # rows look like: ... title="Spain"> ... Spain ... >2155<
    for m in re.finditer(
            r'title="([^"]+)"[^>]*>(?:(?!title=).){0,400}?>(\d{3,4})<',
            html, re.S):
        name, val = m.group(1).strip(), int(m.group(2))
        if 500 <= val <= 2400 and name not in ratings:
            ratings[name] = float(val)
    return ratings


def refetch_elo(teams):
    """
    Try to refresh team Elo ratings from the web. On success the new values
    are used for this run AND written back to data/teams.csv. On any failure
    the existing CSV values are kept and a warning is printed.
    """
    fetched = {}
    for fetcher in (_fetch_from_eloratings, _fetch_from_mirror):
        try:
            fetched = fetcher()
            if len(fetched) >= 100:   # sanity: a full world table
                break
            fetched = {}
        except Exception as e:
            print(f"  [refetch] {fetcher.__name__} failed: {e}")
    if not fetched:
        print("  [refetch] could not fetch ratings - keeping teams.csv values")
        return False

    lookup = {_norm(k): v for k, v in fetched.items()}
    updated, missing = 0, []
    for name, info in teams.items():
        key = _norm(name)
        key = _norm(_ALIASES.get(key, name)) if key in _ALIASES else key
        new = lookup.get(key)
        if new is None:
            # also try alias table keyed by fetched names
            for fk, fv in lookup.items():
                if _ALIASES.get(fk) == name:
                    new = fv
                    break
        if new is None:
            missing.append(name)
            continue
        if abs(new - info["elo"]) >= 0.5:
            print(f"  [refetch] {name}: {info['elo']:.0f} -> {new:.0f}")
        info["elo"] = new
        updated += 1
    if missing:
        print(f"  [refetch] no rating found for: {', '.join(missing)} "
              f"(kept CSV values)")
    if updated:
        path = os.path.join(DATA_DIR, "teams.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["team", "group", "pos", "elo", "host"])
            for g in GROUPS:
                for name in sorted((t for t in teams if teams[t]["group"] == g),
                                   key=lambda t: teams[t]["pos"]):
                    i = teams[name]
                    w.writerow([name, g, i["pos"], int(round(i["elo"])),
                                i["host"] or ""])
        print(f"  [refetch] updated {updated}/48 ratings and saved teams.csv")
    return updated > 0


def load_annex_c():
    """FIFA Annex C: qualified-thirds combination -> third-place assignment."""
    table = {}
    path = os.path.join(DATA_DIR, "annex_c.txt")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            combo, assign = line.split()
            key = frozenset(combo)
            if key in table:
                raise ValueError(f"duplicate Annex C combination {combo}")
            # validation: assignment uses exactly the qualified groups,
            # and each assignment obeys the match's allowed source groups
            if sorted(assign) != sorted(combo):
                raise ValueError(f"Annex C row {combo}: assignment {assign} "
                                 f"is not a permutation of the combination")
            for (winner_group, match_no), third in zip(ANNEX_C_COLUMNS, assign):
                allowed = R32[match_no][3]
                if third not in allowed:
                    raise ValueError(
                        f"Annex C row {combo}: 1{winner_group} vs 3{third} "
                        f"violates match {match_no} constraint ({allowed})")
            table[key] = assign
    if len(table) != 495:
        raise ValueError(f"Annex C table has {len(table)} rows, expected 495")
    return table


def load_results(by_group):
    """Load (or create) the editable results file."""
    path = os.path.join(DATA_DIR, "results.json")
    if not os.path.exists(path):
        doc = {"_help": [
            "Enter real results here as the tournament progresses, then rerun.",
            "GROUP STAGE: set 'score' to [home_goals, away_goals] (e.g. [2,1]).",
            "Leave score null for matches not yet played.",
            "KNOCKOUT: add entries to 'knockout' with the two team names, the",
            "score AFTER extra time, and 'penalties_winner' if it went to pens.",
            "Example: {\"teams\": [\"Mexico\", \"Croatia\"], \"score\": [1,1],",
            "          \"penalties_winner\": \"Mexico\"}",
        ], "group_stage": [], "knockout": []}
        for g in GROUPS:
            names = by_group[g]
            for i, j in GROUP_FIXTURE_PATTERN:
                doc["group_stage"].append({
                    "group": g, "home": names[i - 1], "away": names[j - 1],
                    "score": None})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Created editable results file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------------
# Elo + Poisson machinery
# ----------------------------------------------------------------------------
def elo_expectancy(d):
    """Win expectancy for team A given effective Elo difference d = Ra - Rb."""
    return 1.0 / (1.0 + 10.0 ** (-d / 400.0))


def _poisson_pmf(lam, kmax=MAX_GOALS):
    pmf = []
    p = math.exp(-lam)
    for k in range(kmax + 1):
        if k > 0:
            p *= lam / k
        pmf.append(p)
    s = sum(pmf)
    return [x / s for x in pmf]


def _match_probs(lam_a, lam_b):
    """(P win A, P draw, P win B) under independent Poisson scores."""
    pa = _poisson_pmf(lam_a)
    pb = _poisson_pmf(lam_b)
    win = draw = 0.0
    for i, x in enumerate(pa):
        for j, y in enumerate(pb):
            if i > j:
                win += x * y
            elif i == j:
                draw += x * y
    return win, draw, 1.0 - win - draw


_fit_cache = {}

def fit_poisson(d):
    """
    Fit (lam_a, lam_b) with lam_a + lam_b = AVG_TOTAL_GOALS such that the
    Poisson match expectancy  P(win A) + 0.5 * P(draw)  matches the Elo
    expectancy E(d) as closely as possible.

    At large |d| the achievable expectancy saturates below E(d) - the Poisson
    model cannot be made arbitrarily one-sided at a fixed goal total - which
    is exactly the Elo-vs-Poisson disparity you anticipate at big rating gaps.
    Returns (lam_a, lam_b, p_win_a, p_draw, p_win_b, cdf_a, cdf_b).
    """
    key = round(d)  # 1-Elo-point resolution is plenty
    hit = _fit_cache.get(key)
    if hit is not None:
        return hit

    target = elo_expectancy(key)
    total = AVG_TOTAL_GOALS
    lo, hi = 0.05, total - 0.05  # bisect over lam_a

    def expectancy(lam_a):
        w, dr, _ = _match_probs(lam_a, total - lam_a)
        return w + 0.5 * dr

    # expectancy is monotonically increasing in lam_a
    if expectancy(hi) < target:
        lam_a = hi
    elif expectancy(lo) > target:
        lam_a = lo
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if expectancy(mid) < target:
                lo = mid
            else:
                hi = mid
        lam_a = 0.5 * (lo + hi)

    lam_b = total - lam_a
    w, dr, l = _match_probs(lam_a, lam_b)
    pmf_a, pmf_b = _poisson_pmf(lam_a), _poisson_pmf(lam_b)
    cdf_a, cdf_b = [], []
    s = 0.0
    for x in pmf_a:
        s += x
        cdf_a.append(s)
    s = 0.0
    for x in pmf_b:
        s += x
        cdf_b.append(s)
    res = (lam_a, lam_b, w, dr, l, cdf_a, cdf_b)
    _fit_cache[key] = res
    return res


def _sample_cdf(cdf, rng):
    u = rng.random()
    for k, c in enumerate(cdf):
        if u <= c:
            return k
    return len(cdf) - 1


def goal_diff_index(gd):
    """eloratings.net margin-of-victory multiplier."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def host_bonus(team_info, stage):
    h = team_info["host"]
    if not h:
        return 0.0
    return HOST_BONUS.get(h, {}).get(stage, 0.0)


def effective_diff(elos, teams, a, b, stage):
    da = elos[a] + host_bonus(teams[a], stage)
    db = elos[b] + host_bonus(teams[b], stage)
    return da - db


def update_elo(elos, teams, a, b, ga, gb, stage):
    """eloratings.net update; host bonus enters the expectancy only."""
    d = effective_diff(elos, teams, a, b, stage)
    we = elo_expectancy(d)
    w = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
    delta = K_FACTOR * goal_diff_index(ga - gb) * (w - we)
    elos[a] += delta
    elos[b] -= delta


def play_match(elos, teams, a, b, stage, rng, fixed=None):
    """
    Simulate (or apply a fixed result for) a 90-minute match.
    Returns (ga, gb). Elo is updated in place.
    """
    if fixed is not None:
        ga, gb = fixed
    else:
        d = effective_diff(elos, teams, a, b, stage)
        _, _, _, _, _, cdf_a, cdf_b = fit_poisson(d)
        ga = _sample_cdf(cdf_a, rng)
        gb = _sample_cdf(cdf_b, rng)
    update_elo(elos, teams, a, b, ga, gb, stage)
    return ga, gb

# ----------------------------------------------------------------------------
# Group stage
# ----------------------------------------------------------------------------
def rank_group(names, stats, rng):
    """
    FIFA 2026 group ranking:
      points; goal difference; goals scored; head-to-head points / GD / goals
      among tied teams; (fair play and drawing of lots -> random here).
    """
    def base_key(t):
        s = stats[t]
        return (-s["pts"], -(s["gf"] - s["ga"]), -s["gf"])

    ordered = sorted(names, key=lambda t: (base_key(t), rng.random()))

    # head-to-head resolution inside groups of teams still fully tied
    i = 0
    final = []
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and base_key(ordered[j]) == base_key(ordered[i]):
            j += 1
        tied = ordered[i:j]
        if len(tied) > 1:
            h2h = {t: {"pts": 0, "gf": 0, "ga": 0} for t in tied}
            for (x, y), (gx, gy) in stats["_results"].items():
                if x in h2h and y in h2h:
                    h2h[x]["gf"] += gx; h2h[x]["ga"] += gy
                    h2h[y]["gf"] += gy; h2h[y]["ga"] += gx
                    if gx > gy:
                        h2h[x]["pts"] += 3
                    elif gy > gx:
                        h2h[y]["pts"] += 3
                    else:
                        h2h[x]["pts"] += 1; h2h[y]["pts"] += 1
            tied.sort(key=lambda t: (-h2h[t]["pts"],
                                     -(h2h[t]["gf"] - h2h[t]["ga"]),
                                     -h2h[t]["gf"], rng.random()))
        final.extend(tied)
        i = j
    return final


def simulate_group(g, names, elos, teams, fixed_scores, rng, fixture_tally):
    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in names}
    stats["_results"] = {}
    for i, j in GROUP_FIXTURE_PATTERN:
        a, b = names[i - 1], names[j - 1]
        fixed = fixed_scores.get((a, b))
        ga, gb = play_match(elos, teams, a, b, "group", rng, fixed)
        stats["_results"][(a, b)] = (ga, gb)
        stats[a]["gf"] += ga; stats[a]["ga"] += gb
        stats[b]["gf"] += gb; stats[b]["ga"] += ga
        if ga > gb:
            stats[a]["pts"] += 3
            fixture_tally[(a, b)][0] += 1
        elif gb > ga:
            stats[b]["pts"] += 3
            fixture_tally[(a, b)][2] += 1
        else:
            stats[a]["pts"] += 1; stats[b]["pts"] += 1
            fixture_tally[(a, b)][1] += 1
    order = rank_group(list(names), stats, rng)
    return order, stats

# ----------------------------------------------------------------------------
# Knockout stage
# ----------------------------------------------------------------------------
def play_knockout(elos, teams, a, b, stage, rng, ko_overrides, ko_tally):
    """
    Returns (winner, loser, went_to_extra_time, went_to_penalties).
    Elo: a match decided on penalties counts as a draw (eloratings.net rule).
    """
    key = frozenset((a, b))
    ov = ko_overrides.get(key)
    if ov is not None:
        ga, gb = ov["score"]
        if ov.get("pair") and ov["pair"][0] != a:
            ga, gb = gb, ga
        update_elo(elos, teams, a, b, ga, gb, stage)
        if ga > gb:
            return a, b, ov.get("extra_time", False), False
        if gb > ga:
            return b, a, ov.get("extra_time", False), False
        pw = ov["penalties_winner"]
        return (pw, b if pw == a else a, True, True)

    d = effective_diff(elos, teams, a, b, stage)
    _, _, _, _, _, cdf_a, cdf_b = fit_poisson(d)
    ga = _sample_cdf(cdf_a, rng)
    gb = _sample_cdf(cdf_b, rng)
    et = pens = False
    if ga == gb:
        et = True
        ko_tally[a]["draws90"] += 1
        ko_tally[b]["draws90"] += 1
        # extra time: 30 minutes -> Poisson with one third of the rates
        lam_a, lam_b = fit_poisson(d)[0] / 3.0, fit_poisson(d)[1] / 3.0
        ea = _sample_cdf(_cdf_for(lam_a), rng)
        eb = _sample_cdf(_cdf_for(lam_b), rng)
        ga += ea
        gb += eb
        if ga == gb:
            pens = True
            ko_tally[a]["pens"] += 1
            ko_tally[b]["pens"] += 1
    update_elo(elos, teams, a, b, ga, gb, stage)  # pens -> draw for Elo
    if ga > gb:
        return a, b, et, pens
    if gb > ga:
        return b, a, et, pens
    p = 0.5 + PENALTY_ELO_WEIGHT * (elo_expectancy(d) - 0.5)
    if rng.random() < p:
        return a, b, et, True
    return b, a, et, True


_cdf_cache = {}

def _cdf_for(lam):
    key = round(lam, 3)
    c = _cdf_cache.get(key)
    if c is None:
        pmf = _poisson_pmf(key)
        c, s = [], 0.0
        for x in pmf:
            s += x
            c.append(s)
        _cdf_cache[key] = c
    return c

# ----------------------------------------------------------------------------
# One full tournament
# ----------------------------------------------------------------------------
def run_tournament(teams, by_group, annex_c, fixed_group, ko_overrides,
                   rng, acc):
    elos = {t: teams[t]["elo"] for t in teams}

    winners, runners, thirds = {}, {}, {}
    third_stats = {}
    for g in GROUPS:
        order, stats = simulate_group(g, by_group[g], elos, teams,
                                      fixed_group, rng, acc["fixtures"])
        winners[g], runners[g] = order[0], order[1]
        thirds[g] = order[2]
        s = stats[order[2]]
        third_stats[g] = (s["pts"], s["gf"] - s["ga"], s["gf"])
        acc["group_winner"][order[0]] += 1
        acc["group_runnerup"][order[1]] += 1

    # rank third-placed teams: points, GD, goals scored, (lots -> random)
    ranked_thirds = sorted(GROUPS,
                           key=lambda g: (third_stats[g], rng.random()),
                           reverse=True)
    qualified = frozenset(ranked_thirds[:8])
    assign = annex_c[qualified]  # third-place group facing 1A,1B,1D,1E,1G,1I,1K,1L
    third_opponent = {wg: thirds[tg]
                      for (wg, _), tg in zip(ANNEX_C_COLUMNS, assign)}

    for g in qualified:
        acc["advance"][thirds[g]] += 1
    for g in GROUPS:
        acc["advance"][winners[g]] += 1
        acc["advance"][runners[g]] += 1

    # Round of 32
    ko_winner = {}
    for m, (k1, g1, k2, g2) in R32.items():
        a = winners[g1] if k1 == "W" else runners[g1]
        if k2 == "R":
            b = runners[g2]
        else:
            b = third_opponent[g1]
        w, l, et, pens = play_knockout(elos, teams, a, b, "r32", rng,
                                       ko_overrides, acc["ko"])
        ko_winner[m] = w
        acc["reach_r16"][w] += 1

    for m, (m1, m2) in R16.items():
        w, l, et, pens = play_knockout(elos, teams, ko_winner[m1],
                                       ko_winner[m2], "r16", rng,
                                       ko_overrides, acc["ko"])
        ko_winner[m] = w
        acc["reach_qf"][w] += 1

    for m, (m1, m2) in QF.items():
        w, l, et, pens = play_knockout(elos, teams, ko_winner[m1],
                                       ko_winner[m2], "qf", rng,
                                       ko_overrides, acc["ko"])
        ko_winner[m] = w
        acc["reach_sf"][w] += 1

    finalists = []
    for m, (m1, m2) in SF.items():
        w, l, et, pens = play_knockout(elos, teams, ko_winner[m1],
                                       ko_winner[m2], "sf", rng,
                                       ko_overrides, acc["ko"])
        finalists.append(w)
        acc["reach_final"][w] += 1

    champ, _, _, _ = play_knockout(elos, teams, finalists[0], finalists[1],
                                   "final", rng, ko_overrides, acc["ko"])
    acc["champion"][champ] += 1

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Monte Carlo simulation")
    parser.add_argument("n_sims", nargs="?", type=int, default=N_SIMS_DEFAULT,
                        help=f"number of tournament simulations "
                             f"(default {N_SIMS_DEFAULT})")
    parser.add_argument("--refetch-elo", action="store_true", default=False,
                        help="fetch current Elo ratings from eloratings.net "
                             "before simulating and save them to "
                             "data/teams.csv (default: off, use CSV values)")
    args = parser.parse_args()
    n_sims = args.n_sims
    rng = random.Random(RNG_SEED)

    teams, by_group = load_teams()
    if args.refetch_elo:
        print("Refetching Elo ratings...")
        refetch_elo(teams)
    annex_c = load_annex_c()
    results = load_results(by_group)

    # fixed group results
    fixed_group = {}
    for entry in results.get("group_stage", []):
        if entry.get("score") is not None:
            a, b = entry["home"], entry["away"]
            if a not in teams or b not in teams:
                raise ValueError(f"unknown team in results.json: {a} / {b}")
            fixed_group[(a, b)] = tuple(entry["score"])
            fixed_group[(b, a)] = tuple(reversed(entry["score"]))

    # knockout overrides (matched by the unordered pair of team names)
    ko_overrides = {}
    for entry in results.get("knockout", []):
        t = entry["teams"]
        ga, gb = entry["score"]
        ov = {"score": (ga, gb)}
        if ga == gb:
            pw = entry.get("penalties_winner")
            if pw not in t:
                raise ValueError(f"knockout result {t} is a draw - set "
                                 f"'penalties_winner' to one of the two teams")
            ov["penalties_winner"] = pw
            ov["extra_time"] = True
        ko_overrides[frozenset(t)] = ov
        # store score oriented as (t[0], t[1])
        ko_overrides[frozenset(t)]["pair"] = tuple(t)

    # orient override scores correctly regardless of who is "a" in the sim
    oriented = ko_overrides

    # accumulators
    acc = {
        "advance": defaultdict(int), "group_winner": defaultdict(int),
        "group_runnerup": defaultdict(int),
        "reach_r16": defaultdict(int), "reach_qf": defaultdict(int),
        "reach_sf": defaultdict(int), "reach_final": defaultdict(int),
        "champion": defaultdict(int),
        "fixtures": defaultdict(lambda: [0, 0, 0]),
        "ko": defaultdict(lambda: {"draws90": 0, "pens": 0}),
        "ko_matches": defaultdict(int),
    }

    print(f"Running {n_sims} tournament simulations "
          f"({len(fixed_group)//2} fixed group results, "
          f"{len(ko_overrides)} fixed knockout results)...")
    for i in range(n_sims):
        run_tournament(teams, by_group, annex_c, fixed_group, oriented,
                       rng, acc)
        if (i + 1) % max(1, n_sims // 10) == 0:
            print(f"  {i + 1}/{n_sims}")

    write_outputs(teams, by_group, acc, n_sims)


def write_outputs(teams, by_group, acc, n):
    os.makedirs(OUT_DIR, exist_ok=True)
    pct = lambda c: 100.0 * c / n

    # --- per-team stats ---
    rows = []
    for t in teams:
        rows.append({
            "team": t, "group": teams[t]["group"], "elo": teams[t]["elo"],
            "win_group_%": round(pct(acc["group_winner"][t]), 2),
            "advance_from_group_%": round(pct(acc["advance"][t]), 2),
            "reach_R16_%": round(pct(acc["reach_r16"][t]), 2),
            "reach_QF_%": round(pct(acc["reach_qf"][t]), 2),
            "reach_SF_%": round(pct(acc["reach_sf"][t]), 2),
            "reach_final_%": round(pct(acc["reach_final"][t]), 2),
            "champion_%": round(pct(acc["champion"][t]), 3),
        })
    rows.sort(key=lambda r: -r["champion_%"])
    path = os.path.join(OUT_DIR, "team_stats.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- per-fixture probabilities ---
    fpath = os.path.join(OUT_DIR, "group_match_probs.csv")
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "home", "away", "P_home_win_%", "P_draw_%",
                    "P_away_win_%"])
        for g in GROUPS:
            names = by_group[g]
            for i, j in GROUP_FIXTURE_PATTERN:
                a, b = names[i - 1], names[j - 1]
                wn, dr, ls = acc["fixtures"][(a, b)]
                w.writerow([g, a, b, round(pct(wn), 2), round(pct(dr), 2),
                            round(pct(ls), 2)])

    # --- knockout draw stats ---
    kpath = os.path.join(OUT_DIR, "knockout_stats.csv")
    with open(kpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["team", "avg_KO_draws_after_90min_per_tournament",
                    "avg_penalty_shootouts_per_tournament"])
        for t in sorted(teams, key=lambda x: -acc["ko"][x]["draws90"]):
            w.writerow([t, round(acc["ko"][t]["draws90"] / n, 4),
                        round(acc["ko"][t]["pens"] / n, 4)])

    # --- console summary ---
    print("\n=== TOP 15: probability to WIN the World Cup ===")
    for r in rows[:15]:
        print(f"  {r['team']:<22} {r['champion_%']:6.2f}%   "
              f"(advance {r['advance_from_group_%']:5.1f}%, "
              f"win group {r['win_group_%']:5.1f}%)")
    print("\n=== Advancing from each group (advance% / win-group%) ===")
    for g in GROUPS:
        line = "  " + g + ": " + ",  ".join(
            f"{t} {pct(acc['advance'][t]):.0f}/{pct(acc['group_winner'][t]):.0f}"
            for t in by_group[g])
        print(line)
    print(f"\nFiles written to {OUT_DIR}/:")
    print("  team_stats.csv, group_match_probs.csv, knockout_stats.csv")


if __name__ == "__main__":
    main()
