#!/usr/bin/env python3
"""
Fit the Elo -> goals model from historical international results.

Pipeline:
 1. Replay the eloratings.net algorithm over the full match history
    (1872 -> today) to recover every match's PRE-MATCH Elo difference.
 2. Keep the last 10 years of competitive (non-friendly) matches.
 3. Fit a Poisson regression  log lambda = alpha + beta * d_eff  by
    maximum likelihood (Newton-Raphson), where d_eff is the team's
    effective Elo edge (incl. +100 home advantage when not neutral).
    => expected total goals  T(d) = 2 * e^alpha * cosh(beta * d)
 4. Fit the Dixon-Coles low-score dependence parameter rho by profile MLE.
 5. Print a calibration table (empirical W/D/L by Elo-gap bucket vs models).
 6. Write data/goal_model.json for simulate.py.

Usage:  python3 fit_goal_model.py [path/to/results.csv]
        (downloads the dataset from GitHub if no path given and not cached)
"""
import csv, json, math, os, sys, urllib.error, urllib.request
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_URL = ("https://raw.githubusercontent.com/martj42/"
            "international_results/master/results.csv")
CUTOFF_YEARS = 10
HOME_ADV = 100.0
MAX_G = 12

def _ssl_context():
    """
    Build an SSL context that works on stock macOS python.org installs,
    where the bundled OpenSSL has no CA certificates.
    Order: system default -> certifi (if installed) -> unverified (warned).
    """
    import ssl
    ctx = ssl.create_default_context()
    try:
        if ctx.cert_store_stats().get("x509_ca", 0) > 0:
            return ctx
    except Exception:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    print("  WARNING: no CA certificates available (common with python.org "
          "installs on macOS).\n"
          "  Fix permanently with:  pip install certifi   or by running\n"
          "  /Applications/Python 3.XX/Install Certificates.command\n"
          "  Proceeding WITHOUT certificate verification for this download.")
    ctx = ssl._create_unverified_context()
    return ctx


def _urlopen(url, timeout=30):
    import ssl, urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        return urllib.request.urlopen(req, timeout=timeout,
                                      context=_ssl_context())


def k_factor(tournament):
    t = tournament.lower()
    if "fifa world cup" in t and "qualification" not in t:
        return 60
    if any(s in t for s in ("euro", "copa am", "africa cup", "asian cup",
                            "gold cup", "confederations", "oceania nations")) \
            and "qualification" not in t:
        return 50
    if "qualification" in t or "nations league" in t:
        return 40
    if t == "friendly":
        return 20
    return 30

def g_mult(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)

def replay(path):
    """Replay eloratings algorithm; yield (date, tourn, d_eff_home, hs, as_, neutral)."""
    R = defaultdict(lambda: 1500.0)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
                continue
            h, a = r["home_team"], r["away_team"]
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            neutral = r["neutral"].strip().upper() == "TRUE"
            d_eff = R[h] - R[a] + (0.0 if neutral else HOME_ADV)
            rows.append((r["date"], r["tournament"], d_eff, hs, as_))
            we = 1.0 / (1.0 + 10.0 ** (-d_eff / 400.0))
            w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            delta = k_factor(r["tournament"]) * g_mult(hs - as_) * (w - we)
            R[h] += delta
            R[a] -= delta
    return rows, R

def fit_poisson_regression(obs):
    """obs: list of (d_eff, goals). Fit log lam = a + b*d by Newton-Raphson."""
    a, b = math.log(1.3), 0.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for d, y in obs:
            lam = math.exp(a + b * d)
            r = y - lam
            g0 += r;        g1 += r * d
            h00 += lam;     h01 += lam * d;   h11 += lam * d * d
        det = h00 * h11 - h01 * h01
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        a += da; b += db
        if abs(da) < 1e-12 and abs(db) < 1e-10:
            break
    return a, b

def pmf(lam):
    p, out = math.exp(-lam), []
    for k in range(MAX_G + 1):
        if k: p *= lam / k
        out.append(p)
    s = sum(out)
    return [x / s for x in out]

def dc_tau(i, j, la, lb, rho):
    if i == 0 and j == 0: return 1.0 - la * lb * rho
    if i == 0 and j == 1: return 1.0 + la * rho
    if i == 1 and j == 0: return 1.0 + lb * rho
    if i == 1 and j == 1: return 1.0 - rho
    return 1.0

def wdl(la, lb, rho=0.0):
    pa, pb = pmf(la), pmf(lb)
    W = D = L = 0.0
    for i, x in enumerate(pa):
        for j, y in enumerate(pb):
            p = x * y * dc_tau(i, j, la, lb, rho)
            if   i > j: W += p
            elif i == j: D += p
            else: L += p
    s = W + D + L
    return W / s, D / s, L / s

def fit_rho(matches, a, b):
    """Profile MLE for Dixon-Coles rho over score pmf with regression lambdas."""
    def nll(rho):
        tot = 0.0
        for d, hs, as_ in matches:
            la, lb = math.exp(a + b * d), math.exp(a - b * d)
            pa, pb = pmf(la), pmf(lb)
            norm = sum(pa[i] * pb[j] * dc_tau(i, j, la, lb, rho)
                       for i in range(MAX_G + 1) for j in range(MAX_G + 1))
            if hs <= MAX_G and as_ <= MAX_G:
                p = pa[hs] * pb[as_] * dc_tau(hs, as_, la, lb, rho) / norm
                tot -= math.log(max(p, 1e-300))
        return tot
    lo, hi = -0.15, 0.15
    for _ in range(30):                      # golden-ish bisection on derivative
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(m1) < nll(m2): hi = m2
        else: lo = m1
    return 0.5 * (lo + hi), nll(0.5 * (lo + hi)), nll(0.0)

def elo_E(d): return 1.0 / (1.0 + 10.0 ** (-d / 400.0))

def fit_split_for_total(d, total, rho=0.0):
    """Split 'total' so that W + D/2 (with optional DC rho) == elo_E(d)."""
    target = elo_E(d)
    lo, hi = 0.05, total - 0.05
    def expectancy(la):
        W, D, L = wdl(la, total - la, rho)
        return W + 0.5 * D
    if expectancy(hi) < target: la = hi
    elif expectancy(lo) > target: la = lo
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if expectancy(mid) < target: lo = mid
            else: hi = mid
        la = 0.5 * (lo + hi)
    return la, total - la

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/results.csv"
    if not os.path.exists(path):
        print("Downloading match history...")
        with _urlopen(DATA_URL, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    print("Replaying Elo over full history...")
    rows, ratings = replay(path)
    print(f"  {len(rows)} matches replayed; "
          f"e.g. Spain {ratings['Spain']:.0f}, Argentina {ratings['Argentina']:.0f}, "
          f"Brazil {ratings['Brazil']:.0f}, Germany {ratings['Germany']:.0f}")

    last_date = max(r[0] for r in rows)
    cut = f"{int(last_date[:4]) - CUTOFF_YEARS}{last_date[4:]}"
    recent = [(d_eff, hs, as_) for (dt, tn, d_eff, hs, as_) in rows
              if dt >= cut and tn.lower() != "friendly"]
    print(f"  fitting window: {cut} .. {last_date}, competitive matches: {len(recent)}")

    # two observations per match (each team's perspective) -> symmetric model
    obs = [(d, hs) for d, hs, as_ in recent] + [(-d, as_) for d, hs, as_ in recent]
    a, b = fit_poisson_regression(obs)
    print(f"\nPoisson regression: log lambda = {a:.5f} + {b:.6f} * d_eff")
    print(f"  T(0) = {2*math.exp(a):.3f} goals; "
          f"T(400) = {2*math.exp(a)*math.cosh(b*400):.3f}; "
          f"T(800) = {2*math.exp(a)*math.cosh(b*800):.3f}")

    rho, nll_rho, nll_0 = fit_rho(recent, a, b)
    print(f"Dixon-Coles rho = {rho:+.4f}  "
          f"(log-lik improvement vs rho=0: {nll_0 - nll_rho:.1f})")

    # ---- calibration table ----
    buckets = [(0,75),(75,150),(150,250),(250,350),(350,500),(500,1200)]
    print("\nCalibration by |d_eff| bucket (favorite's perspective):")
    print(f"{'bucket':>10} {'n':>5} | {'emp W/D/L %':>20} | "
          f"{'fixT2.7+Elo':>20} | {'T(d)+Elo':>20} | {'regression':>20} | {'reg+DC':>20}")
    for lo_b, hi_b in buckets:
        sel = [(abs(d), hs, as_) if d >= 0 else (abs(d), as_, hs)
               for d, hs, as_ in recent if lo_b <= abs(d) < hi_b]
        n = len(sel)
        if not n: continue
        eW = sum(1 for _, x, y in sel if x > y) / n
        eD = sum(1 for _, x, y in sel if x == y) / n
        dbar = sum(d for d, _, _ in sel) / n
        def fmt(W, D, L): return f"{100*W:5.1f}/{100*D:4.1f}/{100*L:4.1f}"
        la1, lb1 = fit_split_for_total(dbar, 2.7)
        m1 = wdl(la1, lb1)
        T = 2 * math.exp(a) * math.cosh(b * dbar)
        la2, lb2 = fit_split_for_total(dbar, T)
        m2 = wdl(la2, lb2)
        la3, lb3 = math.exp(a + b * dbar), math.exp(a - b * dbar)
        m3 = wdl(la3, lb3)
        m4 = wdl(la3, lb3, rho)
        print(f"{lo_b:>4}-{hi_b:<5} {n:>5} | {fmt(eW,eD,1-eW-eD):>20} | "
              f"{fmt(*m1):>20} | {fmt(*m2):>20} | {fmt(*m3):>20} | {fmt(*m4):>20}")

    # --- convert beta to the OFFICIAL eloratings.net scale ---
    # The replayed ratings live on a slightly different (compressed) scale.
    # Estimate the affine map replayed ~= s*official + c over the 48 WC teams
    # and store beta_official = beta * s for use with official ratings.
    s = 1.0
    teams_csv = os.path.join(BASE, "data", "teams.csv")
    if os.path.exists(teams_csv):
        alias = {"Curacao": "Curaçao"}
        pairs = []
        with open(teams_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nm = alias.get(r["team"], r["team"])
                if nm in ratings:
                    pairs.append((float(r["elo"]), ratings[nm]))
        n = len(pairs)
        sx = sum(p[0] for p in pairs); sy = sum(p[1] for p in pairs)
        sxx = sum(p[0]**2 for p in pairs); sxy = sum(p[0]*p[1] for p in pairs)
        s = (n*sxy - sx*sy) / (n*sxx - sx*sx)
        print(f"Scale: replayed ~= {s:.4f} * official + c  "
              f"({n} teams matched) -> beta_official = {b*s:.6f}")

    out = {"alpha": a, "beta_replay": b, "scale": s, "beta": b * s,
           "rho": rho,
           "fit_window_start": cut, "fit_window_end": last_date,
           "n_matches": len(recent),
           "note": "per-team rate lambda = exp(alpha + beta*d) with d on the "
                   "OFFICIAL eloratings.net scale (incl. host bonus); "
                   "T(d)=2*exp(alpha)*cosh(beta*d); rho = Dixon-Coles "
                   "low-score dependence"}
    with open(os.path.join(BASE, "data", "goal_model.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote data/goal_model.json")

if __name__ == "__main__":
    main()