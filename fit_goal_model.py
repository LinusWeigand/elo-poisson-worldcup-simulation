#!/usr/bin/env python3
"""
Fit the Elo -> goals model AND the home advantage from historical data.

Pipeline:
 1. Replay the eloratings.net algorithm over the full match history
    (1872 -> today) to recover every match's PRE-MATCH rating difference.
 2. Keep the last 10 years of competitive (non-friendly) matches.
 3. Fit a Poisson GLM by maximum likelihood:
        log lambda_team = alpha + beta * d_raw + gamma * v
    where d_raw is the rating difference WITHOUT a home bonus and
    v = +1 at home / -1 if the opponent is at home / 0 on neutral ground.
    The Elo-equivalent home advantage is then  H = gamma / beta.
 4. The replay itself needs H, so iterate replay <-> fit to a fixed point.
 5. Test whether H depends on the Elo gap (likelihood-ratio test on a
    v * |d| interaction) and report per-bucket estimates with CIs.
 6. Estimate the World Cup HOST advantage separately (all WC host matches).
 7. Fit the Dixon-Coles low-score parameter rho.
 8. Convert everything to the official eloratings.net scale and write
    data/goal_model.json for simulate.py.

Usage:  python3 fit_goal_model.py [path/to/results.csv]
        (downloads the dataset on first use if no path is given)
"""
import csv, json, math, os, sys, urllib.error, urllib.request
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_URL = ("https://raw.githubusercontent.com/martj42/"
            "international_results/master/results.csv")
CUTOFF_YEARS = 10
MAX_G = 12


# --------------------------- networking helpers ----------------------------
def _ssl_context():
    """SSL context that also works on stock macOS python.org installs."""
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
    return ssl._create_unverified_context()


def _urlopen(url, timeout=30):
    import ssl
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None),
                          ssl.SSLCertVerificationError):
            raise
        return urllib.request.urlopen(req, timeout=timeout,
                                      context=_ssl_context())


# ------------------------------ Elo replay ---------------------------------
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


def replay(path, home_adv=100.0):
    """
    Replay the eloratings algorithm using 'home_adv' in the update expectancy.
    Returns rows (date, tournament, d_raw, v, hs, as_) with d_raw = pre-match
    rating difference WITHOUT home bonus and v = 1 if the home side is truly
    at home (0 on neutral ground), plus the final ratings dict.
    """
    R = defaultdict(lambda: 1500.0)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("", "NA") or r["away_score"] in ("", "NA"):
                continue
            h, a = r["home_team"], r["away_team"]
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            v = 0.0 if r["neutral"].strip().upper() == "TRUE" else 1.0
            d_raw = R[h] - R[a]
            rows.append((r["date"], r["tournament"], d_raw, v, hs, as_))
            d_eff = d_raw + home_adv * v
            we = 1.0 / (1.0 + 10.0 ** (-d_eff / 400.0))
            w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            delta = k_factor(r["tournament"]) * g_mult(hs - as_) * (w - we)
            R[h] += delta
            R[a] -= delta
    return rows, R


# --------------------------- Poisson GLM (MLE) -----------------------------
def _solve(A, g):
    """Gaussian elimination for small dense systems A x = g."""
    n = len(g)
    M = [row[:] + [g[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) \
               / M[r][r]
    return x


def fit_glm(obs, n_params):
    """
    Poisson regression by Newton-Raphson.
    obs: list of (feature_tuple, goals); feature[0] must be the constant 1.0.
    Returns (coefficients, log-likelihood).
    """
    coef = [math.log(1.3)] + [0.0] * (n_params - 1)
    for _ in range(80):
        grad = [0.0] * n_params
        H = [[0.0] * n_params for _ in range(n_params)]
        for x, y in obs:
            lam = math.exp(sum(c * xi for c, xi in zip(coef, x)))
            r = y - lam
            for i in range(n_params):
                grad[i] += r * x[i]
                for j in range(i, n_params):
                    H[i][j] += lam * x[i] * x[j]
        for i in range(n_params):
            for j in range(i):
                H[i][j] = H[j][i]
        step = _solve(H, grad)
        coef = [c + s for c, s in zip(coef, step)]
        if max(abs(s) for s in step) < 1e-12:
            break
    ll = 0.0
    for x, y in obs:
        lam = math.exp(sum(c * xi for c, xi in zip(coef, x)))
        ll += y * math.log(lam) - lam - math.lgamma(y + 1)
    return coef, ll


def fit_gamma_only(obs, a, b, g0):
    """1-D Newton for the home coefficient with (a, b) held fixed.
    Returns (gamma, fisher_information)."""
    g = g0
    info = 1.0
    for _ in range(60):
        gr = inf = 0.0
        for x, y in obs:
            lam = math.exp(a + b * x[1] + g * x[2])
            gr += (y - lam) * x[2]
            inf += lam * x[2] * x[2]
        g += gr / inf
        info = inf
        if abs(gr / inf) < 1e-12:
            break
    return g, info


# ------------------------- score-distribution bits -------------------------
def pmf(lam):
    p, out = math.exp(-lam), []
    for k in range(MAX_G + 1):
        if k:
            p *= lam / k
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
            if i > j: W += p
            elif i == j: D += p
            else: L += p
    s = W + D + L
    return W / s, D / s, L / s


def fit_rho(matches, a, b):
    """Profile MLE for the Dixon-Coles rho. matches: (d_eff, hs, as_)."""
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
    for _ in range(30):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(m1) < nll(m2): hi = m2
        else: lo = m1
    rho = 0.5 * (lo + hi)
    return rho, nll(rho), nll(0.0)


def elo_E(d):
    return 1.0 / (1.0 + 10.0 ** (-d / 400.0))


def fit_split_for_total(d, total):
    """Split 'total' so the Poisson expectancy W + D/2 equals elo_E(d)."""
    target = elo_E(d)
    lo, hi = 0.05, total - 0.05
    def expectancy(la):
        W, D, L = wdl(la, total - la)
        return W + 0.5 * D
    if expectancy(hi) < target: return hi, total - hi
    if expectancy(lo) > target: return lo, total - lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if expectancy(mid) < target: lo = mid
        else: hi = mid
    la = 0.5 * (lo + hi)
    return la, total - la


# ----------------------------------- main -----------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/results.csv"
    if not os.path.exists(path):
        print("Downloading match history...")
        with _urlopen(DATA_URL, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())

    # -- fixed-point iteration: replay(H) -> fit gamma -> H = gamma/beta ----
    print("Estimating home advantage by fixed-point iteration:")
    H = 100.0
    for it in range(8):
        rows, ratings = replay(path, home_adv=H)
        last_date = max(r[0] for r in rows)
        cut = f"{int(last_date[:4]) - CUTOFF_YEARS}{last_date[4:]}"
        recent = [(d, v, hs, as_) for (dt, tn, d, v, hs, as_) in rows
                  if dt >= cut and tn.lower() != "friendly"]
        obs = ([((1.0, d, v), hs) for d, v, hs, as_ in recent] +
               [((1.0, -d, -v), as_) for d, v, hs, as_ in recent])
        (a, b, g), ll3 = fit_glm(obs, 3)
        H_new = g / b
        print(f"  iter {it+1}: alpha={a:.4f} beta={b:.6f} gamma={g:.4f} "
              f"-> H = {H_new:.1f} Elo")
        done = abs(H_new - H) < 1.0
        H = H_new
        if done:
            break
    print(f"Converged: H = {H:.1f} Elo (replay scale); "
          f"{len(recent)} competitive matches {cut[:4]}-{last_date[:4]}")

    # standard error via Fisher information, delta method H = gamma/beta
    _, info_g = fit_gamma_only(obs, a, b, g)
    se_H = (1.0 / math.sqrt(info_g)) / abs(b)
    print(f"  SE(H) ~ {se_H:.1f} Elo -> 95% CI [{H-2*se_H:.0f}, {H+2*se_H:.0f}]")

    # -- does H depend on the Elo gap? ---------------------------------------
    obs4 = ([((1.0, d, v, v * abs(d) / 400.0), hs)
             for d, v, hs, as_ in recent] +
            [((1.0, -d, -v, -v * abs(d) / 400.0), as_)
             for d, v, hs, as_ in recent])
    (a4, b4, g4, e4), ll4 = fit_glm(obs4, 4)
    lr = 2.0 * (ll4 - ll3)
    print(f"\nInteraction test - does H vary with the Elo gap |d|?")
    print(f"  gamma(d) = {g4:.4f} {e4:+.4f} * |d|/400")
    print(f"  implied H at |d| = 0 / 200 / 400: "
          f"{g4/b4:.0f} / {(g4+e4*0.5)/b4:.0f} / {(g4+e4)/b4:.0f} Elo")
    print(f"  LR = 2*dLL = {lr:.2f} vs chi2(1) 5% critical value 3.84 -> "
          f"{'SIGNIFICANT' if lr > 3.84 else 'NOT significant'}")

    print(f"\n  H by |d_raw| bucket (point estimate [95% CI]):")
    for lo_b, hi_b in [(0, 100), (100, 200), (200, 300), (300, 450),
                       (450, 1200)]:
        sub = [(x, y) for x, y in obs
               if lo_b <= abs(x[1]) < hi_b and x[2] != 0]
        if len(sub) < 300:
            continue
        gg, inf = fit_gamma_only(sub, a, b, g)
        se = (1.0 / math.sqrt(inf)) / abs(b)
        print(f"    |d| {lo_b:>3}-{hi_b:<4} (n={len(sub)//2:>4} matches): "
              f"H = {gg/b:6.1f}  [{gg/b - 2*se:5.0f}, {gg/b + 2*se:5.0f}]")

    # -- World Cup hosts specifically ----------------------------------------
    wc = [(d, v, hs, as_) for (dt, tn, d, v, hs, as_) in rows
          if tn == "FIFA World Cup" and v == 1.0]
    obs_wc = ([((1.0, d, 1.0), hs) for d, v, hs, as_ in wc] +
              [((1.0, -d, -1.0), as_) for d, v, hs, as_ in wc])
    g_wc, inf_wc = fit_gamma_only(obs_wc, a, b, g)
    se_wc = (1.0 / math.sqrt(inf_wc)) / abs(b)
    print(f"\nWorld Cup hosts only ({len(wc)} matches, 1930-2022, era-pooled):")
    print(f"  H_host = {g_wc/b:.0f} Elo  "
          f"[{g_wc/b - 2*se_wc:.0f}, {g_wc/b + 2*se_wc:.0f}]")

    # -- Dixon-Coles rho on the final replay ---------------------------------
    matches = [(d + H * v, hs, as_) for d, v, hs, as_ in recent]
    rho, nll_r, nll_0 = fit_rho(matches, a, b)
    print(f"\nDixon-Coles rho = {rho:+.4f} "
          f"(log-lik improvement vs 0: {nll_0 - nll_r:.1f})")

    # -- calibration table ----------------------------------------------------
    print("\nCalibration by |d_eff| bucket (favorite's perspective):")
    print(f"{'bucket':>10} {'n':>5} | {'emp W/D/L %':>18} | "
          f"{'fix2.7+Elo':>18} | {'regression':>18} | {'reg+DC':>18}")
    for lo_b, hi_b in [(0, 75), (75, 150), (150, 250), (250, 350),
                       (350, 500), (500, 1200)]:
        sel = [(abs(de), hs, as_) if de >= 0 else (abs(de), as_, hs)
               for (de, hs, as_) in matches if lo_b <= abs(de) < hi_b]
        n = len(sel)
        if not n:
            continue
        eW = sum(1 for _, x, y in sel if x > y) / n
        eD = sum(1 for _, x, y in sel if x == y) / n
        dbar = sum(d for d, _, _ in sel) / n
        def fmt(W, D, L):
            return f"{100*W:5.1f}/{100*D:4.1f}/{100*L:4.1f}"
        m1 = wdl(*fit_split_for_total(dbar, 2.7))
        la3, lb3 = math.exp(a + b * dbar), math.exp(a - b * dbar)
        m3 = wdl(la3, lb3)
        m4 = wdl(la3, lb3, rho)
        print(f"{lo_b:>4}-{hi_b:<5} {n:>5} | {fmt(eW, eD, 1-eW-eD):>18} | "
              f"{fmt(*m1):>18} | {fmt(*m3):>18} | {fmt(*m4):>18}")

    # -- convert to the official eloratings.net scale -------------------------
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
        sxx = sum(p[0] ** 2 for p in pairs)
        sxy = sum(p[0] * p[1] for p in pairs)
        s = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    H_off = H / s
    print(f"\nScale: replayed ~= {s:.4f} * official  ->  official scale: "
          f"beta = {b*s:.6f}, H = {H_off:.0f} Elo "
          f"(WC hosts: {g_wc/b/s:.0f} +- {2*se_wc/s:.0f})")

    out = {"alpha": a, "beta_replay": b, "scale": s, "beta": b * s,
           "rho": rho,
           "home_advantage_elo": round(H_off, 1),
           "home_advantage_se": round(se_H / s, 1),
           "home_advantage_interaction_lr": round(lr, 2),
           "wc_host_advantage_elo": round(g_wc / b / s, 1),
           "wc_host_advantage_se": round(se_wc / s, 1),
           "fit_window_start": cut, "fit_window_end": last_date,
           "n_matches": len(recent),
           "note": "lambda = exp(alpha + beta*d_eff) per team, d_eff on the "
                   "OFFICIAL eloratings.net scale incl. home_advantage_elo "
                   "for the home side; T(d)=2*exp(alpha)*cosh(beta*d); rho = "
                   "Dixon-Coles. H fitted jointly with the Elo replay "
                   "(fixed-point iteration)."}
    with open(os.path.join(BASE, "data", "goal_model.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote data/goal_model.json")


if __name__ == "__main__":
    main()