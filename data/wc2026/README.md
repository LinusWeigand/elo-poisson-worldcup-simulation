# World Cup 2026 Monte Carlo Simulation

Monte Carlo simulation of the FIFA World Cup 2026 (USA / Mexico / Canada,
June 11 - July 19, 2026) driven by eloratings.net Elo ratings.

## Quick start

    python3 simulate.py                  # 10,000 tournaments (~5 s, stdlib only)
    python3 simulate.py 100000           # more precision
    python3 simulate.py --refetch-elo    # pull current Elo ratings first (default: off)

`--refetch-elo` downloads the latest ratings from eloratings.net (with
international-football.net as a fallback mirror), updates `data/teams.csv`,
and then simulates. Without the flag (the default) the ratings in
`data/teams.csv` are used as-is. If fetching or parsing fails for any
reason, the run continues with the CSV values and prints a warning — your
data is never clobbered by a bad fetch.

Outputs land in `output/`:

| file | contents |
|---|---|
| `team_stats.csv` | per team: P(win group), P(advance from group), P(reach R16/QF/SF/Final), P(champion) |
| `group_match_probs.csv` | every group fixture: P(home win), P(draw), P(away win) — averaged over simulations, so evolving Elo and fixed results are reflected |
| `knockout_stats.csv` | per team: expected number of knockout matches drawn after 90' and penalty shootouts per tournament |

## Entering real results as the tournament progresses

Edit `data/results.json` (auto-created on first run) and rerun the script.

**Group stage** — every fixture starts with `"score": null`, which means
"not played yet → simulate it". To lock in a real result, replace null with
the score:

```json
{ "group": "A", "home": "Mexico", "away": "South Africa", "score": [2, 0] }
```

Fixed results are replayed identically in every simulation, including their
Elo update, so all downstream probabilities condition on them. The listed
fixture order is by draw position; since every pair in a group meets exactly
once, you can always find the pairing you need.

**Knockout stage** — append to the `"knockout"` list. Score is *after extra
time*; if it was a draw, name the shootout winner:

```json
{ "teams": ["Mexico", "Croatia"], "score": [1, 1], "penalties_winner": "Mexico" }
```

The override is applied in any simulation in which those two teams actually
meet in a knockout match (in branches where they don't meet, it is ignored —
the group-stage results you entered will normally pin the bracket down).

## Model

**Elo → win expectancy.** `E = 1 / (1 + 10^((Rb − Ra) / 400))`, with effective
ratings that include the host bonus (below).

**Elo → Poisson goals.** Each match samples goals from two independent
Poisson distributions with `λ_home + λ_away = 2.70` (configurable
`AVG_TOTAL_GOALS`). The split is fitted by bisection so the Poisson match
expectancy `P(win) + ½·P(draw)` equals the Elo expectancy — the *closest
fitting* Poisson. Draws therefore emerge naturally (~26% between equals),
giving the goal counts and goal differentials needed for group standings.
At large Elo gaps (≳550) the fit saturates: the favorite's Poisson win
probability stays below the raw Elo expectancy because the underdog's rate
is floored — the Elo-vs-Poisson disparity grows with the rating gap, by
design.

**In-simulation Elo updates.** After every match (simulated *or* fixed),
ratings update with the eloratings.net rule: `ΔR = K · G · (W − We)` with
`K = 60` (World Cup finals) and margin multiplier `G` (1 for ≤1-goal margin,
1.5 for 2, `(11+N)/8` for N ≥ 3). A team on a deep run carries its earned
rating into later rounds.

**Host advantage.** USA, Mexico and Canada receive a +100 Elo bonus
(eloratings.net's home-advantage value, consistent with hosts historically
winning 66.0% vs 39.9% for non-hosts). Defaults in `HOST_BONUS`: all three
hosts through the R16; from the quarter-finals every match is on US soil, so
only the USA keeps the bonus. Edit the dict to change the assumption.

**Group stage.** Full round robin; ranking by FIFA 2026 tiebreakers:
points → goal difference → goals scored → head-to-head (points/GD/goals)
among tied teams → random (stand-in for fair-play points / drawing of lots).

**Round of 32.** Top two per group advance plus the 8 best third-placed
teams (ranked by points, GD, goals; random as the final tiebreaker). The
bracket follows FIFA's official match plan (matches 73–88) and third-place
slotting uses the **full 495-combination Annex C table** from the tournament
regulations (`data/annex_c.txt`, validated on load against the per-match
constraints).

**Knockout draws.** 90 minutes are sampled from the fitted Poisson; a draw
(probability reported in `knockout_stats.csv`) leads to 30 minutes of extra
time (Poisson at ⅓ rate), then penalties. Shootouts are
`0.5 + 0.25·(E − 0.5)` for the higher-rated side (`PENALTY_ELO_WEIGHT`).
Following eloratings.net convention, a match decided on penalties counts as
a draw for the Elo update, while the shootout winner advances.

## Data

- **Elo ratings**: eloratings.net, snapshot June 11, 2026 (`data/teams.csv`) —
  edit freely if you want fresher numbers.
- **Groups**: official Final Draw (Dec 5, 2025) + March 2026 playoff winners
  (Czechia, Bosnia and Herzegovina, Türkiye, Sweden, DR Congo, Iraq).
- **Bracket & Annex C**: FIFA 2026 tournament regulations.

## Tuning knobs (top of `simulate.py`)

`N_SIMS_DEFAULT`, `AVG_TOTAL_GOALS`, `K_FACTOR`, `HOST_BONUS`,
`PENALTY_ELO_WEIGHT`, `RNG_SEED` (set an int for reproducible runs).
