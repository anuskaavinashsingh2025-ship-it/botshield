# Hybrid FSM + ML Scoring

How BotShield turns one session's activity into three scores and a classification.
Implemented in `backend/scoring/service.py` (`evaluate_session_risk`, `compute_hybrid_score`).

## Score contract

Every score is in **[0, 1]** and **higher always means more bot-like**.

| Score | Source | Meaning |
|---|---|---|
| FSM | `backend/detection` behavioral FSM | Rule-based risk from navigation invariants (rapid traversal, path cycling, low telemetry, …). |
| ML | `backend/ml` Isolation Forest | Anomaly score s(x) = −`score_samples(x)` (Liu et al.). **Not a calibrated bot probability.** `None` whenever ML did not produce a score. |
| Hybrid | combination below | Final score used for classification, the dashboard and enforcement. |

The ML model also makes its own binary decision: a session is anomalous when
s(x) exceeds the model's trained boundary (−`offset_`, set by `contamination`).
That threshold is reported as `decision_threshold`.

## Hybrid formula

| Mode | When | Hybrid score H |
|---|---|---|
| `HYBRID` | ML model available and returned a finite score | H = 0.60 · FSM + 0.40 · ML |
| `FSM_ONLY` | ML untrained, incompatible, corrupt or errored | H = FSM |
| `INSUFFICIENT_EVIDENCE` | no events and no requests | H = 0 |

Weights and thresholds live in `HybridConfig` (weights must sum to 1).
Classification uses H × 100: **LOW** < 35 ≤ **MEDIUM** < 70 ≤ **HIGH**.

When ML is unavailable no placeholder is ever substituted: invalid features
(wrong length, NaN/Inf) and non-finite model output return status `error` with
no score, and the session is scored `FSM_ONLY`.

## Detector decisions (dashboard metrics)

A detector "flags a session as bot" when:

- **FSM:** FSM ≥ 0.70 (the HIGH band)
- **ML:** the model's own anomaly decision (s(x) > `decision_threshold`)
- **Hybrid:** classification is HIGH

Precision, recall, F1 and false-positive rate are computed only over sessions
with a ground-truth label in `session_labels`; ML metrics only over sessions ML
actually scored. Labels are written by the traffic simulator
(`human` = 0, `scraper` / `credential` / `stealth` = 1):

```bash
python -m backend.simulation.runner --mode http --url http://127.0.0.1:5000
```

The dashboard also shows the offline benchmark from `backend.ml.evaluation`,
which uses **synthetic** labeled scenarios and is not a real-world accuracy claim.

## Live pipeline

1. `tracker.js` posts browser events with the session ID from the
   `botshield_session` cookie; the request logger correlates server requests with
   the same cookie once the session exists.
2. The background worker (`backend/scoring/worker.py`, started by `python app.py`,
   disable with `BOTSHIELD_SCORING_WORKER=0`) re-scores sessions with new events or
   new requests every 5 s and persists them to `session_risks` plus one row in
   `session_risk_history`.
3. `GET /api/risk/current` runs the same evaluation for the caller's own session
   (no persistence, 1 s cache) for the storefront Monitor panel.
4. `GET /admin/api/analytics` (admin token required) returns the score
   comparison, model status and performance metrics for the dashboard.

Monitor polling (`/api/risk/current`, `/api/automata/status`) and admin API
polling (`/admin/api/*`) are never logged as requests, so they cannot feed the
FSM/ML features of the session that polls them.

## Storage

`session_risks.ml_score` is `NOT NULL` (pre-existing schema), so it holds 0.0
when ML did not run. The `ml_status` and `scoring_mode` columns say whether it is
a real ML score; APIs report `ml_score: null` in that case.
`session_risk_history.ml_score` is nullable and stores `NULL` directly.

## Local demo mode

For a local presentation the admin dashboard can open without the token:

```bash
BOTSHIELD_DEMO_MODE=1 python app.py
```

- Applies only to requests from this machine (127.0.0.1 / ::1) without proxy
  headers; any other client still needs the administrator token.
- `BOTSHIELD_ENV=production` always disables it.
- `/admin/login` redirects straight to the dashboard and a "LOCAL DEMO MODE"
  banner is shown. Without the variable, token authentication is unchanged.

With the demo server running, generate the labeled SIMULATED scenario
(2 sessions each of human, scraper, credential stuffing, stealth crawler):

```bash
python -m backend.simulation.demo
```

Then open http://127.0.0.1:5000/admin/dashboard and press **Simulated**.
These sessions are synthetic; their scores illustrate automation-like behavior
and are not evidence about any real person.

### Live demo page

In demo mode, http://127.0.0.1:5000/admin/demo (also linked from the dashboard)
runs scenarios live: human shopper, human-then-automated-burst, scraper,
credential stuffing, low-and-slow crawler, headless browser. Each run uses a
fresh session labeled `simulation:live-<scenario>`, is driven in real time
through the storefront and request logger, and is re-scored twice a second.
The page shows the FSM/ML/hybrid timeline and explains every change:

- **FSM:** each rule that fired, its points and its evidence (they sum to the score)
- **ML:** score vs the model threshold, and the three inputs furthest from the
  training data (an indication of what is unusual, not a causal attribution)
- **Hybrid:** the formula with the actual numbers

"Send a burst from this browser" sends 20 real requests from your own browser
using a fresh session. The page and its APIs return 404 outside demo mode.

Why a fresh session: ML features summarize a whole session, and the session ID
persists in localStorage, so a long-lived browser session barely changes when
one more click happens.

## Training the model

```bash
python -m backend.ml.train --train --realistic --samples 400 --contamination 0.02   # writes models/isolation_forest.joblib
python -m backend.ml.train --status
python -m backend.ml.train --evaluate --realistic
```

`--realistic` trains on pipeline-faithful simulated human sessions
(`backend/ml/realistic.py`): raw request and event rows that reproduce
tracker.js and the request logger (page view 10–60 ms after the request,
3 s telemetry flushes logged as `POST /api/events`, page exits, multi-visit
sessions), including sessions observed part-way. Counts and durations are
log1p-transformed before scaling. Bots (scraper, credential stuffing, stealth
crawler, headless browser) are generated only for the held-out evaluation.

The previous model (`--synthetic`, drawn directly as feature vectors) flagged
every real session as anomalous; it is kept as
`models/isolation_forest.synthetic-baseline.joblib`.

Limits: training data is still simulated. Held-out results measure separability
under these generators, not real-world accuracy, and unusual real sessions (for
example a long development session full of reloads and admin traffic) can still
score as anomalous. Retraining on real, labeled human sessions is the next step.
