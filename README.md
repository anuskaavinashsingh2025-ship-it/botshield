# BotShield

### AI-Based Website Bot Detection Using Behavioral Automata & Machine Learning

## 1. Project Overview & Status

BotShield is a cybersecurity research project that detects automated bots on websites by analyzing user behavior through deterministic finite-state automata (FSM) and machine-learning anomaly classifiers.

The repository includes a fully functional e-commerce storefront ("BotShield Store") instrumented with client-side behavioral telemetry, server-side HTTP request logging, session correlation, an explainable behavioral FSM transducer, and an unsupervised Isolation Forest machine learning pipeline.

### Verified Status
- **Phase 1 (Storefront & Baseline)**: Complete — Flask serving 5 responsive pages with dark UI and product catalog.
- **Phase 2A (Database Architecture)**: Complete — SQLite with WAL mode, foreign keys, indexed `sessions`, `events`, `requests`, and `session_risks` tables.
- **Phase 2B (HTTP Request Logging & Session Correlation)**: Complete — Server-side request logging, cookie synchronization, fail-open resilience, duplicate prevention.
- **Phase 3 (Finite-State Machine Behavioral Detection)**: Complete — Deterministic finite-state transducer (DFST) evaluating navigation patterns, rapid traversal, path cycling, scraper ratios, and human recovery.
- **Phase 4 (Behavioral Feature Extraction & ML)**: Complete — Deterministic 21-feature schema (v1.0.0), unsupervised Isolation Forest anomaly detection, safe cold-start handling, and non-destructive partial risk persistence.
- **Phase 5 (Hybrid Risk Scoring & Testing)**: Complete — Deterministic weighted composite scoring, categorical risk classification (LOW, MEDIUM, HIGH), safe FSM-only fallback on untrained ML models, and idempotent persistence to `session_risks`.
- **Phase 6 (Risk-Based Enforcement & Mitigation)**: Complete — Policy engine mapping risk levels to actions, safe monitoring-only default, thread-safe bounded sliding-window rate limiting, sensitive route protection (`/login`), and fail-open resilience.
- **Phase 7 (Security Administrator Dashboard)**: Complete — Protected SOC web dashboard (`/admin/dashboard`), fail-closed token authentication with constant-time verification, paginated session queries, risk distribution analytics, and drill-down diagnostics without live evaluation overhead.
- **Phase 8 (ML Training, Validation, and Detection Quality Audit)**: Complete — Audited 21-feature contract, atomic model replacement, Isolation Forest score semantics clarification, synthetic scenario evaluation pipeline, train/val/test data separation, and comprehensive quality metrics (Precision, Recall, F1, FPR, FNR).
- **Phase 9 (End-to-End Integration and System Validation)**: Complete — Comprehensive end-to-end integration test suite (`tests/test_e2e_integration.py`) verifying browser telemetry, request logging, SQLite persistence, FSM + ML evaluation, hybrid risk scoring, monitor vs enforce gating, admin visibility, connection restart persistence, fallback handling, and ML artifact integrity.
- **Phase 10 (Red-Team Bot Traffic Simulator and Benchmarking)**: Complete — Realistic human and adversarial bot profiles (`Human`, `RapidScraper`, `CredentialStuffing`, `StealthCrawler`), in-process and live HTTP execution runners, latency benchmarks, and ANSI/JSON reporting CLI.
- **Phase 11 (Asynchronous Evaluation Worker & Pluggable Distributed Rate Limiting)**: Complete — Background session risk evaluation worker (`backend/scoring/worker.py`) offloading hybrid scoring from request paths, pending session discovery queries, thread-safe daemon lifecycle, and pluggable `DistributedRateLimiterAdapter`.
- **Automated Test Suite**: **199 passed in 25.35s** across all 13 test suites (storefront, database, request logging, telemetry, FSM, ML, hybrid scoring, enforcement, admin dashboard, ML evaluation, end-to-end integration, simulation, and async worker).

---

## 2. Architecture & Data Flow

```text
  [ User Browser ]
         │
         ├── (1) Page Request ───────────────► [ Flask App & Middleware ]
         │                                              │
         │                                      (2) Intercept & Log
         │                                              ▼
         │                                     [ requests table ]
         │
         ├── (3) Behavioral Telemetry ───────► [ POST /api/events ]
         │   (Clicks, Scrolls, Mouse, Exit)             │
         │                                      (4) Store Batch
         │                                              ▼
         │                                     [ events table ]
         │
  [ Evaluation Services ] (On-Demand / Background Evaluation)
         │
         ├── (5) Normalization & Extraction ──► [ events + requests ]
         │                                              │
         ├── (6) FSM Evaluation ──────────────► [ evaluate_session_fsm() ]
         │                                              │
         ├── (7) ML Anomaly Scoring ──────────► [ evaluate_session_ml() ]
         │                                              ▼
         ├── (8) Hybrid Risk Scoring ─────────► [ evaluate_session_risk() ]
         │                                              ▼
         └── (9) Risk Persistence ────────────► [ session_risks table ]
                                               (risk_score, fsm_score, ml_score, reasons)
```

> **Integration Notice**: Steps 1–4 execute automatically in real-time on every browser request and telemetry flush. Steps 5–9 are service-level entry points (`evaluate_session_fsm`, `evaluate_session_ml`, and `evaluate_session_risk`) designed for on-demand evaluation, challenge gating, and scheduled jobs; they do not block standard HTTP request cycles in this phase.

---

## 3. Project Structure

```text
botshield/
├── app.py                      # Flask application and route definitions
├── requirements.txt            # Python dependencies (Flask, pytest, numpy, scikit-learn)
├── botshield.db                # SQLite database (WAL mode)
├── backend/
│   ├── database/
│   │   ├── schema.sql          # SQLite schema definitions
│   │   └── db.py               # Database connection and CRUD helpers
│   ├── middleware/
│   │   └── request_logger.py   # Request logging and session correlation hooks
│   ├── detection/
│   │   ├── fsm.py              # Behavioral FSM automaton (DFST) and anomaly rules
│   │   └── service.py          # Session normalization and FSM evaluation service
│   ├── ml/
│   │   ├── features.py         # 21-dimensional behavioral feature extraction (v1.0.0)
│   │   ├── model.py            # Isolation Forest anomaly model and bundle serializer
│   │   ├── train.py            # Model training script (synthetic or DB data)
│   │   └── service.py          # ML session evaluation and persistence service
│   ├── scoring/
│   │   ├── __init__.py         # Public exports (evaluate_session_risk, AsyncEvaluationWorker, etc.)
│   │   ├── service.py          # Hybrid risk scoring service, weights, and classifications
│   │   └── worker.py           # Asynchronous background session evaluation worker and CLI
│   ├── enforcement/
│   │   ├── __init__.py         # Public exports (EnforcementService, BoundedRateLimiter, etc.)
│   │   ├── limiter.py          # Bounded sliding-window rate limiter & DistributedRateLimiterAdapter
│   │   ├── policy.py           # Risk-level to action mapping and EnforcementConfig
│   │   ├── service.py          # Enforcement decision evaluation and freshness caching
│   │   └── middleware.py       # Flask before_request enforcement hook & challenge responses
│   └── simulation/
│       ├── __init__.py         # Public exports for traffic simulator and profiles
│       ├── profiles.py         # Human, Scraper, Credential Stuffing, and Stealth Crawler profiles
│       └── runner.py           # In-process & HTTP benchmark runner and CLI tool
├── static/
│   ├── css/style.css           # Storefront styling (dark theme)
│   └── js/tracker.js           # Client-side anonymous behavioral telemetry tracker
├── frontend/                   # Jinja2 storefront templates
│   ├── base.html
│   ├── index.html
│   ├── products.html
│   ├── product.html
│   ├── login.html
│   └── about.html
├── docs/
│   ├── checkpoints/            # Handoff checkpoint documentation
│   │   ├── PHASE_5_CHECKPOINT.md
│   │   ├── PHASE_6_CHECKPOINT.md
│   │   ├── PHASE_7_CHECKPOINT.md
│   │   ├── PHASE_8_CHECKPOINT.md
│   │   ├── PHASE_9_CHECKPOINT.md
│   │   ├── PHASE_10_CHECKPOINT.md
│   │   └── PHASE_11_CHECKPOINT.md
│   └── ml_design.md            # Detailed ML feature schema and design notes
└── tests/
    ├── test_app.py             # Route and storefront page tests
    ├── test_database.py        # Database schema, normalization, and CRUD tests
    ├── test_request_logging.py # HTTP request logging and correlation tests
    ├── test_telemetry.py       # Telemetry ingestion and tracker script tests
    ├── test_fsm.py             # FSM unit and integration tests
    ├── test_ml.py              # Feature extraction, Isolation Forest, and service tests
    ├── test_hybrid.py          # Hybrid risk scoring, weighting, and fallback tests
    ├── test_enforcement.py     # Enforcement policy, rate limiting, and middleware tests
    ├── test_admin_dashboard.py # Admin dashboard authentication, stats, and session API tests
    ├── test_ml_evaluation.py   # ML evaluation pipeline, metrics, and quality audit tests
    ├── test_e2e_integration.py # Comprehensive end-to-end integration and system validation tests
    ├── test_simulation.py      # Red-team simulation profiles, execution, and benchmark tests
    └── test_scoring_worker.py  # Asynchronous evaluation worker and distributed rate limiter tests
```

---

## 4. Requirements & Installation (Windows PowerShell)

### Prerequisites
- Python 3.11+ (verified on Python 3.13)
- Windows PowerShell or PowerShell Core

### Setup
```powershell
# Clone the repository
git clone <repo-url>
cd botshield

# Create and activate virtual environment (optional but recommended)
py -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
py -m pip install -r requirements.txt
```

---

## 5. Running the Application

```powershell
py app.py
```

The storefront will be accessible at **http://localhost:5000**.

### Available Storefront Routes

| Page | URL | Description |
|---|---|---|
| Home | `/` | Featured products, value propositions, navigation. |
| Products | `/products` | Catalog listing with category filtering. |
| Product Detail | `/product/<id>` | Item specifications, reviews, Add to Cart action. |
| Login | `/login` | Authentication form. |
| About | `/about` | Store background and project information. |
| Telemetry Ingestion | `/api/events` | `POST` endpoint receiving client telemetry batches. |

---

## 6. Database Schema & Tables

Managed in SQLite via parameterized queries with WAL (`Write-Ahead Logging`) mode and foreign-key enforcement (`backend/database/schema.sql`):

1. **`sessions`**:
   - `session_id` (TEXT PK): UUID identifying the browsing session.
   - `created_at` (TEXT UTC): Session initialization timestamp.
   - `last_seen` (TEXT UTC): Most recent activity timestamp.
2. **`events`**:
   - `id` (INTEGER PK AUTOINCREMENT): Telemetry record ID.
   - `session_id` (TEXT FK): Associated session UUID.
   - `event_type` (TEXT): One of `PAGE_VIEW`, `CLICK`, `SCROLL`, `MOUSE_ACTIVITY`, `KEYBOARD_ACTIVITY`, `PAGE_EXIT`.
   - `timestamp` (TEXT UTC): ISO 8601 event timestamp.
   - `page` (TEXT): Normalized URL path where the event occurred.
   - `metadata` (TEXT JSON): Element tags, click IDs, scroll depth, or activity counts.
3. **`requests`**:
   - `id` (INTEGER PK AUTOINCREMENT): Server request log ID.
   - `session_id` (TEXT FK NULLABLE): Correlated session UUID (or NULL if unassociated).
   - `method` (TEXT): HTTP method (`GET`, `POST`, etc.).
   - `path` (TEXT): Normalized path (query parameters and fragments stripped).
   - `status_code` (INTEGER): HTTP status code (e.g. 200, 404).
   - `timestamp` (TEXT UTC): Request receipt timestamp.
   - `user_agent` (TEXT): Sanitized client User-Agent string.
4. **`session_risks`**:
   - `session_id` (TEXT PK FK): Associated session UUID.
   - `risk_score` (REAL): Combined risk indicator $[0.0, 1.0]$.
   - `fsm_score` (REAL): Behavioral FSM risk score $[0.0, 1.0]$.
   - `ml_score` (REAL): Behavioral ML anomaly score $[0.0, 1.0]$.
   - `reasons` (TEXT JSON): List of explainable anomaly descriptions.
   - `updated_at` (TEXT UTC): Last evaluation timestamp.

---

## 7. Configuration Reference

The application reads configuration from environment variables with sensible defaults:

| Variable / Config Key | Default | Description |
|---|---|---|
| `BOTSHIELD_DB` | `botshield.db` | File path for the SQLite database. |
| `BOTSHIELD_COOKIE_NAME` | `botshield_session` | Name of the session cookie. |
| `BOTSHIELD_COOKIE_SECURE` | `0` (False) | Enforce Secure flag on cookies (set to `1` in production HTTPS). |
| `BOTSHIELD_COOKIE_HTTPONLY` | `False` | HttpOnly flag (`False` allows `tracker.js` to coordinate session ID). |
| `REQUEST_LOGGING_ENABLED` | `True` | Toggle server-side request logging on or off. |
| `REQUEST_LOGGING_EXCLUDE_STATIC` | `True` | Exclude `/static/*` assets from the `requests` table. |
| `REQUEST_LOGGING_EXCLUDE_PATHS` | `{"/favicon.ico"}` | Set of specific routes excluded from request logging. |
| `BOTSHIELD_MODEL_PATH` | `models/isolation_forest.joblib` | Path to the serialized ML model artifact. |
| `ENFORCEMENT_MODE` | `monitor` | Enforcement operational mode (`monitor`, `enforce`, or `disabled`). |
| `ENFORCEMENT_FAIL_OPEN` | `True` | If True, database or internal errors permit traffic without outage. |
| `ENFORCEMENT_SENSITIVE_ROUTES` | `("/login",)` | Routes subject to active rate limiting or blocking. |
| `ENFORCEMENT_EXCLUDED_ROUTES` | `("/static", "/favicon.ico", "/api/events")` | Routes that unconditionally bypass enforcement checks. |
| `ENFORCEMENT_RATE_LIMIT_REQUESTS` | `10` | Maximum allowed requests per client key within sliding window. |
| `ENFORCEMENT_RATE_LIMIT_WINDOW` | `10.0` | Sliding window duration in seconds. |
| `ENFORCEMENT_CACHE_TTL` | `60.0` | Maximum age in seconds before a persisted risk score is considered stale. |

---

## 8. Behavioral FSM Detection (Phase 3)

The behavioral detection engine is modeled as a **Deterministic Finite-State Transducer (DFST)**:
$$M = (Q, \Sigma, \Gamma, \delta, \lambda, q_0)$$

- **States ($Q$)**: `INITIAL`, `BROWSING`, `PRODUCT_VIEW`, `CART_ACTION`, `LOGIN_ATTEMPT`, `SUSPICIOUS`, `RECOVERY`.
- **Alphabet ($\Sigma$)**: `NAV_HOME`, `NAV_CATALOG`, `NAV_PRODUCT`, `NAV_ABOUT`, `NAV_LOGIN`, `ACTION_CART`, `ACTION_LOGIN`, `USER_INTERACTION`, `PAGE_EXIT`, `HTTP_API`, `UNKNOWN`.
- **Anomaly Codes**:
  - `ANOMALY_RAPID_TRAVERSAL`: Navigation dwell time $< 0.150$s.
  - `ANOMALY_PATH_CYCLING`: $\ge 4$ repetitive path accesses within $10.0$s.
  - `ANOMALY_HIGH_REQUEST_LOW_TELEMETRY`: Requests $\ge 5$, telemetry/request ratio $< 0.15$.
  - `ANOMALY_RAPID_REQUEST_BURST`: Burst rate $\ge 6.0$ req/s.
  - `ANOMALY_INVALID_ORDERING`: Cart action triggered in `INITIAL` state without product view.
  - `SIGNAL_LOW_INTERACTION`: Contextual signal only (**never** flags a user on its own).
- **Human Recovery**: Suspicious sessions transition to `RECOVERY` after $\ge 3$ interactions and $\ge 2.0$s dwell time, receiving a 60% penalty discount (`recovery_mitigation_factor = 0.40`).

### FSM Evaluation Example

```python
from backend.detection import evaluate_session_fsm

# Evaluates normalized session history and persists to session_risks
result = evaluate_session_fsm(
    session_id="a73ad347-9c61-4740-9bb6-7f3ae0c721d8",
    db_path="botshield.db",
    persist=True,
)

print("Final State:", result.final_state.value)
print("Risk Score: ", result.risk_score)  # or result.fsm_score
print("Anomalies:  ", result.anomaly_codes)
print("Reasons:    ", result.reasons)
```

---

## 9. Behavioral ML Anomaly Detection & Quality Audit (Phase 8)

BotShield extracts a 21-element numerical feature vector (schema `v1.0.0`) capturing activity volumes, temporal dynamics, inter-event statistics, and interaction ratios from session telemetry and server-observed requests. (See [docs/ml_design.md](docs/ml_design.md) for full feature documentation).

### Audited 21-Feature Contract (`v1.0.0`)

| Index | Feature Name | Type | Bounds | Description |
|---|---|---|---|---|
| 0 | `request_count` | `int` | $[0, \infty)$ | Total server-observed HTTP requests |
| 1 | `telemetry_count` | `int` | $[0, \infty)$ | Total client-side browser telemetry events |
| 2 | `total_activity_count` | `int` | $[0, \infty)$ | Sum of requests and telemetry events |
| 3 | `session_duration_seconds` | `float` | $[0.0, \infty)$ | Duration between first and last observed event |
| 4 | `request_rate` | `float` | $[0.0, \infty)$ | Requests per second over session duration |
| 5 | `telemetry_rate` | `float` | $[0.0, \infty)$ | Telemetry events per second |
| 6 | `telemetry_to_request_ratio`| `float` | $[0.0, \infty)$ | Ratio of telemetry events to server requests |
| 7 | `request_to_total_ratio` | `float` | $[0.0, 1.0]$ | Proportion of total activity from server requests |
| 8 | `unique_path_count` | `int` | $[0, \infty)$ | Distinct normalized URL paths visited |
| 9 | `path_repetition_ratio` | `float` | $[0.0, 1.0]$ | Ratio of repeated path visits to total requests |
| 10 | `page_view_count` | `int` | $[0, \infty)$ | Client `page_view` telemetry events |
| 11 | `click_count` | `int` | $[0, \infty)$ | Client `click` telemetry events |
| 12 | `scroll_count` | `int` | $[0, \infty)$ | Client `scroll` telemetry events |
| 13 | `mouse_activity_count` | `int` | $[0, \infty)$ | Client `mouse_move` telemetry events |
| 14 | `keyboard_activity_count` | `int` | $[0, \infty)$ | Client `key_down` telemetry events |
| 15 | `page_exit_count` | `int` | $[0, \infty)$ | Client `page_exit` telemetry events |
| 16 | `mean_inter_event_time` | `float` | $[0.0, \infty)$ | Mean delta between consecutive event timestamps |
| 17 | `std_inter_event_time` | `float` | $[0.0, \infty)$ | Standard deviation of inter-event deltas |
| 18 | `min_inter_event_time` | `float` | $[0.0, \infty)$ | Minimum delta between consecutive events |
| 19 | `rapid_traversal_count` | `int` | $[0, \infty)$ | Transitions faster than 500ms threshold |
| 20 | `error_status_ratio` | `float` | $[0.0, 1.0]$ | Proportion of requests resulting in HTTP 4xx/5xx |

Inference strictly enforces feature vector length and exact ordering against `FEATURE_CONTRACT`. Any missing, NaN, or infinite values are converted to deterministic defaults before scaling.

### Score Semantics & Anomaly Ranking

Isolation Forest is an **unsupervised anomaly detector**; its outputs are anomaly rankings relative to a trained reference distribution, **not calibrated probabilities of automated bot traffic**:
- **Raw Scikit-Learn Score**: `score_samples(X)` yields negative anomaly scores in $[-1.0, 0.0]$, where lower (more negative) values indicate greater abnormality.
- **BotShield Normalization**: BotShield computes $s_{ml} = \min(1.0, \max(0.0, -\text{raw\_score})) \in [0.0, 1.0]$.
- **Score Direction**: Higher scores indicate greater anomaly / deviation from normal baseline.
- **Threshold Decision**: Sessions with $s_{ml} \ge \tau$ (default $\tau = 0.50$ or optimal validation threshold) are classified as anomalous.
- **Interpretation Rule**: Do **not** cite $s_{ml}$ as a probability percentage (e.g. "80% chance of being a bot"). It represents a statistical anomaly distance.

### Model Lifecycle & Artifact Management

- **Atomic Replacement**: Model bundles are saved via a temporary file (`tempfile.NamedTemporaryFile`) and atomically renamed (`os.replace`) to target path, preventing partial or corrupt artifacts if a process terminates during saving.
- **Metadata Envelope**: Artifacts store `model_version`, `feature_names`, `trained_at`, `random_seed`, `contamination`, `dataset_type`, `training_sample_count`, and the fitted `StandardScaler`.
- **Validation on Load**: Loading validates schema compatibility, feature list equality, and exact feature ordering. Incompatible or corrupt artifacts raise `ModelLoadError` and gracefully transition inference to `MLModelStatus.UNTRAINED`.
- **Trust Boundary**: Deserialization relies on Python `joblib`/`pickle`. **Never load model artifacts from untrusted, remote, or user-supplied locations.** Only load trusted, local artifacts managed by the system administrator.
- **Cold-Start Behavior**: If no model exists or an artifact is corrupt, inference returns `status = MLModelStatus.UNTRAINED`, `anomaly_score = None`, and hybrid scoring falls back safely to `FSM_ONLY` mode.

### Safe CLI Utilities (`backend.ml.train`)

```powershell
# 1. Inspect the active 21-feature contract
py -m backend.ml.train --contract

# 2. Inspect active model status and artifact metadata
py -m backend.ml.train --status --out models/isolation_forest.joblib

# 3. Train on reproducible synthetic behavioral scenarios (default seed=42)
py -m backend.ml.train --train --synthetic --samples 500 --seed 42 --out models/isolation_forest.joblib

# 4. Train on extracted database session features (requires approved DB data)
py -m backend.ml.train --train --db botshield.db --min-samples 20 --out models/isolation_forest.joblib

# 5. Evaluate a trained model against synthetic benchmark scenarios
py -m backend.ml.train --evaluate --model models/isolation_forest.joblib --seed 42 --test-samples 200
```

### Reproducible Evaluation & Quality Metrics

The evaluation workflow (`backend.ml.evaluation`) benchmarks models against 5 controlled synthetic behavioral scenarios:
1. `human_browse`: Normal human navigation with realistic click/scroll telemetry and 2–15s delays.
2. `human_casual`: Casual browsing with varied pauses, low error rate, and mouse movements.
3. `rapid_scraper`: Rapid automated crawling with high request rates, sub-500ms intervals, and 0 telemetry.
4. `cycling_crawler`: Systematic crawler visiting identical paths repeatedly with minimal telemetry.
5. `credential_stuffing`: Rapid targeted login requests with elevated 4xx error rates.

#### Dataset Separation & Zero Test Leakage
- **Train Split (60%)**: Used exclusively to fit the scaler and train the Isolation Forest.
- **Validation Split (20%)**: Used exclusively to optimize the anomaly threshold $\tau$ (maximizing F1).
- **Held-Out Test Split (20%)**: Evaluated once at the optimal validation threshold to compute final test metrics. The test set is **never** used during model fitting or threshold selection.

#### Metrics Reported
- **Confusion Matrix**: True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).
- **Precision, Recall, F1 Score**: Evaluated against known synthetic scenario ground truth.
- **False Positive Rate (FPR)** and **False Negative Rate (FNR)**.
- **Synthetic Data Disclaimer**: Evaluation outputs explicitly declare `is_synthetic: True` and include a prominent notice stating that synthetic benchmark results do **not** imply real-world bot detection accuracy. Real-world validation requires independently labeled production traffic.

---

## 10. Hybrid Risk Scoring (Phase 5)

The hybrid risk scoring engine combines rule-based deterministic FSM detection with unsupervised ML anomaly detection into a unified, explainable session risk classification.

### Architecture & Score Semantics
- **FSM Score ($s_{fsm} \in [0.0, 1.0]$)**: Measures deterministic violations of web navigation invariants (rapid traversal, burst rate, path cycling, scraper ratio).
- **ML Anomaly Score ($s_{ml} \in [0.0, 1.0]$)**: Measures statistical deviation from baseline interaction patterns using an Isolation Forest pipeline.
- **Combined Risk Formula (HYBRID Mode)**:
  When an ML model is trained and available:
  $$\text{normalized\_score} = \min(1.0, \max(0.0, w_{fsm} \cdot s_{fsm} + w_{ml} \cdot s_{ml}))$$
  $$\text{composite\_score} = \text{round}(\text{normalized\_score} \times 100.0, 1) \in [0.0, 100.0]$$
  Defaults: $w_{fsm} = 0.60, w_{ml} = 0.40$ (configurable via `HybridConfig`). FSM is weighted higher due to deterministic domain invariant enforcement.

### Cold-Start Fallback & Insufficient Evidence Policies
- **Untrained / Unavailable ML (FSM_ONLY Mode)**:
  When the ML model is untrained (`MLModelStatus.UNTRAINED`), corrupt, incompatible, or missing, the composite score relies strictly on FSM evidence without fabricating an anomaly score or assuming zero risk:
  $$\text{normalized\_score} = s_{fsm}, \quad \text{composite\_score} = \text{round}(s_{fsm} \times 100.0, 1)$$
  An explicit warning is appended (`"ML model is untrained; evaluated in FSM-only fallback mode."`), and `ml_score` remains `None`.
- **Zero Recorded Activity (INSUFFICIENT_EVIDENCE Mode)**:
  Sessions with no browser events and no HTTP requests return `composite_score = 0.0`, `normalized_score = 0.0`, `RiskLevel.LOW`, and a diagnostic warning (`"INSUFFICIENT_SESSION_EVIDENCE"`).

### Risk Classification Thresholds

| Risk Level | Composite Score Range | Description |
|---|---|---|
| `LOW` | `[0.0, 35.0)` | Typical human browsing activity. No automated challenge required. |
| `MEDIUM` | `[35.0, 70.0)` | Elevated anomalies or suspicious timing. Candidate for progressive challenge. |
| `HIGH` | `[70.0, 100.0]` | Confirmed automaton traversal or severe statistical anomaly. |

### Public API & Usage Example

```python
from backend.scoring import (
    HybridConfig,
    HybridRiskResult,
    RiskLevel,
    ScoringMode,
    evaluate_session_risk,
)

# Evaluate session risk (in-memory by default)
result = evaluate_session_risk(
    session_id="a73ad347-9c61-4740-9bb6-7f3ae0c721d8",
    db_path="botshield.db",
    config=HybridConfig(fsm_weight=0.60, ml_weight=0.40, low_threshold=35.0, high_threshold=70.0),
    persist=True,  # Idempotently upserts normalized score and reasons to session_risks
)

print(f"Session:          {result.session_id}")
print(f"Composite Score:  {result.composite_score:.1f} / 100.0")
print(f"Normalized Score: {result.normalized_score:.4f} / 1.0")
print(f"Risk Level:       {result.risk_level.value}")
print(f"Scoring Mode:     {result.scoring_mode.value}")
print(f"FSM Score:        {result.fsm_score:.4f}")
print(f"ML Score:         {result.ml_score if result.ml_score is not None else 'N/A'}")
print(f"Reasons:          {result.reasons}")
print(f"Warnings:         {result.warnings}")
```

### Persistence Semantics
- `persist=False`: Absolutely zero database writes are performed. Suitable for test suites, telemetry inspection, and dry-run evaluations.
- `persist=True`: Safely upserts `risk_score` (normalized $[0.0, 1.0]$), `fsm_score`, `ml_score`, and combined `reasons` into the SQLite `session_risks` table. Repeated evaluations update the existing row idempotently without generating duplicate records.

---

## 11. Risk-Based Enforcement & Mitigation (Phase 6)

The enforcement layer translates session risk scores into active protection for sensitive storefront endpoints (e.g. `/login`).

### Architecture & Stages
1. **Monitoring-Only Mode (Default)**:
   - Configured via `ENFORCEMENT_MODE = "monitor"`.
   - Evaluates risk, queries rate limits, and logs structured decisions with `enforced=False`.
   - **Never blocks or throttles user traffic.** Safe for staging, baseline collection, and gradual rollouts.
2. **Policy Engine (`backend/enforcement/policy.py`)**:
   - `LOW` Risk $\to$ `ALLOW` (normal browsing).
   - `MEDIUM` Risk $\to$ `RATE_LIMIT` (sliding-window request throttling).
   - `HIGH` Risk $\to$ `BLOCK` (HTTP 403 Forbidden).
   - `INSUFFICIENT_EVIDENCE` $\to$ `ALLOW`.
   - Missing / Stale Risk Records $\to$ Configurable safe fallback (`ALLOW` by default).
3. **Bounded Sliding-Window Rate Limiter (`backend/enforcement/limiter.py`)**:
   - Thread-safe sliding window tracking request timestamps.
   - Bounded memory footprint (`max_tracked_clients = 5000` with automatic staleness pruning).
   - Computes exact `Retry-After` seconds upon quota exhaustion (returns HTTP 429).
   - *Multi-worker note*: Operates in-process. For distributed setups across multiple worker processes, a shared backend (such as Redis) is recommended.
4. **Sensitive Route Gating**:
   - Enforcement evaluates only explicitly configured sensitive routes (`ENFORCEMENT_SENSITIVE_ROUTES = ("/login",)`).
   - Excluded routes (`/static/*`, `/favicon.ico`, `/api/events`) unconditionally bypass enforcement.
5. **Fail-Open Resilience**:
   - In case of database locks, query timeouts, or internal exceptions, requests fail open when `ENFORCEMENT_FAIL_OPEN = True`, preventing application outages.

### Enforcement Configuration Example

```python
from backend.enforcement import (
    EnforcementConfig,
    EnforcementMode,
    EnforcementService,
    init_enforcement_middleware,
)

# Optional custom programmatic configuration:
custom_config = EnforcementConfig(
    mode=EnforcementMode.ENFORCE,         # Enable active blocking & rate limiting
    fail_open=True,                       # Never break site on internal errors
    sensitive_routes=("/login",),         # Protect sensitive paths
    rate_limit_requests=5,                # Max 5 requests
    rate_limit_window_seconds=10.0,       # Per 10-second window
    risk_cache_ttl_seconds=60.0,          # Freshness TTL
)

# Initialize on Flask app:
init_enforcement_middleware(app, config=custom_config)
```

---

## 12. Security Administrator Dashboard (Phase 7)

BotShield provides an authorized Security Operations Center (SOC) dashboard at `/admin/dashboard` for investigating bot-detection activity, real-time risk distribution, and individual session diagnostics.

### Purpose & Capabilities

1. **Telemetry & Traffic Overview**:
   - Total tracked sessions, server requests, telemetry events, and evaluated high-risk anomalies.
   - Real-time calculation using parameterized SQLite aggregation queries without loading full tables into memory.
2. **Risk Distribution Analysis**:
   - Proportional distribution bar and counters for LOW ($< 0.35$), MEDIUM ($0.35 \le s < 0.70$), HIGH ($s \ge 0.70$), and unassessed sessions.
3. **Scoring Mode Transparency**:
   - Counts sessions assessed via `HYBRID` (FSM + ML) vs `FSM_ONLY` (cold start / model unavailable) vs unassessed.
4. **Paginated Session Diagnostics Table**:
   - Bounded queries (`limit` 1–100, default 25; `offset` $\ge 0$).
   - Interactive filtering by risk level (`ALL`, `LOW`, `MEDIUM`, `HIGH`, `UNASSESSED`).
   - Allowlisted sorting by `last_seen`, `created_at`, `risk_score`, or `session_id`.
5. **Drill-Down Session Inspection**:
   - Modal view showing composite risk score, FSM score, ML score, explainability factors/reasons, and telemetry breakdown.
   - Sanitized recent requests and event counts without exposing sensitive payload parameters.
6. **Strict Read-Only Design**:
   - Rendering metrics or inspecting sessions performs no database writes and triggers no synchronous model inferences.

### Administrator Authentication & Access Control

Admin views and APIs are protected by `@require_admin_auth` with a **fail-closed** architecture:
- If no token is configured on the server, all requests to `/admin/*` receive **HTTP 403 Forbidden**.
- Uses constant-time string comparison (`hmac.compare_digest`) to prevent timing side-channel attacks.
- Supports four verification channels:
  1. `Authorization: Bearer <ADMIN_TOKEN>` HTTP header.
  2. `X-Admin-Token: <ADMIN_TOKEN>` HTTP header.
  3. `botshield_admin_token` HTTP-only, SameSite=Lax cookie (established via `/admin/login`).
  4. `?token=<ADMIN_TOKEN>` query parameter.
- Never reveals passwords, raw session cookies, credentials, or filesystem tracebacks in API responses.

#### Configuration

Set the administrative token using an environment variable or Flask configuration:

```powershell
# Set token via environment variable in PowerShell:
$env:BOTSHIELD_ADMIN_TOKEN = "your-secure-admin-token-here"

# Or configure in Flask app:
app.config["ADMIN_TOKEN"] = "your-secure-admin-token-here"
```

### Available Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/admin` or `/admin/dashboard` | `GET` | Required | Interactive SOC Web Dashboard UI |
| `/admin/login` | `POST` | Public | Form submission to verify token and issue admin cookie |
| `/admin/logout` | `GET` | Public | Clears administrative session cookie and redirects |
| `/admin/api/stats` | `GET` | Required | Overview metrics, risk distribution, scoring modes, and recent high-risk sessions |
| `/admin/api/sessions` | `GET` | Required | Paginated session listings (`limit`, `offset`, `risk_level`, `sort_by`, `sort_order`) |
| `/admin/api/sessions/<session_id>` | `GET` | Required | Detailed session diagnostics, explainability factors, and telemetry summary |

---

## 13. Running Tests & Quality Verification

Execute the test suite and static checks using Windows PowerShell:

```powershell
# 1. Run all 199 automated tests across 13 test suites
py -m pytest -v

# 2. Run asynchronous evaluation worker test suite specifically
py -m pytest tests/test_scoring_worker.py -v

# 3. Run simulation and red-team benchmark test suite specifically
py -m pytest tests/test_simulation.py -v

# 4. Run End-to-End integration test suite specifically
py -m pytest tests/test_e2e_integration.py -v

# 5. Run ML evaluation and quality audit test suite specifically
py -m pytest tests/test_ml_evaluation.py -v

# 6. Run admin dashboard test suite specifically
py -m pytest tests/test_admin_dashboard.py -v

# 7. Run background evaluation worker (single batch)
py -m backend.scoring.worker --once --batch-size 25

# 8. Run red-team simulation benchmark CLI (in-process)
py -m backend.simulation.runner --mode in-process --sessions 2

# 9. Run bytecode compilation check
py -m compileall app.py backend tests

# 10. Check git diff for whitespace or trailing newline anomalies
git diff --check
```

---

## 14. Security, Privacy & Reliability

- **Fail-Closed Admin Protection**: Admin endpoints fail closed if the admin token is unconfigured; timing attacks are mitigated via constant-time token comparison.
- **Strict Read-Only SOC Operations**: The dashboard never mutates enforcement policies, blocks IPs manually, or alters persisted telemetry.
- **No Sensitive Telemetry**: Form inputs, passwords, raw keystrokes, and URL query parameters are never logged, displayed, or stored.
- **Fail-Open Resilience**: Request logger errors or database locks fail open for general storefront traffic, ensuring legitimate customer sessions are never blocked by logging failures.
- **SQL Injection Prevention**: All database queries use SQLite parameterized queries (`?`).
- **Input Sanitization**: HTTP headers and User-Agents are sanitized and truncated; paths are normalized without query strings.
- **Trusted Local Artifacts**: ML model deserialization expects trusted local files validated against schema version and feature length.
- **Non-Destructive Persistence**: Hybrid scoring updates preserve existing FSM and ML scores when evaluating under partial or fallback modes.
- **Memory Bounding**: Rate limiter bounds client tracking capacity to prevent memory-exhaustion attacks.

---

## 15. Known Limitations & Production Considerations

1. **Development Server**: The Flask development server (`app.run()`) is for local development; use a WSGI/ASGI server (e.g. Gunicorn/Waitress) behind a reverse proxy (e.g. Nginx) in production.
2. **In-Memory Rate Limiting**: The sliding-window rate limiter is in-process; multi-process deployments require a shared key-value store (e.g. Redis).
3. **Single Admin Token**: Current authentication uses a static shared secret token; production environments may integrate with OpenID Connect (OIDC), OAuth2, or multi-user RBAC.
4. **Client Flush Latency**: Browser telemetry is sent asynchronously via periodic beacons or page exit events; single-request scanners may produce HTTP request logs prior to telemetry arrival.
5. **Synthetic vs. Real-World Evaluation**: The Phase 8 evaluation pipeline uses controlled synthetic behavioral scenarios. While it proves software correctness, feature stability, and threshold optimization without test leakage, it does **not** establish real-world bot classification accuracy. Real-world validation requires independently labeled, representative production traffic.
6. **Unsupervised Anomaly Detector**: Isolation Forest measures statistical distance from reference training data, not ground-truth bot identity. It may produce false positives on novel human browsing patterns or false negatives on human-mimicking stealth bots.
