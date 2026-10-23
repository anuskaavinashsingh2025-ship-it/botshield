# BotShield ML Behavioral Anomaly Detection Design (Phase 4)

## Overview

Phase 4 introduces reproducible behavioral feature extraction and an unsupervised machine learning anomaly detection component to complement the deterministic Finite-State Machine (FSM) implemented in Phase 3.

The ML subsystem operates independently from the FSM:
- **Feature Extraction**: Extracts a deterministic, fixed-length 21-element numerical feature vector from a session's normalized telemetry events and server-observed HTTP requests.
- **Model Architecture**: Unsupervised `IsolationForest` pipeline equipped with `StandardScaler` to handle multi-scale behavioral distributions without label contamination.
- **Cold-Start Resilience**: When no trained model artifact exists on disk, the component explicitly returns status `UNTRAINED` with an anomaly score of `None` rather than generating a false zero.
- **Risk Table Isolation**: Persists `ml_score` to `session_risks` while preserving existing `fsm_score`, combined `risk_score`, and FSM explainability reasons intact.

---

## 1. Feature Definitions (Schema Version `1.0.0`)

The feature extraction engine (`backend/ml/features.py`) maps raw session activities into a fixed 21-dimensional vector:

| Index | Feature Name | Units | Expected Bounds | Description & Empty Session Behavior |
|:---:|---|---|---|---|
| 0 | `request_count` | Count | $[0, 10^7]$ | Total server-observed HTTP requests. (0.0 if empty). |
| 1 | `telemetry_count` | Count | $[0, 10^7]$ | Total browser telemetry events. (0.0 if empty). |
| 2 | `total_activity_count` | Count | $[0, 2 \times 10^7]$ | Total session activities ($N = \text{request\_count} + \text{telemetry\_count}$). (0.0 if empty). |
| 3 | `session_duration_seconds` | Seconds | $[0.0, 2.592 \times 10^6]$ | Total time between first and last recorded timestamp ($\le 30$ days). (0.0 if $<2$ timestamps). |
| 4 | `request_rate` | Reqs / sec | $[0.0, 10000.0]$ | $\text{request\_count} / \max(\text{duration}, 1.0)$. (0.0 if empty). |
| 5 | `telemetry_rate` | Events / sec | $[0.0, 10000.0]$ | $\text{telemetry\_count} / \max(\text{duration}, 1.0)$. (0.0 if empty). |
| 6 | `telemetry_to_request_ratio` | Ratio | $[0.0, 100.0]$ | $\text{telemetry\_count} / \max(\text{request\_count}, 1)$. Scrapers exhibit near-0 values. |
| 7 | `request_to_total_ratio` | Proportion | $[0.0, 1.0]$ | $\text{request\_count} / \max(\text{total\_activity}, 1)$. Bots without telemetry approach 1.0. |
| 8 | `unique_path_count` | Count | $[0, 10^5]$ | Number of distinct normalized URL paths visited. (0.0 if empty). |
| 9 | `path_repetition_ratio` | Ratio | $[0.0, 1.0]$ | Repeated path visit proportion: $1.0 - (\text{unique\_paths} / \text{total\_paths})$. |
| 10 | `page_view_count` | Count | $[0, 10^6]$ | Client `PAGE_VIEW` telemetry events. (0.0 if empty). |
| 11 | `click_count` | Count | $[0, 10^6]$ | Client `CLICK` telemetry events. (0.0 if empty). |
| 12 | `scroll_count` | Count | $[0, 10^6]$ | Client `SCROLL` telemetry events. (0.0 if empty). |
| 13 | `mouse_activity_count` | Count | $[0, 10^6]$ | Client `MOUSE_ACTIVITY` velocity/trajectory aggregates. (0.0 if empty). |
| 14 | `keyboard_activity_count` | Count | $[0, 10^6]$ | Client `KEYBOARD_ACTIVITY` count aggregates. (0.0 if empty). |
| 15 | `page_exit_count` | Count | $[0, 10^6]$ | Client `PAGE_EXIT` unload/visibility beacons. (0.0 if empty). |
| 16 | `mean_inter_event_time` | Seconds | $[0.0, 3600.0]$ | Average duration between consecutive chronological actions. (0.0 if $<2$ items). |
| 17 | `std_inter_event_time` | Seconds | $[0.0, 3600.0]$ | Standard deviation of intervals. Programmatic timers exhibit low variance. |
| 18 | `min_inter_event_time` | Seconds | $[0.0, 3600.0]$ | Minimum observed interval. Sub-10ms intervals signal scripted navigation. |
| 19 | `rapid_traversal_count` | Count | $[0, 10^6]$ | Consecutive page transitions with dwell time $< 150$ms. |
| 20 | `error_status_ratio` | Ratio | $[0.0, 1.0]$ | Proportion of server requests returning $4xx$ or $5xx$ HTTP status codes. |

### Data Quality & Privacy Safeguards
- **Zero PII**: No passwords, tokens, query parameters, cookies, User-Agents, or IP addresses are collected as features.
- **Safety Bounds**: All features are validated for non-finite values (`NaN`, `Inf`) and clamped strictly within defined bounds.
- **Tie-Breaking & Deduplication**: Timestamps with identical values are ordered chronologically by source priority (`HTTP_REQUEST` $\to$ `PAGE_VIEW` $\to$ `CLICK`/`SCROLL` $\to$ `PAGE_EXIT`), and identical duplicate records are filtered.

---

## 2. Model Choice & Rationale

### Unsupervised Anomaly Detection (`IsolationForest`)
In accordance with evaluation principles:
1. **No Ground-Truth Dataset Exists**: The repository does not contain trustworthy labeled historical sessions. Fabricating synthetic binary labels (0 = human, 1 = bot) to report artificial precision or recall metrics would violate empirical integrity.
2. **Isolation Forest Suitability**:
   - Isolates anomalies by randomly selecting a feature and randomly selecting a split value between the minimum and maximum values of the selected feature.
   - Scripted bot patterns (rapid burst rates, zero mouse movement, disproportionate telemetry-to-request ratios) produce short tree paths and isolate quickly compared to complex human browsing distributions.
3. **Reproducible Pipeline**:
   - Preprocessing (`StandardScaler`) standardizes features with wildly different units (e.g. seconds vs counts vs ratios).
   - Single-threaded execution (`n_jobs=1`) prevents process spawning deadlocks across platforms.

---

## 3. Training & Persistence

### Training Entry Point
```python
from backend.ml import train_unsupervised_model

# Feature matrix: shape (N, 21), minimum N >= 10
training_vectors = [...] 

metadata = train_unsupervised_model(
    training_data=training_vectors,
    model_path="models/isolation_forest.joblib",
    contamination=0.10,
    random_state=42,
    min_samples=10,
)
```

### Persistence & Environment Configuration
- **Model Serialization**: Uses `joblib` with an explicit integrity bundle containing:
  - `pipeline`: Fitted `Pipeline([('scaler', StandardScaler()), ('forest', IsolationForest())])`.
  - `metadata`: `schema_version`, `feature_names`, `trained_at`, `training_sample_count`, `model_type`, `contamination`.
- **Configurable Path**: Resolves from `BOTSHIELD_MODEL_PATH` environment variable, falling back to `<project_root>/models/isolation_forest.joblib`.
- **Integrity Validation**: Rejects corrupted files (`MLModelStatus.CORRUPT`) or version/dimension mismatches (`MLModelStatus.INCOMPATIBLE`) gracefully.

---

## 4. Inference & Score Interpretation

### Score Semantics
- **Score Range**: Continuous float bounded in $[0.0, 1.0]$.
- **Interpretation**: Higher values indicate higher behavioral anomaly relative to typical browsing distributions.
- **Rule of Thumb**:
  - `0.0 - 0.45`: Standard browsing patterns.
  - `0.45 - 0.65`: Unusual interaction dynamics or mild scraping indicators.
  - `0.65 - 1.0`: Extreme behavioral outlier (rapid traversal, burst navigation, zero telemetry).
- **Cold-Start Fallback**: If no model is trained, `status="untrained"` and `anomaly_score=None`. Downstream logic must check `status == MLModelStatus.AVAILABLE` before consuming `anomaly_score`.

---

## 5. Service Usage

```python
from backend.ml import evaluate_session_ml, MLModelStatus

# Load session events & requests, compute features, execute inference
result = evaluate_session_ml(
    session_id="f8a12c34-5678-90ab-cdef-1234567890ab",
    db_path="botshield.db",
    persist=True,  # Persists ml_score into session_risks without touching fsm_score
)

if result.status == MLModelStatus.AVAILABLE:
    print(f"ML Anomaly Score: {result.anomaly_score:.4f}")
    print(f"Is Outlier: {result.is_anomaly}")
else:
    print(f"Model unavailable: {result.status.value} (warnings: {result.warnings})")
```
