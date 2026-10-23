"""Tests for BotShield Red-Team Bot Traffic Simulator and Benchmark (Phase 10).

Validates:
1. Behavioral profile generation (Human, Scraper, Credential Stuffing, Stealth Crawler)
2. In-process simulation execution with isolated database and model
3. Accurate detection and classification (Human -> LOW, Scraper -> HIGH, etc.)
4. Benchmark metric aggregation and latency measurements
5. CLI execution and JSON serialization
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest

from backend.database.db import get_connection, init_db, upsert_session
from backend.ml.model import IsolationForestModel, train_unsupervised_model
from backend.ml.train import generate_synthetic_baseline
from backend.simulation.profiles import (
    BaseBehavioralProfile,
    CredentialStuffingProfile,
    HumanProfile,
    ProfileType,
    RapidScraperProfile,
    StealthCrawlerProfile,
    get_profile,
)
from backend.simulation.runner import (
    ProfileBenchmarkResult,
    SessionSimulationResult,
    SimulationBenchmarkSummary,
    SimulationConfig,
    SimulationRunner,
    _format_table,
    main,
    run_simulation,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with complete BotShield schema."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def trained_model_path():
    """Create an isolated trained IsolationForest model artifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "isolation_forest.joblib"
        X = generate_synthetic_baseline(sample_count=80, random_state=42)
        train_unsupervised_model(
            training_data=X,
            model_path=model_file,
            contamination=0.1,
            random_state=42,
        )
        yield str(model_file)


# ==============================================================================
# 1. Profile Registry and Plan Generation Tests
# ==============================================================================

def test_profile_registry_and_factory():
    """get_profile should instantiate valid profiles and reject unknown ones."""
    human = get_profile("human")
    assert isinstance(human, HumanProfile)
    assert human.profile_type == ProfileType.HUMAN

    scraper = get_profile(ProfileType.RAPID_SCRAPER)
    assert isinstance(scraper, RapidScraperProfile)
    assert scraper.profile_type == ProfileType.RAPID_SCRAPER

    credential = get_profile("credential")
    assert isinstance(credential, CredentialStuffingProfile)

    stealth = get_profile("stealth")
    assert isinstance(stealth, StealthCrawlerProfile)

    with pytest.raises(ValueError, match="Unknown profile type"):
        get_profile("unknown_bot")


def test_human_profile_plan():
    """HumanProfile should generate structured requests and rich behavioral telemetry."""
    profile = HumanProfile(dwell_scale=1.0)
    plan = profile.generate_plan("test-sess-001")
    
    assert len(plan) >= 6
    req_steps = [s for s in plan if s.step_type == "request"]
    telem_steps = [s for s in plan if s.step_type == "telemetry"]

    assert len(req_steps) >= 4
    assert len(telem_steps) >= 3

    # Check telemetry contains diverse event types
    event_types = set()
    for s in telem_steps:
        for ev in s.events:
            event_types.add(ev["event_type"])

    assert "page_view" in event_types
    assert "mouse_move" in event_types
    assert "scroll" in event_types
    assert "click" in event_types


def test_rapid_scraper_profile_plan():
    """RapidScraperProfile should generate fast consecutive requests with 0 telemetry."""
    profile = RapidScraperProfile(request_count=8)
    plan = profile.generate_plan("test-sess-002")

    assert len(plan) == 8
    assert all(s.step_type == "request" for s in plan)
    assert all(s.dwell_seconds <= 0.05 for s in plan)


def test_credential_stuffing_profile_plan():
    """CredentialStuffingProfile should target /login with 0 telemetry."""
    profile = CredentialStuffingProfile(attempts=5)
    plan = profile.generate_plan("test-sess-003")

    assert len(plan) == 5
    assert all(s.step_type == "request" for s in plan)
    assert all(s.path == "/login" for s in plan)


def test_stealth_crawler_profile_plan():
    """StealthCrawlerProfile should generate paced requests with 0 telemetry."""
    profile = StealthCrawlerProfile(page_count=6)
    plan = profile.generate_plan("test-sess-004")

    assert len(plan) == 6
    assert all(s.step_type == "request" for s in plan)
    assert all(s.dwell_seconds >= 0.20 for s in plan)


# ==============================================================================
# 2. In-Process Simulation Execution & Detection Tests
# ==============================================================================

def test_in_process_human_simulation(temp_db, trained_model_path, monkeypatch):
    """Human simulation should achieve LOW risk classification under hybrid scoring."""
    monkeypatch.setenv("BOTSHIELD_DB", temp_db)
    monkeypatch.setenv("BOTSHIELD_MODEL_PATH", trained_model_path)

    config = SimulationConfig(
        mode="in-process",
        profiles=[ProfileType.HUMAN],
        sessions_per_profile=1,
        evaluate_detection=True,
        db_path=temp_db,
        model_path=trained_model_path,
        dwell_scale=1.0,
    )
    runner = SimulationRunner(config)
    summary = runner.run()

    assert summary.total_sessions == 1
    assert summary.overall_detection_rate == 100.0

    sess = summary.sessions[0]
    assert sess.profile_type == "human"
    assert sess.evaluated is True
    assert sess.risk_level == "LOW"
    assert sess.risk_score is not None
    assert sess.risk_score < 35.0
    assert sess.request_count >= 3
    assert sess.telemetry_count >= 10
    assert sess.error is None


def test_in_process_scraper_simulation(temp_db, trained_model_path, monkeypatch):
    """Rapid scraper simulation should be detected as HIGH risk."""
    monkeypatch.setenv("BOTSHIELD_DB", temp_db)
    monkeypatch.setenv("BOTSHIELD_MODEL_PATH", trained_model_path)

    config = SimulationConfig(
        mode="in-process",
        profiles=[ProfileType.RAPID_SCRAPER],
        sessions_per_profile=1,
        evaluate_detection=True,
        db_path=temp_db,
        model_path=trained_model_path,
    )
    runner = SimulationRunner(config)
    summary = runner.run()

    assert summary.total_sessions == 1
    assert summary.overall_detection_rate == 100.0

    sess = summary.sessions[0]
    assert sess.profile_type == "scraper"
    assert sess.evaluated is True
    assert sess.risk_level == "HIGH"
    assert sess.risk_score is not None
    assert sess.risk_score >= 70.0
    assert sess.telemetry_count == 0
    assert len(sess.reasons) > 0


def test_in_process_credential_stuffing_simulation(temp_db, trained_model_path, monkeypatch):
    """Credential stuffing simulation targeting /login should be flagged as HIGH risk."""
    monkeypatch.setenv("BOTSHIELD_DB", temp_db)
    monkeypatch.setenv("BOTSHIELD_MODEL_PATH", trained_model_path)

    config = SimulationConfig(
        mode="in-process",
        profiles=[ProfileType.CREDENTIAL_STUFFER],
        sessions_per_profile=1,
        evaluate_detection=True,
        db_path=temp_db,
        model_path=trained_model_path,
    )
    runner = SimulationRunner(config)
    summary = runner.run()

    assert summary.total_sessions == 1
    assert summary.overall_detection_rate == 100.0

    sess = summary.sessions[0]
    assert sess.profile_type == "credential"
    assert sess.evaluated is True
    assert sess.risk_level == "HIGH"


def test_in_process_stealth_crawler_simulation(temp_db, trained_model_path, monkeypatch):
    """Stealth crawler simulation should be classified as MEDIUM or HIGH risk."""
    monkeypatch.setenv("BOTSHIELD_DB", temp_db)
    monkeypatch.setenv("BOTSHIELD_MODEL_PATH", trained_model_path)

    config = SimulationConfig(
        mode="in-process",
        profiles=[ProfileType.STEALTH_CRAWLER],
        sessions_per_profile=1,
        evaluate_detection=True,
        db_path=temp_db,
        model_path=trained_model_path,
    )
    runner = SimulationRunner(config)
    summary = runner.run()

    assert summary.total_sessions == 1
    assert summary.overall_detection_rate == 100.0

    sess = summary.sessions[0]
    assert sess.profile_type == "stealth"
    assert sess.evaluated is True
    assert sess.risk_level in ("MEDIUM", "HIGH")


# ==============================================================================
# 3. Aggregation & Formatting Tests
# ==============================================================================

def test_benchmark_aggregation_math():
    """ProfileBenchmarkResult math should correctly calculate distributions and rates."""
    sessions = [
        SessionSimulationResult(
            session_id="s1",
            profile_type="human",
            request_latencies_ms=[10.0, 20.0, 30.0],
            evaluated=True,
            risk_level="LOW",
            scoring_mode="HYBRID",
            scoring_duration_ms=5.0,
        ),
        SessionSimulationResult(
            session_id="s2",
            profile_type="human",
            request_latencies_ms=[15.0, 25.0],
            evaluated=True,
            risk_level="LOW",
            scoring_mode="HYBRID",
            scoring_duration_ms=7.0,
        ),
    ]

    config = SimulationConfig()
    runner = SimulationRunner(config)
    bm = runner._aggregate_profile(ProfileType.HUMAN, sessions)

    assert bm.total_sessions == 2
    assert bm.successful_sessions == 2
    assert bm.failed_sessions == 0
    assert bm.risk_distribution["LOW"] == 2
    assert bm.risk_distribution["HIGH"] == 0
    assert bm.expected_detection_rate == 100.0
    assert bm.mean_request_latency_ms == pytest.approx(20.0, abs=1e-2)
    assert bm.mean_scoring_latency_ms == pytest.approx(6.0, abs=1e-2)


def test_table_formatter():
    """_format_table should generate an ANSI/text report with correct headers and rows."""
    summary = SimulationBenchmarkSummary(
        mode="in-process",
        target_url=None,
        total_sessions=2,
        total_duration_seconds=1.23,
        overall_detection_rate=100.0,
        profiles={
            "human": ProfileBenchmarkResult(
                profile_type="human",
                total_sessions=1,
                risk_distribution={"LOW": 1, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0},
                mean_request_latency_ms=25.5,
                expected_detection_rate=100.0,
            ),
        },
    )
    table = _format_table(summary)
    assert "BOTSHIELD RED-TEAM SIMULATION & DETECTION BENCHMARK" in table
    assert "human" in table
    assert "100.0%" in table


# ==============================================================================
# 4. CLI Execution Tests
# ==============================================================================

def test_simulation_cli_json_output(temp_db, trained_model_path, monkeypatch, capsys):
    """Running CLI with --json should output a valid, parseable JSON dictionary."""
    monkeypatch.setenv("BOTSHIELD_DB", temp_db)
    monkeypatch.setenv("BOTSHIELD_MODEL_PATH", trained_model_path)

    test_args = [
        "runner.py",
        "--mode", "in-process",
        "--profiles", "scraper",
        "--sessions", "1",
        "--json",
        "--db", temp_db,
        "--model", trained_model_path,
    ]

    with patch("sys.argv", test_args):
        main()

    captured = capsys.readouterr()
    # Output may contain DB init log lines before the JSON
    json_start = captured.out.find("{")
    assert json_start != -1
    json_text = captured.out[json_start:]
    data = json.loads(json_text)

    assert data["mode"] == "in-process"
    assert data["total_sessions"] == 1
    assert "scraper" in data["profiles"]
    assert data["profiles"]["scraper"]["expected_detection_rate"] == 100.0
