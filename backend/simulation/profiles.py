"""Behavioral traffic profiles for BotShield red-team simulation.

Defines realistic human behavior patterns and automated adversarial bot
patterns (rapid scrapers, credential stuffers, stealth crawlers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
import time
from typing import Any, Dict, List, Optional


class ProfileType(str, Enum):
    """Behavioral profile types for traffic simulation."""
    HUMAN = "human"
    RAPID_SCRAPER = "scraper"
    CREDENTIAL_STUFFER = "credential"
    STEALTH_CRAWLER = "stealth"


@dataclass
class SimulationStep:
    """A discrete step in a session simulation plan."""
    step_type: str  # "request" | "telemetry"
    path: str = "/"
    method: str = "GET"
    data: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    dwell_seconds: float = 0.0


class BaseBehavioralProfile:
    """Base class for session behavioral traffic generators."""

    profile_type: ProfileType = ProfileType.HUMAN
    name: str = "Base Profile"
    description: str = "Abstract base profile"

    def generate_plan(self, session_id: str) -> List[SimulationStep]:
        """Generate ordered simulation steps for this profile."""
        raise NotImplementedError


class HumanProfile(BaseBehavioralProfile):
    """Simulates a legitimate human shopper navigating the storefront.
    
    Characteristics:
    - Moderate, realistic inter-request dwell times (>200ms).
    - Rich client-side behavioral telemetry (page views, mouse movements,
      scroll trajectories, element clicks, page exits).
    - Natural path progression (Home -> Products -> Detail -> About -> Login).
    - Expected classification: LOW risk.
    """

    profile_type = ProfileType.HUMAN
    name = "Human Shopper"
    description = "Legitimate human user with natural dwell times, clicks, mouse tracking, and scrolls."

    def __init__(self, dwell_scale: float = 1.0):
        self.dwell_scale = max(0.1, dwell_scale)

    def generate_plan(self, session_id: str) -> List[SimulationStep]:
        steps: List[SimulationStep] = []
        now_ms = int(time.time() * 1000)

        # 1. Home page request + immediate page_view and mouse movements
        steps.append(SimulationStep(
            step_type="request",
            path="/",
            method="GET",
            dwell_seconds=0.25 * self.dwell_scale,
        ))

        home_events = [
            {
                "event_type": "page_view",
                "page": "/",
                "timestamp": now_ms,
                "data": {"title": "BotShield Store", "referrer": ""},
            },
            {
                "event_type": "mouse_move",
                "page": "/",
                "timestamp": now_ms + 40,
                "data": {"x": 210, "y": 150},
            },
            {
                "event_type": "mouse_move",
                "page": "/",
                "timestamp": now_ms + 80,
                "data": {"x": 340, "y": 280},
            },
            {
                "event_type": "scroll",
                "page": "/",
                "timestamp": now_ms + 120,
                "data": {"scroll_y": 300, "max_scroll_y": 800},
            },
            {
                "event_type": "click",
                "page": "/",
                "timestamp": now_ms + 180,
                "data": {"tag": "A", "id": "nav-products", "x": 350, "y": 45},
            },
            {
                "event_type": "page_exit",
                "page": "/",
                "timestamp": now_ms + 220,
                "data": {"dwell_time_ms": 220},
            },
        ]
        steps.append(SimulationStep(step_type="telemetry", events=home_events))

        # 2. Browse Products catalogue
        now_ms += int(250 * self.dwell_scale)
        steps.append(SimulationStep(
            step_type="request",
            path="/products",
            method="GET",
            dwell_seconds=0.25 * self.dwell_scale,
        ))

        products_events = [
            {
                "event_type": "page_view",
                "page": "/products",
                "timestamp": now_ms,
                "data": {"title": "All Products", "referrer": "/"},
            },
            {
                "event_type": "mouse_move",
                "page": "/products",
                "timestamp": now_ms + 30,
                "data": {"x": 400, "y": 320},
            },
            {
                "event_type": "scroll",
                "page": "/products",
                "timestamp": now_ms + 90,
                "data": {"scroll_y": 600, "max_scroll_y": 1500},
            },
            {
                "event_type": "click",
                "page": "/products",
                "timestamp": now_ms + 160,
                "data": {"tag": "A", "id": "product-card-1", "x": 420, "y": 550},
            },
            {
                "event_type": "page_exit",
                "page": "/products",
                "timestamp": now_ms + 210,
                "data": {"dwell_time_ms": 210},
            },
        ]
        steps.append(SimulationStep(step_type="telemetry", events=products_events))

        # 3. Product Detail page
        now_ms += int(250 * self.dwell_scale)
        steps.append(SimulationStep(
            step_type="request",
            path="/product/1",
            method="GET",
            dwell_seconds=0.25 * self.dwell_scale,
        ))

        detail_events = [
            {
                "event_type": "page_view",
                "page": "/product/1",
                "timestamp": now_ms,
                "data": {"title": "Product Detail", "referrer": "/products"},
            },
            {
                "event_type": "mouse_move",
                "page": "/product/1",
                "timestamp": now_ms + 50,
                "data": {"x": 510, "y": 420},
            },
            {
                "event_type": "click",
                "page": "/product/1",
                "timestamp": now_ms + 140,
                "data": {"tag": "BUTTON", "id": "btn-add-cart", "x": 530, "y": 440},
            },
        ]
        steps.append(SimulationStep(step_type="telemetry", events=detail_events))

        # 4. Visit About page
        now_ms += int(250 * self.dwell_scale)
        steps.append(SimulationStep(
            step_type="request",
            path="/about",
            method="GET",
            dwell_seconds=0.25 * self.dwell_scale,
        ))

        about_events = [
            {
                "event_type": "page_view",
                "page": "/about",
                "timestamp": now_ms,
                "data": {"title": "About BotShield", "referrer": "/product/1"},
            },
            {
                "event_type": "scroll",
                "page": "/about",
                "timestamp": now_ms + 70,
                "data": {"scroll_y": 250, "max_scroll_y": 900},
            },
        ]
        steps.append(SimulationStep(step_type="telemetry", events=about_events))

        return steps


class RapidScraperProfile(BaseBehavioralProfile):
    """Simulates an automated high-speed catalog scraper (e.g. Scrapy, curl).
    
    Characteristics:
    - Ultra-rapid traversal (<50ms inter-request deltas).
    - Sweeps across catalog items and products.
    - Zero browser telemetry events emitted (no JavaScript engine).
    - Expected classification: HIGH risk (triggers ANOMALY_RAPID_TRAVERSAL and
      ANOMALY_HIGH_REQ_LOW_TELEMETRY).
    """

    profile_type = ProfileType.RAPID_SCRAPER
    name = "Rapid Catalog Scraper"
    description = "High-speed automated scraper issuing rapid consecutive requests with 0 telemetry."

    def __init__(self, request_count: int = 7, dwell_scale: float = 1.0, **kwargs: Any):
        self.request_count = max(4, request_count)
        self.dwell_scale = dwell_scale

    def generate_plan(self, session_id: str) -> List[SimulationStep]:
        paths = [
            "/",
            "/products",
            "/product/1",
            "/product/2",
            "/product/3",
            "/product/4",
            "/product/5",
        ]
        steps: List[SimulationStep] = []
        for i in range(self.request_count):
            p = paths[i % len(paths)]
            steps.append(SimulationStep(
                step_type="request",
                path=p,
                method="GET",
                dwell_seconds=0.01,  # 10ms - triggers rapid traversal
            ))
        return steps


class CredentialStuffingProfile(BaseBehavioralProfile):
    """Simulates an automated credential stuffing attack targeting /login.
    
    Characteristics:
    - Repeated GET / POST requests directly against /login.
    - Zero client-side behavioral telemetry (no genuine user interactions).
    - Rapid submission bursts.
    - Expected classification: HIGH risk.
    """

    profile_type = ProfileType.CREDENTIAL_STUFFER
    name = "Credential Stuffing Bot"
    description = "Targeted automated requests directly against sensitive /login route with 0 telemetry."

    def __init__(self, attempts: int = 5, dwell_scale: float = 1.0, **kwargs: Any):
        self.attempts = max(3, attempts)
        self.dwell_scale = dwell_scale

    def generate_plan(self, session_id: str) -> List[SimulationStep]:
        steps: List[SimulationStep] = []
        for i in range(self.attempts):
            steps.append(SimulationStep(
                step_type="request",
                path="/login",
                method="GET",
                dwell_seconds=0.02,
            ))
        return steps


class StealthCrawlerProfile(BaseBehavioralProfile):
    """Simulates a low-and-slow crawler attempting to evade rapid-traversal limits.
    
    Characteristics:
    - Moderate request pacing (>200ms) to avoid naive rate limiters.
    - Emits zero or minimal telemetry events, revealing absence of a human
      in front of a browser.
    - Triggers ANOMALY_HIGH_REQ_LOW_TELEMETRY and ML Isolation Forest anomaly score.
    - Expected classification: MEDIUM to HIGH risk.
    """

    profile_type = ProfileType.STEALTH_CRAWLER
    name = "Stealth Low-and-Slow Crawler"
    description = "Paced requests avoiding rapid traversal, but with 0 telemetry revealing automated nature."

    def __init__(self, page_count: int = 5, dwell_scale: float = 1.0, **kwargs: Any):
        self.page_count = max(4, page_count)
        self.dwell_scale = dwell_scale

    def generate_plan(self, session_id: str) -> List[SimulationStep]:
        paths = ["/", "/products", "/product/1", "/product/2", "/about", "/products"]
        steps: List[SimulationStep] = []
        for i in range(self.page_count):
            p = paths[i % len(paths)]
            steps.append(SimulationStep(
                step_type="request",
                path=p,
                method="GET",
                dwell_seconds=0.22,  # >200ms avoids rapid traversal, but 0 telemetry flags it
            ))
        return steps


_PROFILE_REGISTRY = {
    ProfileType.HUMAN: HumanProfile,
    ProfileType.RAPID_SCRAPER: RapidScraperProfile,
    ProfileType.CREDENTIAL_STUFFER: CredentialStuffingProfile,
    ProfileType.STEALTH_CRAWLER: StealthCrawlerProfile,
}


def get_profile(profile_type: ProfileType | str, **kwargs: Any) -> BaseBehavioralProfile:
    """Instantiate a behavioral profile by enum or string key."""
    try:
        key = ProfileType(profile_type) if isinstance(profile_type, str) else profile_type
    except ValueError:
        raise ValueError(f"Unknown profile type: {profile_type}. Valid: {[p.value for p in ProfileType]}")
    cls = _PROFILE_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown profile type: {profile_type}. Valid: {[p.value for p in ProfileType]}")
    return cls(**kwargs)
