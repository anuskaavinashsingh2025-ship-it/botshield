"""BotShield Red-Team Bot Traffic Simulator and Benchmarking Package.

Provides behavioral traffic profiles, execution runners (in-process Flask
and live HTTP), and detection performance benchmarking tools.
"""

from backend.simulation.profiles import (
    BaseBehavioralProfile,
    CredentialStuffingProfile,
    HumanProfile,
    ProfileType,
    RapidScraperProfile,
    StealthCrawlerProfile,
    get_profile,
)

__all__ = [
    "BaseBehavioralProfile",
    "CredentialStuffingProfile",
    "HumanProfile",
    "ProfileBenchmarkResult",
    "ProfileType",
    "RapidScraperProfile",
    "SessionSimulationResult",
    "SimulationBenchmarkSummary",
    "SimulationConfig",
    "SimulationRunner",
    "StealthCrawlerProfile",
    "get_profile",
    "run_simulation",
]


def __getattr__(name: str):
    if name in (
        "ProfileBenchmarkResult",
        "SessionSimulationResult",
        "SimulationBenchmarkSummary",
        "SimulationConfig",
        "SimulationRunner",
        "run_simulation",
    ):
        import backend.simulation.runner as runner_mod
        return getattr(runner_mod, name)
    raise AttributeError(f"module 'backend.simulation' has no attribute '{name}'")

