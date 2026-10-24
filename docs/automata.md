# Behavioral Finite State Automaton

This document outlines the theoretical component of the BotShield FSM, designed to model the sequence of visitor behavior and compute a deterministic automata risk score.

## 1. Formal FSM Definition
The automaton is defined as $M = (Q, \Sigma, \delta, q_0, F)$:
- $Q$ (States): `START`, `NORMAL_BROWSING`, `INTERACTIVE_BROWSING`, `RAPID_REQUESTS`, `REPEATED_REQUESTS`, `SUSPICIOUS_BEHAVIOR`, `HUMAN_LIKE`, `BOT_DETECTED`
- $\Sigma$ (Alphabet): `PAGE_VIEW`, `CLICK`, `SCROLL`, `MOUSE_ACTIVITY`, `KEYBOARD_ACTIVITY`, `WAIT`, `REQUEST`, `RAPID_REQUEST`, `REPEATED_REQUEST`, `LOGIN_ATTEMPT`, `FAILED_REQUEST`
- $\delta$ (Transition Function): Documented in the transition logic below.
- $q_0$ (Initial State): `START`
- $F$ (Final/Terminal States): `BOT_DETECTED`

## 2. Transition Logic
The transition logic ($\delta$) maintains per-session context, updating internal counters and decaying/recovering based on genuine interactions.

| Current State       | Input Symbol               | Next State             | Reason                                 |
|---------------------|----------------------------|------------------------|----------------------------------------|
| `START`             | `PAGE_VIEW` / `REQUEST`    | `NORMAL_BROWSING`      | Initial page visit                     |
| `START`             | `CLICK` / `SCROLL`         | `INTERACTIVE_BROWSING` | Immediate interaction                  |
| `NORMAL_BROWSING`   | `CLICK` / `SCROLL`         | `INTERACTIVE_BROWSING` | User interaction                       |
| `NORMAL_BROWSING`   | `RAPID_REQUEST`            | `RAPID_REQUESTS`       | Abnormally fast requests               |
| `INTERACTIVE_BROWSING`| `PAGE_VIEW` (sustained)    | `HUMAN_LIKE`           | Normal browsing sequence established   |
| `RAPID_REQUESTS`    | `REPEATED_REQUEST`         | `REPEATED_REQUESTS`    | Repeated endpoint after rapid requests |
| `REPEATED_REQUESTS` | `FAILED_REQUEST` / burst   | `SUSPICIOUS_BEHAVIOR`  | Multiple suspicious signals accumulate |
| `SUSPICIOUS_BEHAVIOR`| strong bot pattern         | `BOT_DETECTED`         | High-confidence behavioral pattern     |

## 3. Rolling-Window Logic
The event interpreter uses a rolling window (default: 30 seconds) to evaluate HTTP requests.
- If $N \ge 3$ requests occur within $2.0$ seconds, it yields `RAPID_REQUEST`.
- If the exact same endpoint is requested $N \ge 3$ times in $10.0$ seconds, it yields `REPEATED_REQUEST`.

## 4. Recovery Mechanism
A human could legitimately trigger an anomaly (e.g., clicking twice quickly). If a session enters `RAPID_REQUESTS`, but then begins demonstrating genuine interaction (like `SCROLL` or `CLICK`), the FSM steps back down to `INTERACTIVE_BROWSING`. Scoring formulas also reduce the final risk score when human interactions are detected.

## 5. Automata Scoring Formula
The score $S$ is clamped between $0.0$ and $1.0$:
- **Base Score**: Determined by the state (e.g., `NORMAL_BROWSING` = 0.1, `SUSPICIOUS_BEHAVIOR` = 0.8, `BOT_DETECTED` = 0.95).
- **Modifiers**:
  - `+0.05` for each rapid request detected.
  - `+0.05` for each repeated endpoint detected.
  - `+0.05` for each failed request.
  - `-0.02` for each interaction event (`CLICK`, `SCROLL`), up to `-0.30`.

## 6. Example Human Sequence
```text
PAGE_VIEW
WAIT
SCROLL
CLICK
WAIT
MOUSE_ACTIVITY
PAGE_VIEW
SCROLL
CLICK
```
**Expected Outcome**: `HUMAN_LIKE` state with an automata score near $0.0$.

## 7. Example Bot Sequence
```text
PAGE_VIEW
RAPID_REQUEST
RAPID_REQUEST
REPEATED_REQUEST
FAILED_REQUEST
RAPID_REQUEST
REPEATED_REQUEST
```
**Expected Outcome**: Progressive transitions `RAPID_REQUESTS` → `REPEATED_REQUESTS` → `SUSPICIOUS_BEHAVIOR` → `BOT_DETECTED`, with an automata score near $1.0$.

## 8. Live Automata Monitoring
The BotShield system includes a live monitoring endpoint (`GET /api/automata/status`) that exposes the current FSM state for a given session.
- **How it works:** The backend maintains an in-memory `SessionContext` for active sessions. Every time behavioral telemetry (clicks, scrolls) or HTTP requests hit the server, they are interpreted into FSM symbols and the state is updated. The status endpoint simply returns the current state, score, and reasons from this context.
- **Frontend Display:** A floating "BotShield Monitor" panel on the bottom right of the UI automatically polls this endpoint every 1.5 seconds.
- **Demonstration:** This allows for real-time demonstrations of the FSM's detection capabilities (e.g. generating rapid page loads to instantly watch the score spike, or producing normal mouse activity to watch it decay).
