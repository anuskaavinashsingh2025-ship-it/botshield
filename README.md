# BotShield

### AI-Based Website Bot Detection Using Behavioral Automata

## About

BotShield is a cybersecurity research project that aims to detect automated
bots on websites by analysing user behaviour with finite-state automata and
machine-learning classifiers.

This repository contains a demo e-commerce website (the "BotShield Store") that
will be progressively instrumented to capture behavioural signals and classify
visitors as human or bot.

## Current Status

**Step 1 – Project Foundation** is complete:

- Flask application serving five pages (Home, Products, Product Detail, Login, About).
- Static product catalogue (no database yet).
- Clean, responsive front-end with HTML, CSS, and vanilla JavaScript.
- Basic test suite verifying all routes.

Future steps will add behavioural tracking, automata-based modelling, ML
classification, a dashboard, and bot simulators.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd botshield

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

The website will be available at **http://localhost:5000**.

### URLs

| Page           | URL                  |
|----------------|----------------------|
| Home           | `/`                  |
| Products       | `/products`          |
| Product Detail | `/product/<id>`      |
| Login          | `/login`             |
| About          | `/about`             |

## Running Tests

```bash
pytest
```

## Tech Stack

- Python 3
- Flask
- HTML / CSS / vanilla JavaScript
- SQLite *(planned)*

## License

This project is for educational and research purposes.
