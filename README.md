# KV Comp Agent

AI-assisted comparable-property analysis prototype for Alberta residential lending.

This project was built for the **KV Capital AI Engineer Hackathon**. The challenge was to scope and ship a practical AI agent that helps with residential real estate comp analysis: finding comparable recent sales, ranking them, estimating value, and explaining the reasoning clearly enough to support an underwriting workflow.

## Problem understanding

KV Capital finances home builders in Alberta. A key part of underwriting a loan is understanding what the underlying property is worth. The bottleneck is not just calculating a number; it is finding defensible comparable sales and explaining why those comps support a value conclusion.

For this prototype, I scoped the problem down to:

> Given a subject residential property in Alberta, retrieve relevant comparable sold properties, rank them using transparent scoring, estimate a value range, and generate a concise underwriting-style memo with confidence and risk flags.

The solution is intentionally focused on residential properties rather than trying to be a broad general-purpose real estate agent.

## What the app does

The app takes a subject property as input, including:

- City and neighborhood
- Property type
- Bedrooms and bathrooms
- Living area
- Lot size
- Year built
- Garage spaces
- Condition and feature flags

It then returns:

- A base valuation estimate
- A low-to-high valuation range
- Confidence level
- Ranked comparable sales
- Score breakdown for each comp
- Risk flags / review notes
- A valuation memo written for analyst review

## Demo scenarios

The Streamlit sidebar includes built-in demo scenarios so reviewers can test the workflow quickly:

1. **Strong match: Edmonton detached home**
2. **Sparse market: Leduc townhouse**
3. **Premium case: Calgary detached home**
4. **Lower confidence: Fort McMurray condo**

These scenarios are designed to show how the agent behaves under different comp-quality conditions.

## Architecture

The project is structured as a small production-style Python application rather than a notebook-only prototype.

```text
Streamlit UI
    ↓
Validated subject property input
    ↓
Candidate comparable search
    ↓
Deterministic scoring engine
    ↓
Weighted valuation engine
    ↓
Confidence + risk flags
    ↓
Rule-based or optional LLM memo
```

### Key design choice

The LLM does **not** choose comps or calculate the valuation.

Comparable selection, scoring, confidence, and valuation are deterministic so the workflow remains repeatable and auditable. The optional LLM layer is used only to rewrite the structured results into a more natural underwriting-style memo.

If no `OPENAI_API_KEY` is configured, or if the LLM call fails, the app safely falls back to a deterministic rule-based memo.

## Repository structure

```text
kv-comp-agent/
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
│
├── data/
│   └── alberta_residential_sales.csv
│
├── scripts/
│   └── generate_synthetic_data.py
│
├── src/
│   └── kv_comp_agent/
│       ├── schema.py
│       ├── config.py
│       ├── data_loader.py
│       ├── filters.py
│       ├── scoring.py
│       ├── valuation.py
│       ├── report.py
│       └── agent.py
│
├── tests/
│   ├── test_validation.py
│   ├── test_scoring.py
│   └── test_valuation.py
│
└── docs/
    └── approach.md
```

## Dataset

The brief allowed the use of public or synthetic data. I used a synthetic Alberta residential sales dataset because Canadian sold-price data can be restricted or inconsistent, and the goal of this challenge is to demonstrate the workflow and engineering judgment.

The generated dataset contains **1,200 synthetic residential sales** across Alberta markets, including:

- Edmonton
- Calgary
- St. Albert
- Sherwood Park
- Leduc
- Airdrie
- Red Deer
- Fort McMurray

Each record includes property attributes such as:

- Property type
- Bedrooms and bathrooms
- Living area
- Lot size
- Year built
- Sale date
- Sale price
- Condition
- Garage spaces
- Feature flags such as renovation, basement, transit proximity, and backing onto park

The dataset includes intentional edge cases such as missing lot size and year-built values so the agent can demonstrate graceful handling of imperfect data.

To regenerate the dataset:

```bash
export PYTHONPATH=src
python scripts/generate_synthetic_data.py
```

## How comp search works

The candidate search uses staged fallback logic. It starts with the most relevant possible comps and gradually broadens the search only when needed.

Search stages:

1. Same neighborhood, same property type, recent sales
2. Same city, same property type
3. Same city, wider property/date range
4. Nearby Alberta markets
5. Broad Alberta fallback

This prevents the app from failing when a local market is sparse while still making it clear when the search had to broaden.

Example search message:

> Neighborhood-level comps were limited, so the search expanded to same-city, same-property-type sales.

## Scoring methodology

Each comparable property receives a score from 0 to 100.

The score is based on:

| Factor | Weight |
|---|---:|
| Location similarity | 30% |
| Property type match | 20% |
| Living area similarity | 15% |
| Bedroom/bathroom similarity | 10% |
| Year built similarity | 10% |
| Sale recency | 10% |
| Feature similarity | 5% |

The app also exposes the score breakdown in the UI so the reviewer can see why a comp was ranked highly or poorly.

## Valuation methodology

The valuation uses the top-ranked comparable sales and calculates a weighted price-per-square-foot estimate.

Higher-scoring comps receive more influence. The app then returns a valuation range instead of a single false-precision number.

```text
weighted_ppsf = weighted average of top comp price_per_sqft
base_estimate = weighted_ppsf × subject_living_area
```

The estimate range expands or contracts based on confidence.

## Confidence and risk flags

The confidence score considers:

- Average comp score
- Number of available comps
- Sale recency
- Property-type match

The app produces a confidence label:

- High
- Medium
- Low

It also generates review notes such as:

- Fewer than five comparable sales were available
- Selected comps differ from the subject property type
- Selected comps are outside the subject's immediate location
- Subject year built or lot size is missing
- No major comp-quality risks detected

This is meant to support human review rather than replace formal appraisal or underwriting judgment.

## LLM usage

The app includes an optional LLM memo layer.

If an `OPENAI_API_KEY` is available and the sidebar option is enabled, the app sends the structured valuation result to OpenAI and asks it to rewrite the memo in a concise underwriting style.

The LLM is constrained to the provided facts:

- Subject property
- Top comparable sales
- Valuation range
- Confidence
- Risk flags
- Search note

The LLM does not calculate value, select comps, or override the deterministic output.

If no API key is configured, the app uses the rule-based memo and still works end-to-end.

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd kv-comp-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Optional: configure OpenAI

The app works without an OpenAI key. To enable the optional LLM memo, create a `.env` file:

```bash
cp .env.example .env
```

Then add your key:

```text
OPENAI_API_KEY=your_key_here
```

Do not commit `.env` to GitHub.

### 5. Run the app

```bash
export PYTHONPATH=src
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Running tests

```bash
export PYTHONPATH=src
pytest
```

Current test status:

```text
8 passed
```

The tests cover:

- Subject property validation
- Invalid input rejection
- Property-type scoring
- Living-area scoring
- Sale-recency scoring
- Valuation range generation
- Empty-comp fallback behavior

## Tradeoffs

### Synthetic data instead of public data

I used synthetic data because the brief allowed it and because public residential sold-price data can be difficult to access cleanly. The schema is designed so real transaction data could be dropped in later with minimal changes.

### Deterministic scoring instead of black-box valuation

I intentionally kept comp selection and valuation deterministic. In a lending context, repeatability and auditability matter. The LLM is used for explanation, not valuation math.

### Streamlit instead of a full web stack

A full React/FastAPI/database stack would be unnecessary for this scope. Streamlit made it possible to ship a working, reviewable prototype quickly while keeping the business logic modular and testable.

### Valuation range instead of exact price

The app returns a range because comp quality varies and underwriting workflows need uncertainty surfaced. A single exact number would imply more precision than the data supports.

## What I would build next

If this were moving beyond prototype stage, I would add:

- Integration with real internal transaction or MLS-style data
- More precise geospatial distance scoring
- Analyst feedback loop for tuning scoring weights
- Better condition and renovation adjustment logic
- Market trend/time adjustment for older sales
- Support for commercial borrower workflows
- Exportable PDF valuation memo
- User authentication and saved analyses
- Monitoring for LLM failures and output quality

## Problem-owner call note

The brief recommended an optional call with Sam, the AI problem-owner agent. I attempted the call, but the line repeatedly returned:

> “Sorry, we’re having trouble connecting your call right now. Please try again in a little while.”

Because of that, I proceeded using the written brief and scoped the solution around a focused Alberta residential comp-analysis workflow.

## Demo video

Demo video link: [Watch the demo](https://www.loom.com/share/2b11a2003b654ecc989a54ecc178c10a)

## Submission

Submitted for the KV Capital AI Engineer Hackathon.
