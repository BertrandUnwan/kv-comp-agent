

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

from kv_comp_agent.report import build_rule_based_memo
from kv_comp_agent.schema import SubjectProperty, ValuationEstimate


SYSTEM_PROMPT = """
You are an AI assistant supporting a real estate lending analyst.
Your job is to explain comparable-property valuation results clearly and conservatively.
Do not invent facts. Do not override the deterministic valuation estimate.
Use the provided subject property, comparable sales, confidence, and risk flags only.
Write in a concise underwriting-style tone.
""".strip()


def generate_valuation_memo(
    subject: SubjectProperty,
    top_comps: pd.DataFrame,
    valuation: ValuationEstimate,
    search_message: str | None = None,
    use_llm: bool = True,
) -> str:
    """
    Generate a valuation memo.

    The deterministic rule-based memo is always available. If an OpenAI API key
    exists, the function can ask an LLM to rewrite the same structured facts into
    a more natural underwriting memo. If the LLM fails, the app falls back safely.
    """
    fallback_memo = build_rule_based_memo(subject, top_comps, valuation, search_message)

    if not use_llm:
        return fallback_memo

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return fallback_memo

    try:
        return _generate_openai_memo(subject, top_comps, valuation, search_message, fallback_memo)
    except Exception:
        return fallback_memo


def _generate_openai_memo(
    subject: SubjectProperty,
    top_comps: pd.DataFrame,
    valuation: ValuationEstimate,
    search_message: str | None,
    fallback_memo: str,
) -> str:
    """Call OpenAI only when configured. Kept isolated for graceful fallback."""
    from openai import OpenAI

    client = OpenAI()

    prompt = _build_llm_prompt(subject, top_comps, valuation, search_message, fallback_memo)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    content = response.choices[0].message.content
    if not content:
        return fallback_memo

    return content.strip()


def _build_llm_prompt(
    subject: SubjectProperty,
    top_comps: pd.DataFrame,
    valuation: ValuationEstimate,
    search_message: str | None,
    fallback_memo: str,
) -> str:
    """Build a compact prompt from structured facts."""
    comp_columns = [
        "property_id",
        "city",
        "neighborhood",
        "property_type",
        "living_area_sqft",
        "bedrooms",
        "bathrooms",
        "year_built",
        "sale_date",
        "sale_price",
        "price_per_sqft",
        "total_score",
        "reason_selected",
    ]
    available_columns = [col for col in comp_columns if col in top_comps.columns]
    comps_text = top_comps.head(5)[available_columns].to_markdown(index=False)

    return f"""
Rewrite the following deterministic comp-analysis result as a concise underwriting-style memo.
Keep the same valuation numbers and risk flags. Do not add unsupported claims.

Subject property:
{subject.model_dump()}

Search note:
{search_message or "No search note provided."}

Valuation:
{valuation.model_dump()}

Top comparable sales:
{comps_text}

Fallback memo for reference:
{fallback_memo}
""".strip()