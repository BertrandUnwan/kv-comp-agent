from __future__ import annotations

import pandas as pd

from kv_comp_agent.schema import SubjectProperty, ValuationEstimate


def format_currency(value: int | float) -> str:
    """Format a numeric value as Canadian dollars for Markdown display."""
    return f"\\${value:,.0f}"


def format_percent(value: float) -> str:
    """Format a decimal or score-like value as a percentage."""
    return f"{value:.0%}"


def build_rule_based_memo(
    subject: SubjectProperty,
    top_comps: pd.DataFrame,
    valuation: ValuationEstimate,
    search_message: str | None = None,
) -> str:
    """
    Build a deterministic underwriting-style memo.

    This is used as the default memo and as a fallback if an LLM is unavailable.
    """
    subject_summary = (
        f"{subject.city}"
        + (f" / {subject.neighborhood}" if subject.neighborhood else "")
        + f" {subject.property_type.value.lower()} with "
        f"{subject.bedrooms} bedrooms, {subject.bathrooms:g} bathrooms, "
        f"and {subject.living_area_sqft:,} sqft of living area"
    )

    memo_parts = [
        f"**Subject property:** {subject_summary}.",
        "",
        (
            f"**Estimated value:** {format_currency(valuation.base_estimate)} "
            f"with an indicated range of {format_currency(valuation.low_estimate)} "
            f"to {format_currency(valuation.high_estimate)}."
        ),
        "",
        f"**Confidence:** {valuation.confidence} ({valuation.confidence_score:.2f}/1.00).",
        "",
    ]

    if search_message:
        memo_parts.extend(
            [
                f"**Comp search note:** {search_message}",
                "",
            ]
        )

    memo_parts.extend(
        [
            "**Methodology:**",
            valuation.methodology,
            "",
            "**Comparable support:**",
        ]
    )

    if top_comps.empty:
        memo_parts.append("No comparable sales were available for this analysis.")
    else:
        for index, comp in top_comps.head(5).iterrows():
            sale_price = format_currency(comp["sale_price"])
            ppsf = format_currency(comp["price_per_sqft"])
            score = comp.get("total_score", 0)
            reason = comp.get("reason_selected", "Selected by the comp scoring engine.")

            memo_parts.append(
                (
                    f"- **Comp {index + 1}: {comp['property_id']}** — "
                    f"{comp['neighborhood']}, {comp['city']}; "
                    f"{comp['property_type']}; {comp['living_area_sqft']:,} sqft; "
                    f"sold for {sale_price} ({ppsf}/sqft); "
                    f"score {score:.1f}/100. {reason}"
                )
            )

    memo_parts.extend(
        [
            "",
            "**Risk flags / review notes:**",
        ]
    )

    for flag in valuation.risk_flags:
        memo_parts.append(f"- {flag}")

    memo_parts.extend(
        [
            "",
            "**Human-in-the-loop note:**",
            (
                "This prototype is designed to support analyst review, not replace formal appraisal "
                "or underwriting judgment. The estimate should be reviewed against source documents, "
                "market context, and any borrower-specific information before being used in a credit decision."
            ),
        ]
    )

    return "\n".join(memo_parts)