from __future__ import annotations

import pandas as pd
import streamlit as st

from kv_comp_agent.agent import generate_valuation_memo
from kv_comp_agent.data_loader import DataLoadError, load_sales_data
from kv_comp_agent.filters import find_candidate_comps
from kv_comp_agent.schema import PropertyCondition, PropertyType, SubjectProperty
from kv_comp_agent.scoring import score_comps
from kv_comp_agent.valuation import estimate_value


st.set_page_config(
    page_title="KV Comp Agent",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def cached_load_sales_data() -> pd.DataFrame:
    return load_sales_data()


def format_currency(value: int | float) -> str:
    return f"${value:,.0f}"


DEMO_SCENARIOS = {
    "Strong match: Edmonton detached home": {
        "city": "Edmonton",
        "neighborhood": "Glenora",
        "property_type": "Detached",
        "bedrooms": 4,
        "bathrooms": 3.0,
        "living_area_sqft": 2100,
        "lot_size_sqft": 5500,
        "year_built": 2015,
        "garage_spaces": 2,
        "condition": "Average",
        "finished_basement": False,
        "renovated": False,
        "near_transit": False,
        "backs_onto_park": False,
    },
    "Sparse market: Leduc townhouse": {
        "city": "Leduc",
        "neighborhood": "Meadowview",
        "property_type": "Townhouse",
        "bedrooms": 3,
        "bathrooms": 2.5,
        "living_area_sqft": 1450,
        "lot_size_sqft": 1800,
        "year_built": 2020,
        "garage_spaces": 1,
        "condition": "Good",
        "finished_basement": False,
        "renovated": True,
        "near_transit": False,
        "backs_onto_park": False,
    },
    "Premium case: Calgary detached home": {
        "city": "Calgary",
        "neighborhood": "Aspen Woods",
        "property_type": "Detached",
        "bedrooms": 5,
        "bathrooms": 4.0,
        "living_area_sqft": 2850,
        "lot_size_sqft": 6500,
        "year_built": 2018,
        "garage_spaces": 3,
        "condition": "Excellent",
        "finished_basement": True,
        "renovated": True,
        "near_transit": False,
        "backs_onto_park": True,
    },
    "Lower confidence: Fort McMurray condo": {
        "city": "Fort McMurray",
        "neighborhood": "Abasand",
        "property_type": "Condo",
        "bedrooms": 2,
        "bathrooms": 2.0,
        "living_area_sqft": 950,
        "lot_size_sqft": 0,
        "year_built": 2012,
        "garage_spaces": 1,
        "condition": "Average",
        "finished_basement": False,
        "renovated": False,
        "near_transit": True,
        "backs_onto_park": False,
    },
}


def run_analysis(subject: SubjectProperty, use_llm: bool) -> tuple:
    sales_data = cached_load_sales_data()
    search = find_candidate_comps(subject, sales_data)
    scored = score_comps(subject, search.candidates)
    valuation = estimate_value(subject, scored)
    memo = generate_valuation_memo(
        subject=subject,
        top_comps=scored.head(5),
        valuation=valuation,
        search_message=search.message,
        use_llm=use_llm,
    )
    return search, scored, valuation, memo


st.title("KV Comp Agent")
st.caption(
    "AI-assisted comparable-property analysis prototype for Alberta residential lending."
)

with st.sidebar:
    st.header("Subject property")

    scenario_name = st.selectbox(
        "Demo scenario",
        list(DEMO_SCENARIOS.keys()),
        index=0,
    )
    scenario = DEMO_SCENARIOS[scenario_name]

    city_options = [
        "Edmonton",
        "Calgary",
        "St. Albert",
        "Sherwood Park",
        "Leduc",
        "Airdrie",
        "Red Deer",
        "Fort McMurray",
    ]

    city = st.selectbox(
        "City",
        city_options,
        index=city_options.index(scenario["city"]),
    )

    neighborhood_options = {
        "Edmonton": [
            "Glenora",
            "Windermere",
            "Oliver",
            "Summerside",
            "Terwillegar",
            "Laurel",
            "Rutherford",
            "Westmount",
        ],
        "Calgary": [
            "Beltline",
            "Kensington",
            "Mahogany",
            "Bridgeland",
            "Aspen Woods",
            "Seton",
            "Inglewood",
            "Auburn Bay",
        ],
        "Red Deer": ["Clearview", "Timberlands", "West Park", "Anders", "Lancaster"],
        "St. Albert": ["Erin Ridge", "Lacombe Park", "Oakmont", "Mission", "Riverside"],
        "Sherwood Park": [
            "Summerwood",
            "Emerald Hills",
            "Clarkdale",
            "Nottingham",
            "Davidson Creek",
        ],
        "Leduc": ["Meadowview", "Southfork", "Black Stone", "Tribute", "West Haven"],
        "Airdrie": ["Coopers Crossing", "Kings Heights", "Bayside", "Reunion", "Hillcrest"],
        "Fort McMurray": ["Timberlea", "Thickwood", "Eagle Ridge", "Abasand", "Beacon Hill"],
    }

    neighborhood = st.selectbox(
        "Neighborhood",
        neighborhood_options[city],
        index=(
            neighborhood_options[city].index(scenario["neighborhood"])
            if scenario["neighborhood"] in neighborhood_options[city]
            else 0
        ),
    )

    property_type_options = [item.value for item in PropertyType]
    property_type = st.selectbox(
        "Property type",
        property_type_options,
        index=property_type_options.index(scenario["property_type"]),
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=12,
        value=scenario["bedrooms"],
        step=1,
    )
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.0,
        max_value=10.0,
        value=scenario["bathrooms"],
        step=0.5,
    )
    living_area_sqft = st.number_input(
        "Living area sqft",
        min_value=250,
        max_value=12000,
        value=scenario["living_area_sqft"],
        step=50,
    )
    lot_size_sqft = st.number_input(
        "Lot size sqft",
        min_value=0,
        max_value=100000,
        value=scenario["lot_size_sqft"],
        step=100,
    )
    year_built = st.number_input(
        "Year built",
        min_value=1850,
        max_value=2026,
        value=scenario["year_built"],
        step=1,
    )
    garage_spaces = st.number_input(
        "Garage spaces",
        min_value=0,
        max_value=6,
        value=scenario["garage_spaces"],
        step=1,
    )

    condition_options = [item.value for item in PropertyCondition]
    condition = st.selectbox(
        "Condition",
        condition_options,
        index=condition_options.index(scenario["condition"]),
    )

    finished_basement = st.checkbox("Finished basement", value=scenario["finished_basement"])
    renovated = st.checkbox("Renovated", value=scenario["renovated"])
    near_transit = st.checkbox("Near transit", value=scenario["near_transit"])
    backs_onto_park = st.checkbox("Backs onto park", value=scenario["backs_onto_park"])

    st.divider()

    use_llm = st.checkbox(
        "Use LLM memo if OPENAI_API_KEY is configured",
        value=False,
    )

    analyze = st.button("Run comp analysis", type="primary")


try:
    sales_data = cached_load_sales_data()
    st.success(f"Loaded {len(sales_data):,} synthetic Alberta residential sales.")
except DataLoadError as exc:
    st.error(str(exc))
    st.stop()


if not analyze:
    st.info("Enter a subject property in the sidebar and click **Run comp analysis**.")
    st.stop()


try:
    subject = SubjectProperty(
        city=city,
        neighborhood=neighborhood,
        property_type=PropertyType(property_type),
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        living_area_sqft=living_area_sqft,
        lot_size_sqft=lot_size_sqft if lot_size_sqft > 0 else None,
        year_built=year_built,
        garage_spaces=garage_spaces,
        condition=PropertyCondition(condition),
        finished_basement=finished_basement,
        renovated=renovated,
        near_transit=near_transit,
        backs_onto_park=backs_onto_park,
    )
except Exception as exc:
    st.error(f"Invalid subject property input: {exc}")
    st.stop()


search, scored, valuation, memo = run_analysis(subject, use_llm=use_llm)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Base estimate", format_currency(valuation.base_estimate))
metric_col2.metric("Low estimate", format_currency(valuation.low_estimate))
metric_col3.metric("High estimate", format_currency(valuation.high_estimate))
metric_col4.metric("Confidence", f"{valuation.confidence} ({valuation.confidence_score:.2f})")

st.subheader("Comp search result")
st.write(search.message)

if scored.empty:
    st.warning("No comparable sales were available for this subject property.")
    st.stop()


display_columns = [
    "property_id",
    "city",
    "neighborhood",
    "property_type",
    "sale_date",
    "sale_price",
    "price_per_sqft",
    "living_area_sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
    "condition",
    "total_score",
    "reason_selected",
]

available_display_columns = [col for col in display_columns if col in scored.columns]
top_comps = scored.head(10)[available_display_columns].copy()

top_comps["sale_price"] = top_comps["sale_price"].map(format_currency)
top_comps["price_per_sqft"] = top_comps["price_per_sqft"].map(format_currency)

st.subheader("Top comparable sales")
st.dataframe(top_comps, use_container_width=True, hide_index=True)

st.subheader("Score breakdown")

score_columns = [
    "property_id",
    "location_score",
    "property_type_score",
    "living_area_score",
    "bed_bath_score",
    "year_built_score",
    "recency_score",
    "features_score",
    "total_score",
]

available_score_columns = [col for col in score_columns if col in scored.columns]
st.dataframe(scored.head(10)[available_score_columns], use_container_width=True, hide_index=True)

st.subheader("Risk flags")
for flag in valuation.risk_flags:
    st.write(f"- {flag}")

st.subheader("Valuation memo")
st.markdown(memo)