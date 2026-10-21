"""Streamlit dashboard for the Customer Churn Analytics project.

Run from the project root:  streamlit run app.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.common import DB_PATH, FEATURES, MODEL_PATH, REPORTS_DIR, read_sql

BLUE = "#2a78d6"
INK = "#16202c"
MUTED = "#5a6672"
AXIS = "#d5dce4"

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
      div[data-testid="stMetric"] {
          background: #ffffff;
          border: 1px solid #e3e9f0;
          border-radius: 10px;
          padding: 14px 16px;
      }
      div[data-testid="stMetric"] label { color: #5a6672; }
      .small-note { color: #5a6672; font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data and model loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data(db_mtime: float) -> pd.DataFrame:
    """Customers plus their out-of-fold churn risk."""
    return read_sql(
        """
        SELECT c.*, s.churn_risk
        FROM customers AS c
        LEFT JOIN customer_scores AS s ON c.customerID = s.customerID
        """
    )


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the saved model, retraining once if the file is missing or unreadable."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from src.train import main as train_main

        with st.spinner("Model missing or unreadable — retraining (about 20 seconds)…"):
            train_main()
        return joblib.load(MODEL_PATH)


model = get_model()
df = load_data(DB_PATH.stat().st_mtime)

if df["churn_risk"].isna().any():
    from src.train import main as train_main

    with st.spinner("Risk scores missing — retraining (about 20 seconds)…"):
        train_main()
    load_data.clear()
    df = load_data(DB_PATH.stat().st_mtime)

df = df.assign(
    tenure_group=pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )
)

try:
    METRICS = json.loads((REPORTS_DIR / "metrics.json").read_text())
except Exception:
    METRICS = {}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def suggest_action(row: pd.Series) -> str:
    """Explainable rules, not a black box: risk band + the levers we can pull."""
    risk = float(row["churn_risk"])
    if risk >= 0.70 and row["Contract"] == "Month-to-month":
        return "Annual contract offer + loyalty credit"
    if risk >= 0.70:
        return "Priority retention call + offer"
    if risk >= 0.50 and row["TechSupport"] in ("No", "No internet service"):
        return "Free Tech Support trial + review"
    if risk >= 0.50:
        return "Targeted email + plan review"
    return "Monitor only"


def churn_bar(
    data: pd.DataFrame,
    column: str,
    title: str,
    horizontal: bool = False,
    order: list[str] | None = None,
) -> go.Figure:
    """Churn-rate bar chart with an overall-rate reference line and value labels."""
    grouped = (
        data.groupby(column, observed=True)
        .agg(customers=("churn_flag", "size"), churned=("churn_flag", "sum"))
        .reset_index()
    )
    grouped["churn_rate_pct"] = 100 * grouped["churned"] / grouped["customers"]

    if order is not None:
        grouped[column] = pd.Categorical(grouped[column], categories=order, ordered=True)
        grouped = grouped.sort_values(column)
    elif horizontal:
        grouped = grouped.sort_values("churn_rate_pct")

    overall = 100 * data["churn_flag"].mean()
    labels = grouped[column].astype(str)
    values = grouped["churn_rate_pct"]
    text = [f"{v:.1f}%" for v in values]

    fig = go.Figure()
    if horizontal:
        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=BLUE,
                text=text,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Churn rate %{x:.2f}%<extra></extra>",
            )
        )
        fig.add_vline(
            x=overall,
            line_dash="dash",
            line_color="#8a97a5",
            line_width=1.3,
            annotation_text=f"Overall {overall:.1f}%",
            annotation_position="top",
            annotation_font_color=MUTED,
            annotation_font_size=10,
        )
        fig.update_xaxes(title=None, range=[0, max(values) * 1.22 + 2])
    else:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=BLUE,
                text=text,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>Churn rate %{y:.2f}%<extra></extra>",
            )
        )
        fig.add_hline(
            y=overall,
            line_dash="dash",
            line_color="#8a97a5",
            line_width=1.3,
            annotation_text=f"Overall {overall:.1f}%",
            annotation_position="top right",
            annotation_font_color=MUTED,
            annotation_font_size=10,
        )
        fig.update_yaxes(title=None, range=[0, max(values) * 1.25 + 2])

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK)),
        height=310,
        margin=dict(l=6, r=12, t=52, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans serif", color=MUTED, size=11),
        showlegend=False,
        bargap=0.5,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS, automargin=True)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=AXIS, automargin=True)
    return fig


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.title("Customer Churn Analytics")
st.sidebar.caption("IBM Telco sample · 7,043 customers · snapshot data")

contracts = st.sidebar.multiselect(
    "Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique())
)
internet = st.sidebar.multiselect(
    "Internet service",
    sorted(df["InternetService"].unique()),
    default=sorted(df["InternetService"].unique()),
)
tenure_range = st.sidebar.slider("Tenure (months)", 0, 72, (0, 72))

mask = (
    df["Contract"].isin(contracts)
    & df["InternetService"].isin(internet)
    & df["tenure"].between(*tenure_range)
)
filtered = df[mask]

if filtered.empty:
    st.warning("No customers match these filters. Widen the selection in the sidebar.")
    st.stop()

st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} customers")
st.sidebar.divider()
st.sidebar.markdown(
    """
    **How the risk score works**
    Scores are out-of-fold model predictions: every customer was scored by a model that
    never saw their churn label. They rank risk; they are not calibrated probabilities.
    """
)
if METRICS:
    st.sidebar.caption(
        f"Selected model: {METRICS['best_model'].replace('_', ' ')} · "
        f"test ROC-AUC {METRICS['test']['roc_auc']:.4f}"
    )

st.title("Customer churn analytics")
st.caption(
    "Where churn concentrates, who to contact first, and what a retention campaign "
    "could return under explicit assumptions."
)

tab_overview, tab_risk, tab_campaign, tab_predict = st.tabs(
    ["Overview", "At-risk customers", "Retention campaign backtest", "Predict a customer"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    total = len(filtered)
    churned = int(filtered["churn_flag"].sum())
    rate = 100 * filtered["churn_flag"].mean()
    revenue = float(filtered["MonthlyCharges"].sum())
    lost = float(filtered.loc[filtered["churn_flag"] == 1, "MonthlyCharges"].sum())
    lost_share = 100 * lost / revenue if revenue else 0.0

    cols = st.columns(4)
    cols[0].metric("Customers", f"{total:,}")
    cols[1].metric("Churn rate", f"{rate:.1f}%", help=f"{churned:,} churned customers")
    cols[2].metric("Monthly revenue lost", money(lost), help="Monthly charges of churned customers")
    cols[3].metric("Share of revenue lost", f"{lost_share:.1f}%", help="Of the filtered monthly revenue")

    st.write("")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            churn_bar(
                filtered,
                "Contract",
                "Month-to-month contracts carry the risk",
                order=["Month-to-month", "One year", "Two year"],
            ),
            width="stretch",
        )
        st.plotly_chart(
            churn_bar(
                filtered,
                "tenure_group",
                "Churn concentrates in the first year",
                order=["0-12", "13-24", "25-48", "49-72"],
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            churn_bar(
                filtered,
                "PaymentMethod",
                "Electronic check customers churn most",
                horizontal=True,
            ),
            width="stretch",
        )
        st.plotly_chart(
            churn_bar(
                filtered,
                "InternetService",
                "Fiber optic customers churn more than DSL",
                order=["Fiber optic", "DSL", "No"],
            ),
            width="stretch",
        )

    st.caption(
        "The dashed line is the overall churn rate in the current filter. "
        "Bar labels are churn rates within each group."
    )


# --------------------------------------------------------------------------- #
# Tab 2 — At-risk customers
# --------------------------------------------------------------------------- #
with tab_risk:
    st.subheader("Active customers ranked by churn risk")
    st.caption(
        "Active = not churned in this snapshot. Scores are out-of-fold predictions, "
        "so no customer is scored by a model that saw their label."
    )

    active = filtered[filtered["churn_flag"] == 0]
    if active.empty:
        st.info("No active customers in this filter.")
    else:
        threshold = st.slider(
            "Minimum risk score",
            min_value=0.0,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Contact only customers at or above this risk score.",
        )
        flagged = active[active["churn_risk"] >= threshold].sort_values(
            "churn_risk", ascending=False
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Customers at risk", f"{len(flagged):,}")
        k2.metric("Monthly revenue at risk", money(float(flagged["MonthlyCharges"].sum())))
        k3.metric(
            "Average risk score",
            f"{flagged['churn_risk'].mean():.2f}" if len(flagged) else "—",
        )

        if flagged.empty:
            st.info("No customers are above this risk threshold. Lower the slider.")
        else:
            table = flagged[
                [
                    "customerID",
                    "tenure",
                    "Contract",
                    "InternetService",
                    "MonthlyCharges",
                    "churn_risk",
                ]
            ].copy()
            table["suggested_action"] = flagged.apply(suggest_action, axis=1)
            table = table.sort_values("churn_risk", ascending=False)

            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height=420,
                column_config={
                    "customerID": st.column_config.TextColumn("Customer ID", width=120),
                    "tenure": st.column_config.NumberColumn("Tenure (mo)", format="%d", width=80),
                    "Contract": st.column_config.TextColumn("Contract", width=135),
                    "InternetService": st.column_config.TextColumn("Internet", width=100),
                    "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f", width=95),
                    "churn_risk": st.column_config.ProgressColumn(
                        "Risk score", min_value=0.0, max_value=1.0, format="%.3f", width=120
                    ),
                    "suggested_action": st.column_config.TextColumn("Suggested action", width=300),
                },
            )

            st.download_button(
                "Download this list as CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name=f"at_risk_customers_{threshold:.2f}.csv",
                mime="text/csv",
            )
            st.caption(
                "Suggested actions are fixed rules based on risk band, contract type and "
                "Tech Support, not another model."
            )


# --------------------------------------------------------------------------- #
# Tab 3 — Retention campaign backtest
# --------------------------------------------------------------------------- #
with tab_campaign:
    st.subheader("Retention campaign backtest")
    st.caption(
        "Backtest on the historical customers in the current filter. They are ranked by "
        "their out-of-fold risk score, exactly as a campaign would rank them."
    )

    churned_revenue = float(
        (filtered["churn_flag"] * filtered["MonthlyCharges"]).sum()
    )
    if churned_revenue == 0:
        st.info("This filter contains no churners, so the backtest has nothing to measure.")
    else:
        c1, c2, c3 = st.columns(3)
        cost_per_contact = c1.slider("Cost per contact ($)", 5, 100, 25, 5)
        success_rate = c2.slider("Offer success rate (%)", 5, 60, 25, 5) / 100
        months_kept = c3.slider("Months of revenue kept", 1, 24, 12, 1)

        ranked = filtered.sort_values("churn_risk", ascending=False).reset_index(drop=True)
        n = len(ranked)
        contacted = np.arange(1, n + 1)
        monthly_value = (ranked["churn_flag"].to_numpy() * ranked["MonthlyCharges"].to_numpy())
        model_net = monthly_value.cumsum() * success_rate * months_kept - contacted * cost_per_contact

        # Random targeting: expected monthly churner revenue per randomly chosen customer.
        random_net = (
            contacted * (monthly_value.sum() / n) * success_rate * months_kept
            - contacted * cost_per_contact
        )

        best_idx = int(np.argmax(model_net))
        best_share = 100 * (best_idx + 1) / n
        best_net = float(model_net[best_idx])
        best_contacts = best_idx + 1

        at20_idx = max(int(0.2 * n) - 1, 0)
        net_at_20 = float(model_net[at20_idx])
        random_at_20 = float(random_net[at20_idx])

        k1, k2, k3, k4 = st.columns(4)
        if best_net > 0:
            k1.metric("Best share to contact", f"{best_share:.1f}%", help=f"{best_contacts:,} customers")
            k2.metric("Net at the best point", money(best_net))
        else:
            k1.metric("Best share to contact", "None profitable")
            k2.metric("Net at the best point", money(best_net))
        k3.metric("Net at 20% contacted", money(net_at_20), help="Model-ranked targeting")
        k4.metric("Net at 20%, random", money(random_at_20), help="Random targeting baseline")

        # Thin the curves for a light chart payload.
        step = max(n // 250, 1)
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=100 * (np.array(idx) + 1) / n,
                y=random_net[idx],
                mode="lines",
                name="Random targeting",
                line=dict(color="#9aa6b2", width=2, dash="dash"),
                hovertemplate="Contacted %{x:.1f}%<br>Net $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=100 * (np.array(idx) + 1) / n,
                y=model_net[idx],
                mode="lines",
                name="Model targeting",
                line=dict(color=BLUE, width=2.6),
                hovertemplate="Contacted %{x:.1f}%<br>Net $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_color=AXIS, line_width=1)
        if best_net > 0:
            fig.add_vline(
                x=best_share,
                line_dash="dot",
                line_color="#8a97a5",
                line_width=1,
                annotation_text=f"Best {best_share:.1f}%",
                annotation_position="top",
                annotation_font_color=MUTED,
                annotation_font_size=10,
            )
        fig.update_layout(
            title=dict(text="Model-ranked contacts turn profitable earlier", font=dict(size=14, color=INK)),
            height=420,
            margin=dict(l=10, r=10, t=56, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans serif", color=MUTED, size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            hovermode="x unified",
        )
        fig.update_xaxes(title="Share of customers contacted, riskiest first (%)", showgrid=False, linecolor=AXIS)
        fig.update_yaxes(title="Net value ($)", showgrid=True, gridcolor="#e8edf3", zeroline=False, linecolor=AXIS)
        st.plotly_chart(fig, width="stretch")

        st.warning(
            "These money figures depend entirely on the assumptions above "
            f"({money(cost_per_contact)} per contact, {success_rate:.0%} success, "
            f"{months_kept} months kept). They are a scenario, not audited return. "
            "The backtest also assumes the score ranks churners the same way in a live campaign."
        )


# --------------------------------------------------------------------------- #
# Tab 4 — Predict a customer
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Score a customer profile")
    st.caption(
        "Enter the profile, then submit. Add-ons are made consistent automatically "
        "(no internet → add-ons set to 'No internet service'; no phone → 'No phone service')."
    )

    with st.form("customer_profile"):
        a, b, c = st.columns(3)
        tenure = a.number_input("Tenure (months)", 0, 72, 12, 1)
        monthly = b.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 1.0)
        total = c.number_input(
            "Total charges ($)",
            0.0,
            10000.0,
            840.0,
            10.0,
            help="Approximate: tenure × monthly charges for a new profile.",
        )

        d, e, f = st.columns(3)
        contract = d.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = e.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        payment = f.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        g, h, i = st.columns(3)
        phone_service = g.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = h.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        paperless = i.selectbox("Paperless billing", ["Yes", "No"])

        j, k, l = st.columns(3)
        senior = j.selectbox("Senior citizen", ["No", "Yes"])
        partner = k.selectbox("Partner", ["No", "Yes"])
        dependents = l.selectbox("Dependents", ["No", "Yes"])

        st.markdown("**Internet add-ons**")
        m, n, o = st.columns(3)
        online_security = m.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = n.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = o.selectbox("Device protection", ["No", "Yes", "No internet service"])
        p, q, r = st.columns(3)
        tech_support = p.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = q.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = r.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Score this customer", type="primary")

    if submitted:
        # Keep the profile internally consistent before scoring.
        if internet_service == "No":
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
        if phone_service == "No":
            multiple_lines = "No phone service"

        profile = {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
        }
        input_row = pd.DataFrame([profile])[FEATURES]
        risk = float(model.predict_proba(input_row)[0, 1])
        percentile = 100 * float((df["churn_risk"] < risk).mean())

        row_for_action = pd.Series({**profile, "churn_risk": risk})
        action = suggest_action(row_for_action)

        s1, s2 = st.columns(2)
        s1.metric("Churn risk score", f"{risk:.3f}")
        s2.metric("Risk percentile", f"{percentile:.0f}th", help="Versus all 7,043 scored customers")

        st.info(f"**Suggested action:** {action}")
        st.progress(min(max(risk, 0.0), 1.0))
        st.caption(
            "Scores rank risk, they are not exact probabilities: a 0.70 means this profile "
            "ranks above about 70% of customers, not that churn has a 70% chance. "
            "The score comes from the model trained on all 7,043 customers."
        )
