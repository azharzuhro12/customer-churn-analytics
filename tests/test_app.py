"""AppTest checks for the Streamlit dashboard.

Run from the project root:  python -m pytest tests -q
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_four_tabs() -> None:
    at = run_app()
    assert at.title[0].value == "Customer churn analytics"
    assert [tab.label for tab in at.tabs] == [
        "Overview",
        "At-risk customers",
        "Retention campaign backtest",
        "Predict a customer",
    ]


def test_sidebar_filter_updates_kpis() -> None:
    at = run_app()
    at.sidebar.multiselect[0].set_value(["Two year"]).run()
    assert not at.exception
    assert at.metric[0].value == "1,695"
    assert at.metric[1].value == "2.8%"


def test_empty_filter_warns_and_stops() -> None:
    at = run_app()
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    assert any("No customers match" in w.value for w in at.warning)
    assert not at.tabs  # st.stop() ran before the tabs were rendered


def test_risk_threshold_changes_count() -> None:
    at = run_app()
    before = at.metric[4].value
    at.slider[0].set_value(0.9).run()
    assert not at.exception
    after = at.metric[4].value
    assert before != after


def test_predict_form_submits() -> None:
    at = run_app()
    at.number_input[0].set_value(24)
    at.selectbox[1].set_value("No")  # internet service
    at.selectbox[3].set_value("No")  # phone service
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Churn risk score" in labels
    assert "Risk percentile" in labels
