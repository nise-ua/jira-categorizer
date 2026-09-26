from jira_categorizer.analytics.hierarchy import build_income_tree
from jira_categorizer.config import project_root
from jira_categorizer.data.ingest import load_jira_export


def test_income_tree_has_two_layers():
    df = load_jira_export(project_root() / "sample_data" / "public_jira" / "kafka_100.csv")
    tree = build_income_tree(df, period="3m")
    assert tree["total"] >= 1
    assert tree["sunburst"]["name"] == "Tickets"
    assert tree["sunburst"]["children"]
    # at least one area with label children
    assert any(c.get("children") for c in tree["sunburst"]["children"])
    assert "largest branch" in tree["headline"] or "tickets" in tree["headline"].lower()
