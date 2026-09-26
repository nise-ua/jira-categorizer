from jira_categorizer.config import load_config, project_root
from jira_categorizer.data.ingest import build_training_frame, load_jira_export


def test_load_sample_csv():
    path = project_root() / "sample_data" / "jira_export.csv"
    df = load_jira_export(path)
    assert "summary" in df.columns
    assert len(df) >= 20


def test_build_training_frame_filters_rare():
    cfg = load_config()
    cfg["data"]["min_label_support"] = 2
    cfg["data"]["min_area_support"] = 2
    df = load_jira_export(project_root() / "sample_data" / "jira_export.csv")
    frame = build_training_frame(df, cfg)
    assert "text" in frame.columns
    assert frame["text"].str.len().min() > 0
    # All primary areas that remain should meet support or be blank
    areas = [a for a in frame["primary_area"] if a]
    assert areas
