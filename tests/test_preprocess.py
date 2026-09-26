from jira_categorizer.data.preprocess import clean_text, combine_summary_description
from jira_categorizer.data.schema import split_multi_value


def test_clean_text_strips_html_and_urls():
    raw = '<p>See https://example.com for details</p>'
    assert "https" not in clean_text(raw)
    assert "See" in clean_text(raw)
    assert "<p>" not in clean_text(raw)


def test_combine_summary_description():
    text = combine_summary_description("Timeout", "Gateway 504 on checkout")
    assert "Timeout" in text
    assert "Gateway" in text


def test_split_multi_value_pipe():
    assert split_multi_value("payments|p1") == ["payments", "p1"]


def test_split_multi_value_dedupes():
    assert split_multi_value("Auth; auth; AUTH") == ["auth"]
