"""Monitoring: category surge anomalies and new-label suggestions."""

__all__ = ["detect_category_surges", "suggest_new_labels", "score_novelty"]


def __getattr__(name: str):
    if name == "detect_category_surges":
        from jira_categorizer.monitor.surge import detect_category_surges

        return detect_category_surges
    if name == "suggest_new_labels":
        from jira_categorizer.monitor.new_labels import suggest_new_labels

        return suggest_new_labels
    if name == "score_novelty":
        from jira_categorizer.monitor.novelty import score_novelty

        return score_novelty
    raise AttributeError(name)
