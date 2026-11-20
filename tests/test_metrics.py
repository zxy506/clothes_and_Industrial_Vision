from src.evaluation.evaluate_fashion import metrics


def test_metrics_match_known_confusion_matrix():
    result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["accuracy"] == 0.75
    assert result["recall_by_class"][:2] == [0.5, 1.0]
    assert result["confusion_matrix"][0][:2] == [1, 1]
    assert result["confusion_matrix"][1][:2] == [0, 2]
