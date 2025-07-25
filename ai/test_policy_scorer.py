import pytest
from ai.policy_scorer import PolicyScorer

@pytest.fixture
def scorer():
    return PolicyScorer(batch_size=2, max_workers=2)

def test_segment_text(scorer):
    text = "Para1.\n\nPara2 with long text" * 10
    chunks = scorer._segment_text(text, max_chunk_length=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(c) >= 10 for c in chunks)  # min adjusted for test

def test_score_policy_text(scorer, mocker):
    mock_classifier = mocker.patch.object(scorer, 'classifier')
    mock_classifier.side_effect = lambda batch, labels, multi_label: [
        {'labels': labels, 'scores': [0.9 if 'positive' in l else 0.1 for l in labels]} for _ in batch
    ]
    scores = scorer.score_policy_text("Test text", confidence_threshold=0.5)
    assert all(-1 <= s <= 1 for s in scores.values())
    assert len(scores) == 5  # All domains

def test_export_for_iqa(scorer, tmp_path):
    scores = {"Economic": 0.5}
    output = tmp_path / "test.json"
    scorer.export_for_iqa("Text", scores, str(output))
    assert output.exists()

def test_fine_tune(scorer):
    with pytest.raises(ValueError):  # No data
        scorer.fine_tune([])