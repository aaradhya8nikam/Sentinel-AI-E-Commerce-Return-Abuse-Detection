from src.cost_optimizer import evaluate_threshold, optimize_threshold

def test_cost_formula():
    result = evaluate_threshold([0, 0, 1, 1], [0.9, 0.1, 0.2, 0.8], 0.5)
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["total_cost_inr"] == 5000

def test_optimizer_returns_valid_threshold():
    best, table = optimize_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.05 <= best["threshold"] <= 0.95
    assert not table.empty
