from src.evaluation.scoring import calculate_official_score, mean_official_score


def test_no_delivery_always_scores_zero() -> None:
    assert calculate_official_score([], horizon=400, total_team_timeouts=50) == 0


def test_zero_based_delivery_timestamps_and_multiple_soups() -> None:
    assert calculate_official_score([0], horizon=400, total_team_timeouts=0) == 14_400
    assert calculate_official_score([0, 399], horizon=400, total_team_timeouts=0) == 20_410


def test_simultaneous_deliveries_count_as_two_soups() -> None:
    assert calculate_official_score([10, 10], horizon=400, total_team_timeouts=0) == 24_290


def test_team_timeout_penalty_is_capped() -> None:
    without_timeouts = calculate_official_score([100], horizon=400, total_team_timeouts=0)
    assert calculate_official_score([100], horizon=400, total_team_timeouts=3) == without_timeouts - 300
    assert calculate_official_score([100], horizon=400, total_team_timeouts=100) == without_timeouts - 5_000


def test_three_seed_average() -> None:
    assert mean_official_score([10, 20, 30]) == 20.0
