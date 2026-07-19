from shadowtrace.replay.ladder import Candidate, load_ladder, parse_ladder_yaml, sorted_by_cost


def test_load_default_ladder_from_package() -> None:
    candidates = load_ladder()
    assert len(candidates) >= 3
    names = {c.name for c in candidates}
    assert "claude-haiku-4-5" in names
    assert any(c.local for c in candidates)


def test_candidate_cost_usd_matches_hand_computed() -> None:
    c = Candidate(
        name="claude-haiku-4-5", provider="anthropic", price_per_mtok_in=1.0, price_per_mtok_out=5.0
    )
    # 2,000,000 in-tokens * $1/M = $2.00; 100,000 out-tokens * $5/M = $0.50
    cost = c.cost_usd(tokens_in=2_000_000, tokens_out=100_000)
    assert abs(cost - 2.50) < 1e-9


def test_local_candidate_is_always_free() -> None:
    c = Candidate(
        name="ollama-x",
        provider="ollama",
        price_per_mtok_in=999,
        price_per_mtok_out=999,
        local=True,
    )
    assert c.cost_usd(1_000_000, 1_000_000) == 0.0


def test_sorted_by_cost_puts_local_first_then_ascending_price() -> None:
    candidates = [
        Candidate("expensive", "anthropic", 15.0, 75.0),
        Candidate("cheap-cloud", "openrouter", 0.4, 0.4),
        Candidate("local", "ollama", 0.0, 0.0, local=True),
        Candidate("mid-cloud", "anthropic", 1.0, 5.0),
    ]
    ordered = [c.name for c in sorted_by_cost(candidates)]
    assert ordered == ["local", "cheap-cloud", "mid-cloud", "expensive"]


def test_parse_ladder_yaml_handles_minimal_entry() -> None:
    text = "candidates:\n  - name: x\n    provider: ollama\n    local: true\n"
    candidates = parse_ladder_yaml(text)
    assert candidates == [
        Candidate(
            name="x", provider="ollama", price_per_mtok_in=0.0, price_per_mtok_out=0.0, local=True
        )
    ]


def test_parse_ladder_yaml_empty() -> None:
    assert parse_ladder_yaml("") == []
