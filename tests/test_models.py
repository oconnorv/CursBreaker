from types import SimpleNamespace

from cursbreaker.models import TokenUsage


def test_add_response_from_sdk_object():
    u = TokenUsage()
    u.add_response(
        SimpleNamespace(
            prompt_token_count=1000,
            candidates_token_count=200,
            thoughts_token_count=50,
            total_token_count=1250,
        )
    )
    assert u.input == 1000
    assert u.output == 200
    assert u.thinking == 50
    assert u.calls == 1
    assert u.total == 1250


def test_add_response_from_dict():
    u = TokenUsage()
    u.add_response({"prompt_token_count": 10, "candidates_token_count": 5})
    assert (u.input, u.output, u.thinking, u.calls) == (10, 5, 0, 1)


def test_add_response_derives_output_from_total_when_candidates_missing():
    # Some SDK/model combinations omit candidates but report a total.
    u = TokenUsage()
    u.add_response(
        SimpleNamespace(
            prompt_token_count=100, thoughts_token_count=20, total_token_count=170
        )
    )
    assert u.output == 50  # 170 - 100 - 20
    assert u.total == 170


def test_add_response_none_still_counts_the_call():
    u = TokenUsage()
    u.add_response(None)
    assert u.calls == 1
    assert u.total == 0


def test_add_response_tolerates_none_fields():
    u = TokenUsage()
    u.add_response(
        SimpleNamespace(
            prompt_token_count=None,
            candidates_token_count=None,
            thoughts_token_count=None,
        )
    )
    assert (u.input, u.output, u.thinking, u.calls) == (0, 0, 0, 1)


def test_add_accumulates_across_calls():
    u = TokenUsage()
    md = {"prompt_token_count": 100, "candidates_token_count": 20}
    u.add_response(md)
    u.add_response(md)
    assert u.input == 200
    assert u.output == 40
    assert u.calls == 2


def test_add_and_sub_are_fieldwise():
    a = TokenUsage(input=100, output=20, thinking=5, calls=1)
    b = TokenUsage(input=50, output=10, thinking=1, calls=1)
    s = a + b
    assert (s.input, s.output, s.thinking, s.calls) == (150, 30, 6, 2)
    d = a - b
    assert (d.input, d.output, d.thinking, d.calls) == (50, 10, 4, 0)


def test_sub_clamps_at_zero():
    # A per-file delta must never go negative even if usage is read oddly.
    d = TokenUsage(input=1) - TokenUsage(input=5)
    assert d.input == 0


def test_cost_bills_thinking_at_the_output_rate():
    u = TokenUsage(input=1_000_000, output=500_000, thinking=500_000)
    # input: 1M * $2 = $2 ; (output + thinking) = 1M * $10 = $10
    assert u.cost(2.0, 10.0) == 12.0


def test_cost_zero_prices_is_zero():
    assert TokenUsage(input=999, output=999, thinking=999).cost(0.0, 0.0) == 0.0
