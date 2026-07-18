from shadowtrace.ingest.otel_sink import (
    OtelSink,
    cross_check_completeness,
    parse_otlp_metrics_json,
)

_SAMPLE_PAYLOAD = {
    "resourceMetrics": [
        {
            "scopeMetrics": [
                {
                    "metrics": [
                        {
                            "name": "claude_code.token.usage",
                            "sum": {
                                "dataPoints": [
                                    {
                                        "asInt": "150",
                                        "timeUnixNano": "1752800000000000000",
                                        "attributes": [
                                            {
                                                "key": "model",
                                                "value": {"stringValue": "claude-sonnet-5"},
                                            },
                                            {
                                                "key": "session.id",
                                                "value": {"stringValue": "sess-1"},
                                            },
                                        ],
                                    }
                                ]
                            },
                        },
                        {"name": "claude_code.cost.usage", "sum": {"dataPoints": []}},
                    ]
                }
            ]
        }
    ]
}


def test_parse_otlp_metrics_extracts_token_points() -> None:
    points = parse_otlp_metrics_json(_SAMPLE_PAYLOAD)
    assert len(points) == 1
    assert points[0].value == 150.0
    assert points[0].model == "claude-sonnet-5"
    assert points[0].session_id == "sess-1"


def test_sink_ingest_and_total_tokens() -> None:
    sink = OtelSink()
    n = sink.ingest(_SAMPLE_PAYLOAD)
    assert n == 1
    assert sink.total_tokens() == 150.0


def test_sink_ingest_malformed_payload_does_not_raise() -> None:
    sink = OtelSink()
    assert sink.ingest({"resourceMetrics": "not-a-list-of-dicts"}) == 0


def test_cross_check_completeness_matches() -> None:
    assert cross_check_completeness(150.0, 150.0) == 1.0
    assert cross_check_completeness(100.0, 200.0) == 0.5
    assert cross_check_completeness(0.0, 0.0) == 1.0
    assert cross_check_completeness(50.0, 0.0) == 0.0
