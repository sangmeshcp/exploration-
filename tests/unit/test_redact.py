import hashlib

from hypothesis import given
from hypothesis import strategies as st

from shadowtrace.proxy.redact import entropy_scan, redact_trace, scrub_text

TRUE_SECRETS = [
    "AKIAABCDEFGHIJKLMNOP",
    "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "AIzaSyD-1234567890abcdefghijklmnopqrstuv",
    (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    ),
    "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END RSA PRIVATE KEY-----",
    "xoxb-" + "NOTREAL-faketoken-fortests",
    "ghp_" + "a" * 40,
    "sk-ant-" + "a1B2c3D4e5F6" * 3,
    "api_key: 'sk_live_abcdefgh12345678'",
]


def test_true_secrets_are_scrubbed() -> None:
    for secret in TRUE_SECRETS:
        scrubbed, flags = scrub_text(f"here is a secret {secret} ok")
        assert secret not in scrubbed, f"leaked: {secret}"
        assert flags, f"no pattern matched: {secret}"


def test_high_entropy_code_is_not_scrubbed_but_flagged() -> None:
    code_hash = hashlib.sha256(b"def foo(): pass").hexdigest()
    text = f"commit {code_hash} fixed the bug"
    scrubbed, flags = scrub_text(text)
    assert code_hash in scrubbed  # not scrubbed
    assert flags == []
    hits = entropy_scan(text)
    assert code_hash in hits  # but flagged for quarantine


def test_redact_trace_quarantines_without_deleting() -> None:
    code_hash = hashlib.sha256(b"minified js blob content").hexdigest()
    request = {"messages": [{"role": "user", "content": f"look at {code_hash}"}]}
    response = {"content": [{"type": "text", "text": "ok"}]}
    result = redact_trace(request, response)
    assert result.quarantined is True
    assert code_hash in result.request_json["messages"][0]["content"]  # preserved
    assert "entropy_quarantine" in result.flags


def test_redact_trace_scrubs_nested_structures() -> None:
    request = {
        "messages": [{"role": "user", "content": "my key is AKIAABCDEFGHIJKLMNOP please help"}]
    }
    response = {"content": [{"type": "text", "text": "no secrets here"}]}
    result = redact_trace(request, response)
    assert "AKIAABCDEFGHIJKLMNOP" not in json_dump(result.request_json)
    assert "aws_access_key_id" in result.flags
    assert result.quarantined is False


def json_dump(value: object) -> str:
    import json

    return json.dumps(value)


@given(st.text(alphabet="0123456789abcdefABCDEF", min_size=25, max_size=64))
def test_hex_like_strings_never_crash_entropy_scan(s: str) -> None:
    entropy_scan(s)  # must not raise
    scrub_text(s)  # must not raise
