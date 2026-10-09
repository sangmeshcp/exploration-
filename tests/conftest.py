from collections.abc import Iterator

import pytest

from shadowtrace.common.tracing import register_process_tracer_provider


@pytest.fixture(autouse=True)
def _isolate_process_tracer() -> Iterator[None]:
    yield
    register_process_tracer_provider(None)
