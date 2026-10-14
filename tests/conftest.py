import pytest

from interp.model import load_model, model_spec


@pytest.fixture(scope="session")
def spec():
    return model_spec("gpt2-small")


@pytest.fixture(scope="session")
def model(spec):
    return load_model(spec)
