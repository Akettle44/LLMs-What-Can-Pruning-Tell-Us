### Shared functions for unit tests ###

import os
import pytest

@pytest.fixture()
def setUp():
    test_dir = os.getcwd()
    root_dir = os.path.dirname(test_dir)
    model_dir = os.path.join(root_dir, "models")
    return test_dir, root_dir, model_dir

@pytest.fixture
def tearDown():
    pass