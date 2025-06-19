from pathlib import Path
from pasna_analysis.gui import Model

import pytest

from pasna_analysis import Config

DATASET_1 = Path("./tests/assets/data/20250210")
DATASET_2 = Path("./tests/assets/data/20250220")
GROUP_NAME = "WT"


@pytest.fixture(scope="module")
def model_single_exp():
    config = Config(DATASET_1, rel_root_path="tests")
    model = Model()
    model.add_group(GROUP_NAME)
    model.create_experiment(config, GROUP_NAME)

    return model


@pytest.fixture(scope="module")
def model_two_exps():
    config = Config(DATASET_1, rel_root_path="tests")
    model = Model()
    model.add_group(GROUP_NAME)
    model.create_experiment(config, GROUP_NAME)

    config2 = Config(DATASET_2, rel_root_path="tests")
    model.create_experiment(config2, GROUP_NAME)

    return model


def test_initial_state_is_empty():
    model = Model()

    assert not any(
        [
            model.groups,
            model.curr_group,
            model.to_remove,
            model.curr_exp,
            model.curr_emb_name,
        ]
    )


def test_can_create_experiment(model_single_exp):
    assert model_single_exp is not None


def test_curr_group_matches_provided_group(model_single_exp):
    assert model_single_exp.curr_group == GROUP_NAME


def test_curr_experiment_matches_provided_exp(model_single_exp):
    curr_exp = model_single_exp.get_curr_experiment()
    assert curr_exp.name == DATASET_1.name


def test_reset_exp_resets_current_embryo(model_single_exp):
    first_emb_name = "emb1"
    next_emb_name = "emb3"
    model_single_exp.set_curr_emb(next_emb_name)
    model_single_exp.reset_current_experiment()
    assert model_single_exp.curr_emb_name != next_emb_name
    assert model_single_exp.curr_emb_name == first_emb_name


def test_can_create_group_with_two_experiments(model_two_exps):
    assert model_two_exps.has_combined_experiments()


def test_add_experiment_keeps_first_as_current(model_two_exps):
    curr_exp = model_two_exps.get_curr_experiment()
    assert curr_exp.name == DATASET_1.name


def test_add_group_keeps_current_group(model_two_exps):
    curr_group = model_two_exps.curr_group
    new_group = "Mutant"
    model_two_exps.add_group(new_group)
    assert len(model_two_exps.groups) == 2
    assert model_two_exps.curr_group != new_group
    assert model_two_exps.curr_group == curr_group
