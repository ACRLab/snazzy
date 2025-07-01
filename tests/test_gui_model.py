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
    model.create_experiment(config, GROUP_NAME)

    return model


@pytest.fixture(scope="module")
def model_two_exps():
    config = Config(DATASET_1, rel_root_path="tests")
    model = Model()
    model.create_experiment(config, GROUP_NAME)

    config2 = Config(DATASET_2, rel_root_path="tests")
    model.create_experiment(config2, GROUP_NAME)

    return model


def test_initial_state_is_empty():
    model = Model()

    assert not any(
        [
            model.groups,
            model.selected_experiment,
            model.selected_group,
        ]
    )


def test_can_create_experiment(model_single_exp):
    assert model_single_exp is not None


def test_creating_experiment_also_creates_group_if_needed(model_single_exp):
    assert model_single_exp.selected_group.name == GROUP_NAME


def test_can_add_experiment_to_existing_group(model_two_exps):
    assert len(model_two_exps.groups) == 1
    assert model_two_exps.selected_group.name == GROUP_NAME


def test_selected_experiment_matches_provided_exp(model_single_exp):
    assert model_single_exp.selected_experiment.name == DATASET_1.name


def test_embs_marked_as_removed_are_synced_with_experiment(model_single_exp):
    exp = model_single_exp.selected_experiment
    to_remove = exp.exp_params.get("to_remove", [])
    assert len(to_remove) > 0
    for emb_name in to_remove:
        assert emb_name in model_single_exp.selected_experiment.to_remove


def test_reset_exp_resets_current_embryo(model_single_exp):
    first_emb_name = "emb1"
    next_emb_name = "emb3"
    next_emb = model_single_exp.selected_experiment.get_embryo(next_emb_name)
    model_single_exp.select_embryo(next_emb)
    model_single_exp.reset_current_experiment()
    emb = model_single_exp.selected_experiment.selected_embryo
    assert emb.name != next_emb_name
    assert emb.name == first_emb_name


def test_toggle_twice_adds_embryo_back(model_single_exp):
    exp = model_single_exp.selected_experiment
    first_emb_name = "emb1"
    model_single_exp.toggle_emb_visibility(first_emb_name, should_remove=True)
    model_single_exp.toggle_emb_visibility(first_emb_name, should_remove=False)
    emb_names = [e.name for e in exp.embryos]
    assert first_emb_name in emb_names


def test_can_remove_embryo(model_single_exp):
    exp = model_single_exp.selected_experiment
    first_emb_name = "emb1"
    model_single_exp.toggle_emb_visibility(first_emb_name, should_remove=True)
    assert first_emb_name in exp.to_remove


def test_after_removing_emb_embryos_does_not_contain_it(model_single_exp):
    exp = model_single_exp.selected_experiment
    emb_names = [e.name for e in exp.embryos]
    first_emb_name = emb_names[0]
    model_single_exp.toggle_emb_visibility(first_emb_name, should_remove=True)
    emb_names = [e.name for e in exp.embryos]
    assert first_emb_name not in emb_names


def test_can_create_group_with_two_experiments(model_two_exps):
    assert model_two_exps.has_combined_experiments()


def test_add_experiment_selects_most_recent_exp(model_two_exps):
    assert model_two_exps.selected_experiment.name == DATASET_2.name


def test_add_group_keeps_current_group(model_two_exps):
    prev_name = model_two_exps.selected_group.name
    new_group = "Mutant"
    model_two_exps.create_group(new_group)
    assert len(model_two_exps.groups) == 2
    assert model_two_exps.selected_group.name == prev_name
    assert model_two_exps.selected_group.name != new_group
