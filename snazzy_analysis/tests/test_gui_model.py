from pathlib import Path
from snazzy_analysis.gui import Model

import pytest

from snazzy_analysis import Config

DATA_DIR = Path(__file__).parent.joinpath("assets", "data")
DATASET_1 = DATA_DIR.joinpath("20250210")
DATASET_2 = DATA_DIR.joinpath("20250220")
GROUP_NAME = "WT"


@pytest.fixture(scope="module")
def model_single_exp():
    config = Config(DATASET_1)
    model = Model()
    model.create_experiment(config, GROUP_NAME)

    return model


@pytest.fixture(scope="module")
def model_two_exps():
    config = Config(DATASET_1)
    model = Model()
    model.create_experiment(config, GROUP_NAME)

    config2 = Config(DATASET_2)
    model.create_experiment(config2, GROUP_NAME)

    return model


def test_initial_state_is_empty():
    model = Model()

    assert model.selected_experiment is None
    assert model.selected_embryo is None
    assert model.selected_group is None
    assert model.groups == []


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
    model_single_exp.toggle_emb_visibility(first_emb_name, should_remove=False)


def test_after_removing_emb_embryos_does_not_contain_it(model_single_exp):
    exp = model_single_exp.selected_experiment
    emb_names = [e.name for e in exp.embryos]
    first_emb_name = emb_names[0]
    model_single_exp.toggle_emb_visibility(first_emb_name, should_remove=True)
    emb_names = [e.name for e in exp.embryos]
    assert first_emb_name not in emb_names
    model_single_exp.toggle_emb_visibility(first_emb_name, should_remove=False)


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


def test_can_save_trim_index_in_config(model_single_exp):
    curr_emb = model_single_exp.selected_embryo
    updated_trim_idx = len(curr_emb.activity) // 2
    model_single_exp.save_trim_idx(updated_trim_idx)

    curr_exp = model_single_exp.selected_experiment

    assert curr_exp

    config = curr_exp.config

    assert config

    manual_data = config.get_corrected_peaks(curr_emb.name)

    assert manual_data["manual_trim_idx"] == updated_trim_idx


def test_can_add_peak(model_single_exp):
    new_peak_index = 300
    curr_trace = model_single_exp.selected_experiment.get_embryo("emb1").trace

    original_peaks_len = len(curr_trace.peak_idxes)

    _, new_peaks = model_single_exp.add_peak(new_peak_index, "emb1", curr_trace)

    assert len(new_peaks) == original_peaks_len + 1


def test_can_remove_peak(model_single_exp):
    emb_name = "emb1"
    curr_trace = model_single_exp.selected_experiment.get_embryo(emb_name).trace
    to_remove_index = curr_trace.peak_idxes[1]

    original_peaks_len = len(curr_trace.peak_idxes)

    _, new_peaks = model_single_exp.remove_peak(to_remove_index, emb_name, curr_trace)

    assert len(new_peaks) == original_peaks_len - 1

    config = model_single_exp.selected_experiment.config

    manual_data = config.get_corrected_peaks(emb_name)

    assert to_remove_index in manual_data["manual_remove"]


def test_can_add_and_remove_peak(model_single_exp):
    emb_name = "emb1"
    new_peak_index = 353
    curr_trace = model_single_exp.selected_experiment.get_embryo(emb_name).trace

    model_single_exp.add_peak(new_peak_index, emb_name, curr_trace)

    model_single_exp.calc_peaks_all_embs()

    model_single_exp.remove_peak(new_peak_index, emb_name, curr_trace)

    config = model_single_exp.selected_experiment.config

    manual_data = config.get_corrected_peaks(emb_name)

    assert new_peak_index not in manual_data["manual_peaks"]
    assert new_peak_index in manual_data["manual_remove"]


def test_can_clear_manual_data_from_single_emb(model_single_exp):
    emb_name = "emb1"
    new_peak_index = 400
    curr_trace = model_single_exp.selected_experiment.get_embryo(emb_name).trace

    model_single_exp.add_peak(new_peak_index, emb_name, curr_trace)

    another_emb_name = "emb3"
    another_trace = model_single_exp.selected_experiment.get_embryo(
        another_emb_name
    ).trace
    model_single_exp.add_peak(new_peak_index, another_emb_name, another_trace)

    assert "embryos" in model_single_exp.selected_experiment.config.data

    orig_config = model_single_exp.selected_experiment.config.data["embryos"]

    assert emb_name in orig_config

    model_single_exp.clear_manual_data_by_embryo(emb_name)

    assert curr_trace.to_add == []
    assert curr_trace.to_remove == []

    manual_data = model_single_exp.selected_experiment.config.data["embryos"]

    assert emb_name not in manual_data
    assert another_emb_name in manual_data


def test_can_clear_all_manual_data(model_single_exp):
    emb_name = "emb1"
    new_peak_index = 400
    curr_trace = model_single_exp.selected_experiment.get_embryo(emb_name).trace

    model_single_exp.add_peak(new_peak_index, emb_name, curr_trace)

    another_emb_name = "emb3"
    another_trace = model_single_exp.selected_experiment.get_embryo(
        another_emb_name
    ).trace
    model_single_exp.add_peak(new_peak_index, another_emb_name, another_trace)

    assert "embryos" in model_single_exp.selected_experiment.config.data

    model_single_exp.clear_all_manual_data()

    manual_data = model_single_exp.selected_experiment.config.data["embryos"]

    assert emb_name not in manual_data
    assert another_emb_name not in manual_data

    assert manual_data == {}


def test_clear_emb_manual_data_raises_if_emb_not_exists(model_single_exp):
    emb_name = "emb1"
    new_peak_index = 400
    curr_trace = model_single_exp.selected_experiment.get_embryo(emb_name).trace

    model_single_exp.add_peak(new_peak_index, emb_name, curr_trace)

    with pytest.raises(ValueError):
        model_single_exp.clear_manual_data_by_embryo("emb2")


def test_clear_emb_manual_data_works_when_no_saved_data(model_single_exp):
    emb_name = "emb1"
    curr_trace = model_single_exp.selected_experiment.get_embryo(emb_name).trace

    assert curr_trace.to_add == []
    assert curr_trace.to_remove == []

    model_single_exp.clear_manual_data_by_embryo("emb1")

    assert curr_trace.to_remove == []
