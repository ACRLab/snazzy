from pathlib import Path
from snazzy_analysis.gui import Model

import pytest

from snazzy_analysis import Config

DATA_DIR = Path(__file__).parent.joinpath("assets", "data")
DATASET_1 = DATA_DIR.joinpath("20250210")
DATASET_2 = DATA_DIR.joinpath("20250220")
GROUP_NAME = "WT"


@pytest.fixture(scope="module")
def model_single_dataset():
    config = Config(DATASET_1)
    model = Model()
    model.create_dataset(config, GROUP_NAME)

    return model


@pytest.fixture(scope="module")
def model_two_datasets():
    config = Config(DATASET_1)
    model = Model()
    model.create_dataset(config, GROUP_NAME)

    config2 = Config(DATASET_2)
    model.create_dataset(config2, GROUP_NAME)

    return model


def test_initial_state_is_empty():
    model = Model()

    assert model.selected_dataset is None
    assert model.selected_embryo is None
    assert model.selected_group is None
    assert model.groups == []


def test_can_create_dataset(model_single_dataset):
    assert model_single_dataset is not None


def test_creating_dataset_also_creates_group_if_needed(model_single_dataset):
    assert model_single_dataset.selected_group.name == GROUP_NAME


def test_can_add_dataset_to_existing_group(model_two_datasets):
    assert len(model_two_datasets.groups) == 1
    assert model_two_datasets.selected_group.name == GROUP_NAME


def test_selected_dataset_matches_provided_dataset(model_single_dataset):
    assert model_single_dataset.selected_dataset.name == DATASET_1.name


def test_embs_marked_as_removed_are_synced_with_dataset(model_single_dataset):
    dataset = model_single_dataset.selected_dataset
    to_remove = dataset.exp_params.get("to_remove", [])
    assert len(to_remove) > 0
    for emb_name in to_remove:
        assert emb_name in model_single_dataset.selected_dataset.to_remove


def test_reset_dataset_resets_current_embryo(model_single_dataset):
    first_emb_name = "emb1"
    next_emb_name = "emb3"
    next_emb = model_single_dataset.selected_dataset.get_embryo(next_emb_name)
    model_single_dataset.select_embryo(next_emb)
    model_single_dataset.reset_current_dataset()
    emb = model_single_dataset.selected_dataset.selected_embryo
    assert emb.name != next_emb_name
    assert emb.name == first_emb_name


def test_toggle_twice_adds_embryo_back(model_single_dataset):
    dataset = model_single_dataset.selected_dataset
    first_emb_name = "emb1"
    model_single_dataset.toggle_emb_visibility(first_emb_name, should_remove=True)
    model_single_dataset.toggle_emb_visibility(first_emb_name, should_remove=False)
    emb_names = [e.name for e in dataset.embryos]
    assert first_emb_name in emb_names


def test_can_remove_embryo(model_single_dataset):
    dataset = model_single_dataset.selected_dataset
    first_emb_name = "emb1"
    model_single_dataset.toggle_emb_visibility(first_emb_name, should_remove=True)
    assert first_emb_name in dataset.to_remove
    model_single_dataset.toggle_emb_visibility(first_emb_name, should_remove=False)


def test_after_removing_emb_embryos_does_not_contain_it(model_single_dataset):
    dataset = model_single_dataset.selected_dataset
    emb_names = [e.name for e in dataset.embryos]
    first_emb_name = emb_names[0]
    model_single_dataset.toggle_emb_visibility(first_emb_name, should_remove=True)
    emb_names = [e.name for e in dataset.embryos]
    assert first_emb_name not in emb_names
    model_single_dataset.toggle_emb_visibility(first_emb_name, should_remove=False)


def test_can_create_group_with_two_datasets(model_two_datasets):
    assert model_two_datasets.has_combined_datasets()


def test_add_dataset_selects_most_recent_dataset(model_two_datasets):
    assert model_two_datasets.selected_dataset.name == DATASET_2.name


def test_add_group_keeps_current_group(model_two_datasets):
    prev_name = model_two_datasets.selected_group.name
    new_group = "Mutant"
    model_two_datasets.create_group(new_group)
    assert len(model_two_datasets.groups) == 2
    assert model_two_datasets.selected_group.name == prev_name
    assert model_two_datasets.selected_group.name != new_group


def test_can_save_trim_index_in_config(model_single_dataset):
    curr_emb = model_single_dataset.selected_embryo
    updated_trim_idx = len(curr_emb.activity) // 2
    model_single_dataset.save_trim_idx(updated_trim_idx)

    curr_dataset = model_single_dataset.selected_dataset

    assert curr_dataset

    config = curr_dataset.config

    assert config

    manual_data = config.get_corrected_peaks(curr_emb.name)

    assert manual_data["manual_trim_idx"] == updated_trim_idx


def test_can_add_peak(model_single_dataset):
    new_peak_index = 300
    curr_trace = model_single_dataset.selected_dataset.get_embryo("emb1").trace

    original_peaks_len = len(curr_trace.peak_idxes)

    _, new_peaks = model_single_dataset.add_peak(new_peak_index, "emb1", curr_trace)

    assert len(new_peaks) == original_peaks_len + 1


def test_can_remove_peak(model_single_dataset):
    emb_name = "emb1"
    curr_trace = model_single_dataset.selected_dataset.get_embryo(emb_name).trace
    to_remove_index = curr_trace.peak_idxes[1]

    original_peaks_len = len(curr_trace.peak_idxes)

    _, new_peaks = model_single_dataset.remove_peak(
        to_remove_index, emb_name, curr_trace
    )

    assert len(new_peaks) == original_peaks_len - 1

    config = model_single_dataset.selected_dataset.config

    manual_data = config.get_corrected_peaks(emb_name)

    assert to_remove_index in manual_data["manual_remove"]


def test_can_add_and_remove_peak(model_single_dataset):
    emb_name = "emb1"
    new_peak_index = 353
    curr_trace = model_single_dataset.selected_dataset.get_embryo(emb_name).trace

    model_single_dataset.add_peak(new_peak_index, emb_name, curr_trace)

    model_single_dataset.calc_peaks_all_embs()

    model_single_dataset.remove_peak(new_peak_index, emb_name, curr_trace)

    config = model_single_dataset.selected_dataset.config

    manual_data = config.get_corrected_peaks(emb_name)

    assert new_peak_index not in manual_data["manual_peaks"]
    assert new_peak_index in manual_data["manual_remove"]


def test_can_clear_manual_data_from_single_emb(model_single_dataset):
    emb_name = "emb1"
    new_peak_index = 400
    curr_trace = model_single_dataset.selected_dataset.get_embryo(emb_name).trace

    model_single_dataset.add_peak(new_peak_index, emb_name, curr_trace)

    another_emb_name = "emb3"
    another_trace = model_single_dataset.selected_dataset.get_embryo(
        another_emb_name
    ).trace
    model_single_dataset.add_peak(new_peak_index, another_emb_name, another_trace)

    assert "embryos" in model_single_dataset.selected_dataset.config.data

    orig_config = model_single_dataset.selected_dataset.config.data["embryos"]

    assert emb_name in orig_config

    model_single_dataset.clear_manual_data_by_embryo(emb_name)

    assert curr_trace.to_add == []
    assert curr_trace.to_remove == []

    manual_data = model_single_dataset.selected_dataset.config.data["embryos"]

    assert emb_name not in manual_data
    assert another_emb_name in manual_data


def test_can_clear_all_manual_data(model_single_dataset):
    emb_name = "emb1"
    new_peak_index = 400
    curr_trace = model_single_dataset.selected_dataset.get_embryo(emb_name).trace

    model_single_dataset.add_peak(new_peak_index, emb_name, curr_trace)

    another_emb_name = "emb3"
    another_trace = model_single_dataset.selected_dataset.get_embryo(
        another_emb_name
    ).trace
    model_single_dataset.add_peak(new_peak_index, another_emb_name, another_trace)

    assert "embryos" in model_single_dataset.selected_dataset.config.data

    model_single_dataset.clear_all_manual_data()

    manual_data = model_single_dataset.selected_dataset.config.data["embryos"]

    assert emb_name not in manual_data
    assert another_emb_name not in manual_data

    assert manual_data == {}


def test_clear_emb_manual_data_raises_if_emb_not_exists(model_single_dataset):
    emb_name = "emb1"
    new_peak_index = 400
    curr_trace = model_single_dataset.selected_dataset.get_embryo(emb_name).trace

    model_single_dataset.add_peak(new_peak_index, emb_name, curr_trace)

    with pytest.raises(ValueError):
        model_single_dataset.clear_manual_data_by_embryo("emb2")


def test_clear_emb_manual_data_works_when_no_saved_data(model_single_dataset):
    emb_name = "emb1"
    curr_trace = model_single_dataset.selected_dataset.get_embryo(emb_name).trace

    assert curr_trace.to_add == []
    assert curr_trace.to_remove == []

    model_single_dataset.clear_manual_data_by_embryo("emb1")

    assert curr_trace.to_remove == []
