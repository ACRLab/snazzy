from pathlib import Path
import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from snazzy_analysis import Experiment
from snazzy_analysis.gui.gui import (
    ComparePlotWindow,
    ExperimentParamsDialog,
    ImageWindow,
    JsonViewer,
    MainWindow,
)
from snazzy_analysis.gui.model import GroupModel, ExperimentModel

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Running headless tests on linux only."
)

VALID_DIR = Path(__file__).parent.joinpath("assets", "data", "20250210")


@pytest.fixture
def exp():
    return Experiment(VALID_DIR)


@pytest.fixture
def group_model(exp):
    exp_model = ExperimentModel(exp)
    group_model = GroupModel("wt")
    group_model.add_experiment(exp_model)
    return group_model


def test_can_display_initial_screen(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.show()

    assert window.isVisible()


def test_can_create_experiment_via_menu(qtbot, monkeypatch):
    monkeypatch.setattr(
        "snazzy_analysis.gui.gui.QFileDialog.getExistingDirectory",
        lambda *a, **kw: VALID_DIR,
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    # first action is the 'File' menu and first action in there is 'Open Directory'
    actions = window.menuBar().actions()
    file_menu = actions[0].menu()
    open_action = file_menu.actions()[0]
    open_action.trigger()

    window.exp_params_dialog.accept()

    qtbot.waitUntil(lambda: window.add_experiment_action.isEnabled(), timeout=3000)

    assert window.add_experiment_action.isEnabled()
    assert window.top_app_bar is not None
    assert window.bottom_layout is not None
    # for VALID_DIR dataset, emb1 is the first embryo:
    assert window.model.selected_embryo.name == "emb1"


def test_exp_dialog_parses_emb_ids(qtbot):
    props = {"to_remove": ["emb1", "emb2"]}
    exp_dialog = ExperimentParamsDialog(props)
    qtbot.addWidget(exp_dialog)

    assert props["to_remove"] == [1, 2]


def test_exp_dialog_parses_emb_ids_when_receives_only_digits(qtbot):
    props = {"to_remove": ["1", "2"]}
    exp_dialog = ExperimentParamsDialog(props)
    qtbot.addWidget(exp_dialog)

    assert props["to_remove"] == [1, 2]


def test_can_generate_plots(qtbot, group_model):
    window = ComparePlotWindow([group_model])
    qtbot.addWidget(window)

    for plt_fn in window.btns.values():
        plt_fn()
        assert window.ax


def test_when_save_dir_not_provided_compare_plot_raises(qtbot, group_model):
    window = ComparePlotWindow([group_model])
    qtbot.addWidget(window)

    assert window.btns, "Could not find btn dictionary."
    plt_fn = next(iter(window.btns.values()))

    with pytest.raises(ValueError):
        plt_fn(save=True, save_dir=None)


def test_can_render_FOV(qtbot):
    image_path = VALID_DIR / "emb_numbers.png"
    exp_name = VALID_DIR.stem
    window = ImageWindow(exp_name=exp_name, image_path=str(image_path))

    qtbot.addWidget(window)

    labels = window.findChildren(QLabel)
    assert len(labels) >= 2

    pixmap = labels[1].pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()


def test_can_render_json_config(qtbot, exp):
    config_data = exp.config.data
    config_data["exp_params"]["to_remove"] = ["emb1", "emb2"]

    json_viewer = JsonViewer(config_data)
    qtbot.addWidget(json_viewer)

    with qtbot.waitSignal(json_viewer.update_config_signal) as signals:
        qtbot.mouseClick(json_viewer.save_btn, Qt.MouseButton.LeftButton)

    assert signals.args
    assert signals.args[0] == config_data
