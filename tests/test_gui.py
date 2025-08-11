from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from pasna_analysis import Experiment
from pasna_analysis.gui.gui import (
    ComparePlotWindow,
    ImageWindow,
    JsonViewer,
    MainWindow,
)
from pasna_analysis.gui.model import GroupModel, ExperimentModel

VALID_DIR = Path("./tests/assets/data/20250210")


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
        "pasna_analysis.gui.gui.QFileDialog.getExistingDirectory",
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
    window = ImageWindow(str(image_path))

    qtbot.addWidget(window)

    label = window.findChild(QLabel)

    assert label is not None, "Could not find QLabel."

    pixmap = label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()


def test_can_render_json_config(qtbot, exp):
    config_data = exp.config.data

    json_viewer = JsonViewer(config_data)
    qtbot.addWidget(json_viewer)

    with qtbot.waitSignal(json_viewer.update_config_signal) as signals:
        qtbot.mouseClick(json_viewer.save_btn, Qt.MouseButton.LeftButton)

    assert signals.args
    assert signals.args[0] == config_data
