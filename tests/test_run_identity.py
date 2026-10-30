"""Different experiments should not overwrite each other's measurements."""

from resnet_research.utils.config import load_config, research_signature, resolve_run_name


def test_shortcut_changes_run_identity():
    a = load_config(["configs/models/resnet_56.yaml"])
    b = load_config(["configs/models/resnet_56.yaml"], overrides=["model.shortcut=B"])
    assert research_signature(a) != research_signature(b)
    assert resolve_run_name(a) != resolve_run_name(b)


def test_hyperparameter_changes_run_identity():
    a = load_config(["configs/models/resnet_20.yaml"])
    b = load_config(["configs/models/resnet_20.yaml"], overrides=["training.lr=0.01"])
    c = load_config(["configs/models/resnet_20.yaml"], overrides=["data.augment=minimal"])
    assert len({resolve_run_name(a), resolve_run_name(b), resolve_run_name(c)}) == 3


def test_output_path_not_part_of_identity():
    a = load_config(["configs/models/resnet_20.yaml"])
    b = load_config(["configs/models/resnet_20.yaml"], overrides=["logging.out_dir=tmp"])
    assert research_signature(a) == research_signature(b)
