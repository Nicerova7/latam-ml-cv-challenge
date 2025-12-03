import os
import glob
import yaml
import random
import pytest
from pathlib import Path

# Note: I could have created a separate test-specific YAML, 
# but normalizing paths in code keeps everything consistent without duplicating config files.
BASE_DIR = Path(__file__).resolve().parent  # we are in tests/ directory
DATA_YAML_PATH = (BASE_DIR / "../data/data.yaml").resolve()


def _load_data_cfg():
    assert os.path.exists(DATA_YAML_PATH)
    with open(DATA_YAML_PATH, "r") as f:
        data_cfg = yaml.safe_load(f)

    # --- normalize paths inside the YAML ---
    base_dir = Path(DATA_YAML_PATH).resolve().parent # we are in data/
    for key in ("train", "val", "test"):
        if key in data_cfg and isinstance(data_cfg[key], str): # check it's a path
            data_cfg[key] = (base_dir / data_cfg[key]).resolve().as_posix()

    return data_cfg


def _collect_split_images(img_dir: str):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(paths)


def _img_to_label(img_path: str) -> str:
    lbl = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    lbl = os.path.splitext(lbl)[0] + ".txt"
    return lbl


@pytest.fixture(scope="session")
def data_cfg():
    return _load_data_cfg()


@pytest.fixture(scope="session")
def class_names(data_cfg):
    return data_cfg.get("names", [])


@pytest.fixture(scope="session")
def test_images(data_cfg):
    test_dir = data_cfg.get("test") or data_cfg.get("val")
    assert test_dir
    imgs = _collect_split_images(test_dir)
    assert imgs
    k = int(os.environ.get("TEST_SAMPLE_SIZE", 6))
    random.seed(7)
    return random.sample(imgs, min(k, len(imgs)))


@pytest.fixture()
def yolo_labels():
    def _read(label_path: str):
        rows = []
        if not os.path.exists(label_path):
            return rows
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    rows.append((cls_id, cx, cy, w, h))
        return rows

    return _read


@pytest.fixture()
def to_label_path():
    def _fn(img_path: str) -> str:
        return _img_to_label(img_path)

    return _fn
