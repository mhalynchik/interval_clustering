"""Datasets and published method names for the September revision."""
from pathlib import Path
import csv, random
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from Algs.weights import A0, A1, A2
ROOT = Path(__file__).resolve().parents[1]
WEIGHT_FUNCS = {"A0": A0, "A1": A1, "A2": A2}
PREPROCESS = dict(center_data=True, minmax_normilize=True, scale_by_var=False)
BETA = 1.0
BANCO_ER = 1e-10
PLUS_SEED = 42
MINUS_SEED0 = 42
HIT_TOL = 1e-12
DATA_ROOT = ROOT / "subdatasets"
NEW_DATA_ROOT = ROOT / "experiments" / "data" / "new"
LETTER_DIR = ROOT / "letter"
TABLES_DIR = LETTER_DIR / "tables"

# Datasets whose interval matrix ships as data.csv (comma-separated intervals)
# instead of data.npy, and which live under NEW_DATA_ROOT.
CSV_DATASETS = {"Iris"}

PAPER_DATASETS = (
    ("Fungi_genera_3", "Fungi_3"),
    ("Fungi_genera_4", "Fungi_4"),
    ("Fungi_genera_5", "Fungi_5"),
    ("BSP_GAP_3", "BSP_3"),
    ("BSP_GAP_4", "BSP_4"),
    ("BSP_GAP_5", "BSP_5"),
    ("Iris", "Iris"),
)
PAPER_COLS = [short for _folder, short in PAPER_DATASETS]

# BANCO letter n/c/o = A0/A1/A2. KM letter n/c/o/cs/os as below. cs/os => 3d.
PLUS_SAME = (
    ("BIKMn", "A0", "IKM", "A0", "2d"),
    ("BIKMc", "A1", "IKM", "A1", "2d"),
    ("BIKMo", "A2", "IKM", "A2", "2d"),
    ("BIKMcs", "A1", "IKM", "A1", "3d"),
    ("BIKMos", "A2", "IKM", "A2", "3d"),
    ("BKMn", "A0", "KM", "A0", "2d"),
    ("BKMc", "A1", "KM", "A1", "2d"),
    ("BKMcs", "A1", "KM", "A1", "3d"),
)
PLUS_MIXED = (
    ("BnIKMc", "A0", "IKM", "A1", "2d"),
    ("BnIKMo", "A0", "IKM", "A2", "2d"),
    ("BnIKMcs", "A0", "IKM", "A1", "3d"),
    ("BnIKMos", "A0", "IKM", "A2", "3d"),
    ("BcIKMn", "A1", "IKM", "A0", "2d"),
    ("BcIKMo", "A1", "IKM", "A2", "2d"),
    ("BcIKMos", "A1", "IKM", "A2", "3d"),
    ("BoIKMn", "A2", "IKM", "A0", "2d"),
    ("BoIKMc", "A2", "IKM", "A1", "2d"),
    ("BoIKMcs", "A2", "IKM", "A1", "3d"),
    ("BnKMc", "A0", "KM", "A1", "2d"),
    ("BnKMcs", "A0", "KM", "A1", "3d"),
    ("BcKMn", "A1", "KM", "A0", "2d"),
    ("BoKMn", "A2", "KM", "A0", "2d"),
    ("BoKMc", "A2", "KM", "A1", "2d"),
    ("BoKMcs", "A2", "KM", "A1", "3d"),
)
MINUS_SPECS = (
    ("IKMn", "IKM", "A0", "2d"),
    ("IKMc", "IKM", "A1", "2d"),
    ("IKMo", "IKM", "A2", "2d"),
    ("IKMcs", "IKM", "A1", "3d"),
    ("IKMos", "IKM", "A2", "3d"),
    ("KMn", "KM", "A0", "2d"),
    ("KMc", "KM", "A1", "2d"),
    ("KMcs", "KM", "A1", "3d"),
)
PLUS_ALL = PLUS_SAME + PLUS_MIXED


def load_clusters(path: Path) -> list[list[int]]:
    clusters = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Preserve duplicates until labels_from_clusters validates membership.
        clusters.append([int(x) for x in line.split(",")])
    return clusters


from Algs.partitions import labels_from_clusters


def _intervals_from_csv(path: Path) -> np.ndarray:
    """Read a comma-separated interval table (index,label,feat=[lo,hi],...) into
    the [lows..., highs...] layout used by the interval algorithms."""
    import ast

    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Missing interval CSV header")
        feature_cols = [c for c in reader.fieldnames if c not in ("index", "label")]
        for record in reader:
            lows: list[float] = []
            highs: list[float] = []
            for col in feature_cols:
                lo, hi = ast.literal_eval(record[col])
                lows.append(float(lo))
                highs.append(float(hi))
            rows.append(lows + highs)
    return np.asarray(rows, dtype=np.float64)


def dataset_folder(folder_name: str) -> Path:
    if folder_name in CSV_DATASETS:
        return NEW_DATA_ROOT / folder_name
    return DATA_ROOT / folder_name


def load_dataset(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = Path(folder)
    if folder.name in CSV_DATASETS:
        data = _intervals_from_csv(folder / "data.csv")
    else:
        data = np.load(folder / "data.npy")
    clusters = load_clusters(folder / "real_clusters.txt")
    return data, labels_from_clusters(clusters, data.shape[0])


def idxs_for_weights(weight_name: str, n_cols: int, kind: str = 'IKM') -> list[np.ndarray]:
    """Interval simplex constraints are separate for the two endpoint groups.

    Doubled-coordinate KM uses one simplex; A2 has one product constraint
    across both groups in either representation.
    """
    if weight_name not in WEIGHT_FUNCS or kind not in ('IKM', 'KM', 'BANCO'):
        raise ValueError('Unknown weighting scheme or algorithm')
    if n_cols < 1:
        raise ValueError('At least one coordinate is required')
    half = n_cols // 2
    if weight_name == "A2" or (weight_name == 'A1' and kind != 'KM'):
        if n_cols % 2:
            raise ValueError('Interval weighting requires paired endpoint columns')
        return [np.arange(half), np.arange(half, n_cols)]
    return [np.arange(n_cols)]


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def score_partition(y_true: np.ndarray, clusters: list[set[int]]) -> tuple[float, float]:
    y_pred = labels_from_clusters(clusters, y_true.size)
    return (
        float(adjusted_rand_score(y_true, y_pred)),
        float(normalized_mutual_info_score(y_true, y_pred, average_method='max')),
    )
