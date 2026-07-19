import numpy as np
import pandas as pd
import pytest

from cognitivecyber.data.loaders import load_dataset
from cognitivecyber.data.preprocessing import clean_and_encode, preprocess, split_and_scale
from cognitivecyber.data.schemas import SCHEMAS, detect_schema
from cognitivecyber.data.synthetic import generate_synthetic_flows


def test_synthetic_generator_shape_and_labels():
    df = generate_synthetic_flows(n_samples=5000, attack_ratio=0.3, random_state=1)
    assert len(df) > 0
    assert set(df["label"].unique()) <= {0, 1}
    assert df["is_synthetic"].all()
    assert 0.2 < df["label"].mean() < 0.45


def test_synthetic_generator_reproducible():
    df1 = generate_synthetic_flows(n_samples=1000, random_state=7)
    df2 = generate_synthetic_flows(n_samples=1000, random_state=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_dataset_synthetic_default():
    df, meta = load_dataset(n_synthetic=1000, random_state=3)
    assert meta["is_synthetic"] is True
    assert meta["label_column"] in df.columns


def test_load_dataset_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset(path="/nonexistent/dataset.csv")


def test_detect_schema_unsw():
    cols = ["id", "dur", "proto", "service", "state", "label", "attack_cat"]
    schema = detect_schema(cols)
    assert schema is not None
    assert schema.name == "UNSW-NB15"


def test_detect_schema_no_match_returns_none():
    schema = detect_schema(["foo", "bar", "baz"])
    assert schema is None


def test_clean_and_encode_no_nans_no_object_dtypes():
    df, meta = load_dataset(n_synthetic=2000, random_state=5)
    X, y = clean_and_encode(df, meta)
    assert X.isna().sum().sum() == 0
    assert X.select_dtypes(exclude=[np.number]).shape[1] == 0
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(df)


def test_split_and_scale_shapes_and_stratification():
    df, meta = load_dataset(n_synthetic=4000, random_state=9)
    X, y = clean_and_encode(df, meta)
    pdata = split_and_scale(X, y, test_size=0.2, val_size=0.1, random_state=9)

    n = len(X)
    assert pdata.X_train.shape[0] == len(pdata.y_train)
    assert pdata.X_val.shape[0] == len(pdata.y_val)
    assert pdata.X_test.shape[0] == len(pdata.y_test)
    total = pdata.X_train.shape[0] + pdata.X_val.shape[0] + pdata.X_test.shape[0]
    assert total == n

    # Stratification: class ratio roughly preserved across splits (loose tolerance)
    overall_ratio = y.mean()
    for split_y in (pdata.y_train, pdata.y_val, pdata.y_test):
        assert abs(split_y.mean() - overall_ratio) < 0.05


def test_split_and_scale_scaler_applied():
    df, meta = load_dataset(n_synthetic=3000, random_state=11)
    X, y = clean_and_encode(df, meta)
    pdata = split_and_scale(X, y, random_state=11)
    # StandardScaler -> train features should be roughly zero-mean, unit-variance
    means = pdata.X_train.mean(axis=0)
    stds = pdata.X_train.std(axis=0)
    assert np.allclose(means, 0, atol=0.3)
    assert np.allclose(stds, 1, atol=0.3)


def test_preprocess_end_to_end():
    df, meta = load_dataset(n_synthetic=2000, random_state=13)
    pdata = preprocess(df, meta, random_state=13)
    assert pdata.X_train.shape[1] == len(pdata.feature_names)
    assert pdata.X_val.shape[1] == pdata.X_train.shape[1]
    assert pdata.X_test.shape[1] == pdata.X_train.shape[1]


def test_all_registered_schemas_have_required_fields():
    for name, schema in SCHEMAS.items():
        assert schema.label_column
        assert isinstance(schema.categorical_columns, list)
        assert isinstance(schema.drop_columns, list)
        assert isinstance(schema.binary_positive_labels, list)
