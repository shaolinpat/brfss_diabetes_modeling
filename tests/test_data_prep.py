# tests/test_data_prep.py

from flower_classifier.data_prep import load_iris_data


def test_load_iris_data():
    df = load_iris_data()
    # 150 rows, at least these columns:
    assert len(df) == 150
    for col in [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]:
        assert col in df.columns


def test_pipeline_runs():
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = load_iris_data()
    X = df.drop(columns=["species"])
    y = df["species"]
    X_train, _, y_train, _ = train_test_split(X, y, random_state=42)

    pipe = Pipeline([("scale", StandardScaler()), ("clf", RandomForestClassifier())])
    pipe.fit(X_train, y_train)
    assert hasattr(pipe, "predict")

