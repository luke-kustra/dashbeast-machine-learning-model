import os

from src import logreg, gboost


def test_logistic_regression_train_and_predict(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(repo_root, 'data', 'workouts_multiclass.csv')

    label_map_path = tmp_path / 'label_map.json'
    model_path = tmp_path / 'logreg.joblib'

    metrics = logreg.train(
        data_path=data_path,
        label_map_path=str(label_map_path),
        model_path=str(model_path),
        val_split=0.2,
        seed=0,
    )
    assert model_path.exists()
    assert 'val_accuracy' in metrics

    results = logreg.predict(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        model_path=str(model_path),
        label_map_path=str(label_map_path),
        topk=2,
    )
    assert len(results) == 2
    assert isinstance(results[0][0], str)


def test_gradient_boosting_train_and_predict(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(repo_root, 'data', 'workouts_multiclass.csv')

    label_map_path = tmp_path / 'label_map.json'
    model_path = tmp_path / 'gboost.joblib'

    metrics = gboost.train(
        data_path=data_path,
        label_map_path=str(label_map_path),
        model_path=str(model_path),
        val_split=0.2,
        seed=0,
    )
    assert model_path.exists()
    assert 'val_accuracy' in metrics

    results = gboost.predict(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        model_path=str(model_path),
        label_map_path=str(label_map_path),
        topk=2,
    )
    assert len(results) == 2
    assert isinstance(results[0][0], str)
