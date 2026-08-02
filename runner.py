"""
Drives an experiment end to end: CV, fit, score, and optionally plot.

Accepts anything satisfying `ptypes.Classifier` over documents, so one code path serves a
sklearn Pipeline and a torch estimator alike. `param_grid=None` fits once without searching.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

import evaluator
import plotter
from evaluator import Metrics
from preprocessing import Dataset
from ptypes import Classifier, Experiment, ParamGrid


@dataclass(frozen=True)
class Result:
    """Everything one experiment produced, including the fitted estimator."""

    name: str
    metrics: Metrics
    estimator: Classifier
    best_params: dict | None = None
    cv_score: float | None = None


def run(
    estimator: Classifier,
    train: Dataset,
    test: Dataset,
    *,
    name: str,
    param_grid: ParamGrid | None = None,
    cv: int = 5,
    n_jobs: int = -1,
    plot: bool = False,
) -> Result:
    """
    Fit on `train`, searching `param_grid` if one is given, then score on `test`.

    The vectorizer is a step of `estimator`, so it is refit inside every fold -- no
    validation data ever leaks into the vocabulary or the IDF weights.
    """
    if param_grid:
        assert isinstance(estimator, BaseEstimator)
        search = GridSearchCV(estimator, param_grid, cv=cv, n_jobs=n_jobs)
        search.fit(train.documents, train.targets)
        fitted = search.best_estimator_
        best_params, cv_score = search.best_params_, float(search.best_score_)
    else:
        estimator.fit(train.documents, train.targets)
        fitted = estimator
        best_params, cv_score = None, None

    metrics = evaluator.evaluate(fitted, test.documents, test.targets)

    if plot:
        plotter.plot_confusion_matrix(
            fitted, test.documents, test.targets, test.target_names, name=name
        )

    return Result(name, metrics, fitted, best_params, cv_score)


def run_all(
    experiments: Iterable[Experiment],
    train: Dataset,
    test: Dataset,
    *,
    cv: int = 5,
    plot: bool = False,
) -> list[Result]:
    """`run` each experiment in turn, reporting as it goes."""
    results = []
    for experiment in experiments:
        print(f"running {experiment.name} ...", flush=True)
        result = run(
            experiment.estimator,
            train,
            test,
            name=experiment.name,
            param_grid=experiment.param_grid,
            n_jobs=experiment.n_jobs,
            cv=cv,
            plot=plot,
        )
        print(f"  {summarize(result)}", flush=True)
        results.append(result)

    return results


def summarize(result: Result) -> str:
    """One-line metrics summary of the given `Result`."""
    metrics = result.metrics
    cv = "" if result.cv_score is None else f"CV={result.cv_score:.4f}, "

    return (
        f"{cv}Accuracy={metrics.accuracy:.2%}, Precision={metrics.precision:.4f}, "
        f"Recall={metrics.recall:.4f}, F1={metrics.f1:.4f}"
    )
