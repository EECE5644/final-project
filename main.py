"""Entry point for the model comparison on 20 Newsgroups."""

import runner
from models import MODELS, ModelSpec, build
from preprocessing import load_dataset
from vectorizers import VECTORIZERS, Method

MATRIX_METHODS = tuple(m for m in VECTORIZERS if m is not Method.SEQUENCE)


def _methods(spec: ModelSpec) -> tuple[Method, ...]:
    if spec.default_method is Method.SEQUENCE:
        return (Method.SEQUENCE,)

    return MATRIX_METHODS


def main() -> None:
    train_dataset, test_dataset = load_dataset()
    print(
        f"Train docs: {len(train_dataset)}, Test docs: {len(test_dataset)},",
        f"Classes: {len(train_dataset.target_names)}",
    )

    experiments = [
        build(name, method)
        for name, spec in MODELS.items()
        for method in _methods(spec)
    ]
    print(f"{len(experiments)} experiments queued\n")

    runner.run_all(experiments, train_dataset, test_dataset, plot=True)


if __name__ == "__main__":
    main()
