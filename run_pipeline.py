import click
from pipelines.training_pipeline import ml_pipeline


@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    ml_pipeline.configure(enable_cache=False)

    result = ml_pipeline(file_path="/home/sanjaylinux/hpp/data/AmesHousing.csv")

    # The pipeline call may return either the step outputs or a pipeline
    # run response depending on the ZenML version. Handle both cases.
    if isinstance(result, tuple) and len(result) == 2:
        evaluation_metrics, mse = result
        print(f"Evaluation Metrics: {evaluation_metrics}")
        print(f"MSE: {mse}")
    else:
        print("Pipeline finished. Result object:")
        print(result)

    print(
        "\n✓ Pipeline completed successfully!\n"
        "To view experiment metrics, run:\n"
        "    mlflow ui\n"
        "Then open http://localhost:5000 in your browser."
    )


if __name__ == "__main__":
    main()
