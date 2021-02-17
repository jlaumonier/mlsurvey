import mlsurvey as mls


def main():
    mlw = mls.sl.workflows.MultipleLearningWorkflow(mlflow_log=True)
    mlw.run()


if __name__ == "__main__":
    main()
