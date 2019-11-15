import mlsurvey as mls


def main():
    # client = Client(processes=False, threads_per_worker=4,
    #                 n_workers=1, memory_limit='2GB')

    mlw = mls.sl.workflows.MultipleLearningWorkflow()
    mlw.run()


if __name__ == "__main__":
    main()
