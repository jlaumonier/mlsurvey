import os

from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import SequentialRunner


import mlsurvey as mls
from mlsurvey.workflows.learning_workflow import LearningWorkflow


class SupervisedLearningWorkflow(LearningWorkflow):

    @staticmethod
    def visualize_class():
        return mls.sl.visualize.VisualizeLogSL

    def run(self):
        """
        Run all tasks
        """
        # data
        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'dataset': MemoryDataSet(),
                                    'data': MemoryDataSet()})
        data_catalog.save('config', self.config)
        data_catalog.save('log', self.log)

        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        self.terminate()
