from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner


import mlsurvey as mls
from mlsurvey.workflows.learning_workflow import LearningWorkflow


class MultipleLearningWorkflow(LearningWorkflow):

    def run(self):
        """
        Run the workflow : run each config
        """
        # data
        data_catalog = DataCatalog({'config': MemoryDataSet(),
                                    'log': MemoryDataSet(),
                                    'base_directory': MemoryDataSet()})
        data_catalog.save('config', self.config)
        data_catalog.save('log', self.log)
        data_catalog.save('base_directory', self.base_directory)

        expand_config_node = mls.sl.workflows.tasks.ExpandConfigTask.get_node()
        multiple_learning_node = mls.sl.workflows.tasks.MultipleLearningTask.get_node()

        # Assemble nodes into a pipeline
        pipeline = Pipeline([expand_config_node, multiple_learning_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        result = runner.run(pipeline, data_catalog)
        if len(result) == 0:
            self.terminate()

