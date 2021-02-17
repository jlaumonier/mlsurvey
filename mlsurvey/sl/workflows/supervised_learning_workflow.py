from kedro.io import DataCatalog, MemoryDataSet
from kedro.pipeline import Pipeline
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
                                    'base_directory': MemoryDataSet(),
                                    'dataset': MemoryDataSet(),
                                    'data': MemoryDataSet()})
        data_catalog.save('config', self.config)
        data_catalog.save('log', self.log)
        data_catalog.save('base_directory', self.base_directory)

        load_data_node = mls.workflows.tasks.LoadDataTask.get_node()
        prepare_data_node = mls.sl.workflows.tasks.PrepareDataTask.get_node()
        split_data_node = mls.sl.workflows.tasks.SplitDataTask.get_node()
        learn_node = mls.sl.workflows.tasks.LearnTask.get_node()
        evaluate_node = mls.sl.workflows.tasks.EvaluateTask.get_node()
        # Assemble nodes into a pipeline
        pipeline = Pipeline([load_data_node, prepare_data_node, split_data_node, learn_node, evaluate_node])
        # Create a runner to run the pipeline
        runner = SequentialRunner()
        # Run the pipeline
        runner.run(pipeline, data_catalog)
        self.terminate()

