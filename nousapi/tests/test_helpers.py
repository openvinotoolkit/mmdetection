#  COSMONiO © All rights reserved.
#  This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#  which is part of this source code package.

import time
import unittest
from typing import Optional
from unittest.case import TestCase

from noussdk.entities.datasets import Subset
from noussdk.entities.metrics import Performance
from noussdk.entities.model import NullModel
from noussdk.entities.optimized_model import OptimizedModel
from noussdk.entities.project import Project
from noussdk.entities.resultset import ResultSet
from noussdk.entities.task_environment import TaskEnvironment
from noussdk.tests.test_helpers import generate_and_save_random_annotated_video, \
    generate_training_dataset_of_all_annotated_media_in_project
from noussdk.usecases.repos import ModelRepo, DatasetRepo
from noussdk.usecases.tasks.image_deep_learning_task import ImageDeepLearningTask
from noussdk.usecases.tasks.interfaces.model_optimizer import IModelOptimizer
from noussdk.utils.gpu_utils import GPUMonitor


def train_task(
        test_case: unittest.TestCase,
        task: ImageDeepLearningTask,
        project: Project,
        random_seed: Optional[int] = None,
        f1_threshold: Optional[float] = 0.5,
        add_video_to_project: bool = True,
) -> Performance:
    """
    Test a detection task object, with a given project.
    Trains model, performs inference and computes performance
    :param task: Task implementation to test
    :param project: Project to test
    :param random_seed: Seed to initialize the random number generator. If None is given, current system time is used
    :param f1_threshold: F-measure threshold for the test case.
    :param add_video_to_project: True to add an additional annotated video to the project
    :return: None
    """
    if add_video_to_project:
        # Add annotated video with annotations to projects
        generate_and_save_random_annotated_video(project=project,
                                                 video_name="Video for Detection",
                                                 width=480, height=360, random_seed=random_seed)

    # Generate training environment
    training_dataset = generate_training_dataset_of_all_annotated_media_in_project(project, seed=random_seed)
    DatasetRepo(project).save(training_dataset)
    model = task.train(dataset=training_dataset)
    test_case.assertFalse(isinstance(model, NullModel))
    ModelRepo(project).save(model)
    if isinstance(task, IModelOptimizer):
        optimized_models = task.optimize_loaded_model()
        test_case.assertGreater(len(optimized_models), 0, "Task must return an Optimised model.")
        for m in optimized_models:
            test_case.assertIsInstance(m, OptimizedModel,
                                       "Optimised model must be an Openvino or DeployableTensorRT model.")

    # Run inference
    inference_dataset = training_dataset.with_empty_annotations()
    start_time = time.time()
    result_dataset = task.analyse(dataset=inference_dataset)
    end_time = time.time()
    print(f"{len(inference_dataset)} analysed in {end_time - start_time} seconds")

    performance = compute_validation_performance(task, task.task_environment)

    print(f"Evaluated model to have a performance of {performance}")

    test_case.assertGreater(performance.score.value, f1_threshold, "Expected F-measure to be higher than 0.5 [flaky]")
    return performance


def create_and_start_gpu_monitor(test_case: TestCase, show_in_console: bool = True,
                                 interval: float = 5.0) -> GPUMonitor:
    """
    creates and starts gpu monitor to monitor test

    :param test_case:
    :param show_in_console:
    :param interval:
    :return:
    """
    gpu_monitor = GPUMonitor()
    gpu_monitor.interval = interval
    gpu_monitor.show_in_console = show_in_console
    gpu_monitor.start()
    test_case.addCleanup(function=gpu_monitor.stop)
    return gpu_monitor


def compute_validation_performance(task: ImageDeepLearningTask, task_environment: TaskEnvironment) -> Performance:
    """
    Computes a task's validation performance using the current model and training dataset.
    """
    dataset = task_environment.model.train_dataset
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    prediction_dataset = validation_dataset.with_empty_annotations()
    prediction_dataset = task.analyse(dataset=prediction_dataset)

    result_set = ResultSet(
        model=task_environment.model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=prediction_dataset
    )
    performance = task.compute_performance(result_set)
    return performance
