import logging
import os
from enum import Enum

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Experiments(Enum):
    MovieLens = 0,
    Pinterest = 1


DATASETS_PATS = {
    Experiments.MovieLens.name: 'datasets/movie_lens',
    Experiments.Pinterest.name: 'datasets/pinterest'
}


class RecommenderExperiment:
    column_names = ['Users', 'Items', 'Ratings', 'Ids']

    def __init__(self):
        self.train_data = pd.read_csv('{}/{}'.format(self.dataset_path(), self.train_fn()), sep='\t',
                                      names=self.column_names)
        self.test_data = pd.read_csv('{}/{}'.format(self.dataset_path(), self.test_fn()), sep='\t',
                                     names=self.column_names)
        self.negative_data = pd.read_csv('{}/{}'.format(self.dataset_path(), self.negative_fn()), sep='\t')

    def dataset_path(self):
        raise NotImplementedError

    def train_fn(self):
        raise NotImplementedError

    def test_fn(self):
        raise NotImplementedError

    def negative_fn(self):
        raise NotImplementedError

    def train(self):
        return self.train_data

    def test(self):
        return self.test_data

    def negative(self):
        return self.negative_data

    def num_users(self):
        return max(self.train_data['Users']) + 1

    def num_items(self):
        return max(self.train_data['Items']) + 1


class MovieLens(RecommenderExperiment):

    def dataset_path(self):
        return DATASETS_PATS[Experiments.MovieLens.name]

    def train_fn(self):
        return 'train.rating'

    def test_fn(self):
        return 'test.rating'

    def negative_fn(self):
        return 'test.negative'


class Pinterest(RecommenderExperiment):

    def dataset_path(self):
        return DATASETS_PATS[Experiments.Pinterest.name]

    def train_fn(self):
        return 'train.rating'

    def test_fn(self):
        return 'test.rating'

    def negative_fn(self):
        return 'test.negative'


EXPERIMENTS = {
    Experiments.MovieLens.name: MovieLens(),
    Experiments.Pinterest.name: Pinterest()
}


def prepare_experiment(model_dir, experiment: str, num_negatives=2):
    dataset = EXPERIMENTS[experiment]
    train_with_negatives = get_train_instances(dataset, num_negatives)
    return train_with_negatives


def get_train_instances(dataset, num_negatives, force=False):
    """
    Randomly generates negatives samples for each training example.
    :param dataset:
    :param num_negatives:
    :param force:
    :return:
    """
    dataset_path = dataset.dataset_path()
    train_data = dataset.train()
    num_items = dataset.num_items()
    if force:  # generate data and save it to file
        training_data = _create_training_file(dataset_path, train_data, num_items, num_negatives)
    else:  # read data from file
        file_to_load = '{}/training_data.csv'.format(dataset_path)
        training_data = pd.read_csv(file_to_load)
        logger.info('Loaded training data from: %s', file_to_load)
    return training_data


def _create_training_file(model_dir, train, num_items, num_negatives):
    os.makedirs(model_dir, exist_ok=True)
    user_input, item_input, labels = [], [], []
    users_items = train[['Users', 'Items']].as_matrix()
    logger.info('Corpus contains %d samples, generating negatives will take a while...', len(train))
    logger.info('Generating training instances...')
    for user, item in users_items:
        # positive instance
        user_input.append(user)
        item_input.append(item)
        labels.append(1)
        # negative instances
        user_positive_items = users_items[users_items[:, 0] == user, 1]
        for t in range(num_negatives):
            sample_item = np.random.randint(num_items)
            while sample_item in user_positive_items:
                sample_item = np.random.randint(num_items)
            user_input.append(user)
            item_input.append(sample_item)
            labels.append(0)
    training_data = pd.DataFrame({'users': user_input, 'items': item_input, 'labels': labels})
    file_to_save = '{}/training_data.csv'.format(model_dir)
    training_data.to_csv(file_to_save, index=False)
    logger.info('Saved training data to: %s', file_to_save)
    logger.info('Finished generating training instances.')
    return training_data