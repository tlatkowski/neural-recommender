import logging
from enum import Enum

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class Experiments(Enum):
    MovieLens = 0,
    Pinterest = 1


class RecommenderExperiment:
    column_names = ['Users', 'Items', 'Ratings', 'Ids']

    def __init__(self):
        self.train_data = pd.read_csv(self.train_fn(), sep='\t', names=self.column_names)
        self.test_data = pd.read_csv(self.test_fn(), sep='\t', names=self.column_names)
        self.negative_data = pd.read_csv(self.negative_fn(), sep='\t')

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

    def train_fn(self):
        return 'datasets/movie_lens/train.rating'

    def test_fn(self):
        return 'datasets/movie_lens/test.rating'

    def negative_fn(self):
        return 'datasets/movie_lens/test.negative'


class Pinterest(RecommenderExperiment):

    def train_fn(self):
        return 'datasets/pinterest/train.rating'

    def test_fn(self):
        return 'datasets/pinterest/test.rating'

    def negative_fn(self):
        return 'datasets/pinterest/test.negative'


EXPERIMENTS = {
    Experiments.MovieLens.name: MovieLens(),
    Experiments.Pinterest.name: Pinterest()
}


def prepare_experiment(experiment: str, num_negatives=2):
    dataset = EXPERIMENTS[experiment]
    train_with_negatives = get_train_instances(dataset.train(), dataset.num_items(), num_negatives)
    return train_with_negatives


def get_train_instances(train, num_items, num_negatives):
    user_input, item_input, labels = [], [], []
    users_items = train[['Users', 'Items']].as_matrix()
    log.info('Generating training instances...')
    for user, item in users_items:
        # positive instance
        user_input.append(user)
        item_input.append(item)
        labels.append(1)
        # negative instances
        user_positive_items = users_items[users_items[:, 0] == user, 1]
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while user_positive_items.__contains__(j):
                j = np.random.randint(num_items)
            user_input.append(user)
            item_input.append(j)
            labels.append(0)

    log.info('Finished generating training instances.')
    return user_input, item_input, labels
