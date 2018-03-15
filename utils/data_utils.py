from enum import Enum

import numpy as np
import pandas as pd


class Experiments(Enum):
    MovieLens = 0,
    Pinterest = 1


class RecommenderExperiment:
    column_names = ['Users', 'Items', 'Ratings', 'Ids']

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def negative(self):
        raise NotImplementedError

    def num_users(self):
        return max(self.train()['Users'])

    def num_items(self):
        return max(self.train()['Items'])


class MovieLens(RecommenderExperiment):
    train_fn = 'datasets/movie_lens/train.rating'
    test_fn = 'datasets/movie_lens/test.rating'
    negative_fn = 'datasets/movie_lens/test.negative'

    def __init__(self):
        self.train_data = pd.read_csv(self.train_fn, sep='\t', names=self.column_names)
        self.test_data = pd.read_csv(self.test_fn, sep='\t', names=self.column_names)
        self.negative_data = pd.read_csv(self.negative_fn, sep='\t')

    def train(self):
        return self.train_data

    def test(self):
        return self.test_data

    def negative(self):
        return self.negative_data


class Pinterest(RecommenderExperiment):
    train_fn = 'datasets/pinterest/train.rating'
    test_fn = 'datasets/pinterest/test.rating'
    negative_fn = 'datasets/pinterest/pinterest-20.train.negative'

    def train(self):
        return self.train_data

    def test(self):
        return self.test_data

    def negative(self):
        return self.negative_data


EXPERIMENTS = {
    Experiments.MovieLens.name: MovieLens(),
    Experiments.Pinterest.name: Pinterest()
}


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels
