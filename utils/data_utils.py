from enum import Enum
import pandas as pd


class Experiments(Enum):
    MovieLens = 0,
    Pinterest = 1


class RecommenderExperiment:

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def negative(self):
        raise NotImplementedError


class MovieLens(RecommenderExperiment):

    train_fn = 'datasets/ml-1m.train.rating'
    test_fn = 'datasets/ml-1m.test.rating'
    negative_fn = 'datasets/ml-1m.test.negative'

    def __init__(self):
        self.train_data = pd.read_csv(self.train_fn, sep='\t')
        self.test_data = pd.read_csv(self.test_fn, sep='\t')
        self.negative_data = pd.read_csv(self.negative_fn, sep='\t')

    def train(self):
        return self.train_data

    def test(self):
        return self.test_data

    def negative(self):
        return self.negative_data


class Pinterest(RecommenderExperiment):

    train_fn = 'datasets/pinterest-20.train.rating'
    test_fn = 'datasets/pinterest-20.test.rating'
    negative_fn = 'datasets/pinterest-20.train.negative'

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

