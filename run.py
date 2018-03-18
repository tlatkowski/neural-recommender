import configparser
import logging
from argparse import ArgumentParser

import tensorflow as tf

from utils.data_utils import prepare_experiment, EXPERIMENTS
from utils.model_config import MODEL

log = logging.getLogger(__name__)


def train(model, experiment_name, main_config):
    # train = experiment.train()
    # neg = experiment.negative()
    experiment = EXPERIMENTS[experiment_name]

    num_items = experiment.num_items()
    num_users = experiment.num_users()

    model = model(num_users,
                  num_items,
                  user_embedding_size=50,
                  item_embedding_size=50)

    log.info('Loaded {} model'.format(model))
    user_input, item_input, labels = prepare_experiment(experiment_name)
    log.info('Loaded {} experiment'.format(experiment))

    num_epochs = main_config['TRAINING']['num_epochs']
    batch_size = main_config['TRAINING']['batch_size']

    num_batches = len(labels) // batch_size

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            # shuffle data

            for batch in range(num_batches):
                user_batch = user_input[batch * batch_size: (batch + 1) * batch_size]
                item_batch = item_input[batch * batch_size: (batch + 1) * batch_size]
                labels_batch = labels[batch * batch_size: (batch + 1) * batch_size]
                feed_dict = {model.users: user_batch, model.items: item_batch, model.ratings: labels_batch}
                loss, opt = session.run([model.loss, model.optimizer], feed_dict=feed_dict)
                print(loss)


def main():
    parser = ArgumentParser()
    parser.add_argument('model',
                        choices=['mf', 'ncf', 'acf'],
                        help='model to be used')
    parser.add_argument('experiment',
                        choices=['MovieLens', 'Pinterest'],
                        help='experiment to be used')
    args = parser.parse_args()

    main_config = configparser.ConfigParser()
    main_config.read('config/main.ini')

    model = MODEL[args.model]

    train(model, args.experiment, main_config)


if __name__ == '__main__':
    main()
