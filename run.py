import configparser
import logging
from argparse import ArgumentParser
from tqdm import tqdm

import tensorflow as tf

from utils.data_utils import prepare_experiment, EXPERIMENTS
from utils.model_config import MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, experiment_name, main_config):
    model_dir = 'model'

    experiment = EXPERIMENTS[experiment_name]
    num_items = experiment.num_items()
    num_users = experiment.num_users()

    model = model(num_users,
                  num_items,
                  user_embedding_size=50,
                  item_embedding_size=50)

    logger.info('Loaded {} model'.format(model))
    training_df = prepare_experiment(model_dir, experiment_name)
    user_input, item_input, labels = training_df['users'].as_matrix(), training_df['items'].as_matrix(), \
                                     training_df['labels'].as_matrix()

    logger.info('Loaded {} experiment'.format(experiment))

    num_epochs = int(main_config['TRAINING']['num_epochs'])
    batch_size = int(main_config['TRAINING']['batch_size'])

    num_batches = len(labels) // batch_size
    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter('{}/{}/test/'.format(model_dir, 'recommender'), graph=session.graph)
        session.run(tf.global_variables_initializer())

        global_step = 0
        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            # shuffle data

            tqdm_iter = tqdm(range(num_batches), total=num_batches, desc="Batches", leave=False)
            for batch in range(num_batches):
                global_step += 1
                user_batch = user_input[batch * batch_size: (batch + 1) * batch_size]
                item_batch = item_input[batch * batch_size: (batch + 1) * batch_size]
                labels_batch = labels[batch * batch_size: (batch + 1) * batch_size]
                feed_dict = {model.users: user_batch, model.items: item_batch, model.ratings: labels_batch}
                loss, opt, summary_op = session.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_op, global_step)
                if batch % 10 == 0:
                    pass  # make eval

                tqdm_iter.set_postfix(
                    loss='{:.2f}'.format(float(loss)),
                    barch='{}|{}'.format(batch, num_batches),
                    epoch=epoch)


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
