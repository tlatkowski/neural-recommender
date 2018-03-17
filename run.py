from utils.data_utils import prepare_experiment, Experiments, EXPERIMENTS
from utils.model_config import MODEL
import tensorflow as tf

experiment = EXPERIMENTS['MovieLens']
train = experiment.train()
neg = experiment.negative()

num_items = experiment.num_items()
num_users = experiment.num_users()

model = MODEL['ncf'](num_users,
                     num_items,
                     user_embedding_size=50,
                     item_embedding_size=50,
                     layers_sizes=[200, 100])

user_input, item_input, labels = prepare_experiment(Experiments.MovieLens)

num_epochs = 10
batch_size = 128

num_batches = len(labels) // batch_size

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        #shuffle data

        for batch in range(num_batches):
            user_batch = user_input[batch * batch_size: (batch + 1) * batch_size]
            item_batch = item_input[batch * batch_size: (batch + 1) * batch_size]
            labels_batch = labels[batch * batch_size: (batch + 1) * batch_size]
            feed_dict = {model.users: user_batch, model.items: item_batch, model.ratings: labels_batch}
            loss, opt = session.run([model.loss, model.optimizer], feed_dict=feed_dict)
            print(loss)