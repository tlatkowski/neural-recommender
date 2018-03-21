import tensorflow as tf


class BaseRecommenderModel:

    def __init__(self, num_users, num_items, user_embedding_size=50, item_embedding_size=50, learning_rate=0.001):
        self.users = tf.placeholder(dtype=tf.int32, shape=[None])
        self.items = tf.placeholder(dtype=tf.int32, shape=[None])

        self.ratings = tf.placeholder(dtype=tf.float32, shape=[None])

        with tf.name_scope('embeddings'):
            users_embeddings = tf.get_variable('users_embeddings', [num_users, user_embedding_size])
            items_embeddings = tf.get_variable('items_embeddings', [num_items, item_embedding_size])

            users_embedded = tf.gather(users_embeddings, self.users)
            items_embedded = tf.gather(items_embeddings, self.items)

        with tf.name_scope('model'):
            self.prediction = self.model_implementation(users_embedded, items_embedded)

        with tf.name_scope('loss'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.ratings, self.prediction)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope('metrics'):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def model_implementation(self, users_embedded, items_embedded):
        raise NotImplementedError
