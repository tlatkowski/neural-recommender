import tensorflow as tf


class NeuralCollaborativeFiltering:

    def __init__(self, num_users, num_items, user_embedding_size, item_embedding_size, layers_sizes=[200, 100]):

        with tf.Graph().as_default() as graph:

            self.users = tf.placeholder(dtype=tf.float32, shape=[None])
            self.items = tf.placeholder(dtype=tf.float32, shape=[None])

            self.ratings = tf.placeholder(dtype=tf.float32, shape=[None])

            with tf.name_scope('embeddings'):
                users_embeddings = tf.get_variable('users_embeddings', [num_users, user_embedding_size])
                items_embeddings = tf.get_variable('items_embeddings', [num_items, item_embedding_size])

                users_embedded = tf.gather(users_embeddings, self.users)
                items_embedded = tf.gather(items_embeddings, self.items)

            with tf.name_scope('factorization'):
                users_items = tf.concat([users_embedded, items_embedded], axis=0)

                output = users_items
                for i, layer_size in enumerate(layers_sizes):
                    output = tf.layers.dense(output, units=layers_sizes[i], activation=tf.nn.relu)

                prediction = tf.layers.dense(output, units=1, activation=tf.nn.sigmoid)