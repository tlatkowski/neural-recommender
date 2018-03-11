from utils.data_utils import EXPERIMENTS
from utils.model_config import MODEL

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