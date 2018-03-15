from enum import Enum

from models import attention_collaborative_filtering, matrix_factorization, neural_collaborative_filtering


class AttentionType(Enum):
    additive = 0,
    multiplicative = 1


class ModelType(Enum):
    matrix_factor = 0,
    ncf = 1,
    acf = 2


MODEL = {
    ModelType.matrix_factor.name: matrix_factorization.MatrixFactorization,
    ModelType.ncf.name: neural_collaborative_filtering.NeuralCollaborativeFiltering,
    ModelType.acf.name: attention_collaborative_filtering.AttentionCollaborativeFiltering
}
