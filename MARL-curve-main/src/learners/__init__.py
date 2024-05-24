# DS DONE 1
from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner

# DS: learners REGISTRY, assign different classes based on what is passed in
# DS: 3 learner classes: QLearner, COMALearner, QTranLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
