# DS DONE 1

# DS: modules/agents REGISTRY, assign different classes based on what is passed in
# DS: 1 modules/agents classes: RNNAgent (default as in default.yaml)
REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
