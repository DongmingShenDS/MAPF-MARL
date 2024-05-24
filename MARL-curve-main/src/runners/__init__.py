# DS DONE 1

# DS: runners REGISTRY, assign different classes based on what is passed in
# DS: 3 runner classes: EpisodeRunner, ParallelRunner
REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner  # DS: episode by default => default.yaml

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
