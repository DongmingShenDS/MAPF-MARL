# DS DONE 1

# DS: controller REGISTRY, assign different classes based on what is passed in
# DS: 1 controller classes: BasicMAC (multi-agent controller)
REGISTRY = {}

from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC
