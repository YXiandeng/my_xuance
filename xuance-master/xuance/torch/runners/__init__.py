from .runner_drl import Runner_DRL
from .runner_marl import Runner_MARL
from .runner_pettingzoo import Pettingzoo_Runner
from .runner_magent import MAgent_Runner
from .runner_sc2 import SC2_Runner
from .runner_football import Football_Runner
from .runner_marl_EA import Runner_MARL_EA

REGISTRY = {
    "DL_toolbox": "PyTorch",
    "DRL": Runner_DRL,
    "MARL": Runner_MARL,
    "MARL_EA": Runner_MARL_EA,
    "Pettingzoo_Runner": Pettingzoo_Runner,
    "MAgent_Runner": MAgent_Runner,
    "StarCraft2_Runner": SC2_Runner,
    "Football_Runner": Football_Runner
}
