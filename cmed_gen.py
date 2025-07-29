#cmd_gen.py
from mmengine.config import read_base

with read_base():
    from .cmed_gen_c13365 import cmed_datasets  # noqa: F401, F403
    #from opencompass.configs.datasets.cmed.cmed_gen_c13365 import cmed_datasets