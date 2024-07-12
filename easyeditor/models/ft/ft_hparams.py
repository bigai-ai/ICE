from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str
    objective_optimization: str

    # Defaults
    batch_size: int = 64
    max_length: int = 40
    model_parallel: bool = False

    # ICE
    num_return_sequences: int = 10
    max_new_tokens: int = 5
    static_target: bool = False
    sample_with_context: bool = True
    target_update_interval: int = 1
    temperature: float = 1.0
    print_kl: bool = False
    grad_norm_constraint: float = 1.0

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str, unknown_args): #

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
            for i in config:
                if i in unknown_args:
                    config[i] = unknown_args[i]
        assert (config and config['alg_name'] == 'FT') or print(f'FTHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
