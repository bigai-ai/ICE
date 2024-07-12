# Examples

This README is about reproducing the EMNLP 2023 paper "[Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)".
We host a wide range of examples to elaborate the basic usage of EasyEdit. 

Please discuss in an [issue](https://github.com/zjunlp/EasyEdit/issues) a feature you would  like to implement in an example before submitting a PR; we welcome bug fixes, but since we want to keep the examples as simple as possible it's unlikely that we will merge a pull request adding more functionality at the cost of readability.

---

## Table of Contents

- [Data](#data)
- [Edit llama2 on ZsRE](#edit-llama2-on-zsre)
  - [ROME](#rome)
  - [MEMIT](#memit)
  - [FT-L](#ft)
  - [KN](#kn)
  - [IKE](#ike)
  - [LoRA](#lora)

## Data

The datasets used can be downloaded from [here](https://drive.google.com/file/d/1WRo2SqqgNtZF11Vq0sF5nL_-bHi18Wi4/view?usp=sharing). Unzip the file and put it to `./data`, and the final directory structure is as follows:

```text
editing-data
├── counterfact
│   ├── counterfact-edit.json
│   ├── counterfact-train.json
│   └── counterfact-val.json
├── locality
│   ├── Commonsense Task
│   │   ├── piqa_valid-labels.lst
│   │   └── piqa_valid.jsonl
│   ├── Distracting Neighbor
│   │   └── counterfact_distracting_neighbor.json
│   └── Other Attribution
│       └── counterfact_other_attribution.json
├── portability
│   ├── Inverse Relation
│   │   └── zsre_inverse_relation.json
│   ├── One Hop
│   │   ├── counterfact_portability_gpt4.json
│   │   └── zsre_mend_eval_portability_gpt4.json
│   └── Subject Replace
│       ├── counterfact_subject_replace.json
│       └── zsre_subject_replace.json
└── zsre
    ├── zsre_mend_eval.json
    ├── zsre_mend_train_10000.json
    └── zsre_mend_train.json
```

- counterfact: original counterfact dataset using Entity replacement
- zsre: original question answering dataset using question rephrasings
- locality (evaluation for locality, see details in this [paper](https://arxiv.org/abs/2305.13172))
    - Commonsense Task: evaluation for other downstream tasks such as commonsense task
    - Distracting Neighbor: test on distracting neighborhood ([reference: Detecting Edit Failures...](https://arxiv.org/abs/2305.17553))
    - Other Attribution
- portability
    - Inverse Relation: evaluation for one-to-one relationship such as `spouse`
    - One Hop: evaluation for one-hop reasoning
    - Subject Replace: evaluation for synonym replacement


## Edit llama2 on ZsRE

In the paper [EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models](https://arxiv.org/abs/2308.07269), the data set we used is `zsre_mend_eval_portability_gpt4.json`, so you should place it in the `./data` directory.

- `editing_method`: Knowledge Editing Method (e.g., `ROME`, `MEMIT`, `IKE`)
- `hparams_dir`: HyperParams Path.
- `data_dir`: dataset Path.

- Metric results for each editing are stored at `metrics_save_dir`(default: `./YOUR_EDITING_METHOD_results.json`)

> Note that place the model to be edited in the `./hugging_cache` directory.



### ROME
```shell
python run_zsre_llama2.py \
    --editing_method=ROME \
    --hparams_dir=../hparams/ROME/llama-7b \
    --data_dir=./data
```

**params in `hparams_dir`**:

- `mom2_adjustment`: Set it to `false` to skip computing the second-order momentum (default is false, which can speed up the editing process). 
    - If set to `true`, computation regarding Wikipedia's npz is required.


### MEMIT

```shell
python run_zsre_llama2.py \
    --editing_method=MEMIT \
    --hparams_dir=../hparams/MEMIT/llama-7b \
    --data_dir=./data
```

- `MEMIT` cannot bypass the computation of second-order momentum, so it requires the `npz` related to Wikipedia (either computed locally or obtained online).
- Here, we provide the pre-trained weights for layers `[4, 5, 6, 7, 8]` in `llama2`. You can download them [here](https://drive.google.com/drive/folders/1IGt7NNV-OxXqIljjr02_k0dDY50Z5N_E?usp=sharing).
    - Place several `npz` files in the directory **`./data/stats/._hugging_cache_llama-2-7b/wikipedia_stats`**, as shown in the following.
    - ```text
        examples
        ├── data
        │   ├── stats
        │   │   └── ._hugging_cache_llama-2-7b
        │   │       └── wikipedia_stats
        │   │           ├── model.layers.4.mlp.down_proj_float32_mom2_100000.npz
        │   │           ├── model.layers.5.mlp.down_proj_float32_mom2_100000.npz
        │   │           ├── model.layers.6.mlp.down_proj_float32_mom2_100000.npz
        │   │           ├── model.layers.7.mlp.down_proj_float32_mom2_100000.npz
        │   │           └── model.layers.8.mlp.down_proj_float32_mom2_100000.npz
        └── └── zsre_mend_eval_portability_gpt4.json
        ```

### FT

```shell
python run_zsre_llama2.py \
    --editing_method=FT \
    --hparams_dir=../hparams/FT/llama-7b \
    --data_dir=./data
```

### KN

```shell
python run_zsre_llama2.py \
    --editing_method=KN \
    --hparams_dir=../hparams/KN/llama-7b \
    --data_dir=./data
```

### IKE

```shell
python run_zsre_llama2.py \
    --editing_method=IKE \
    --hparams_dir=../hparams/IKE/llama-7b \
    --data_dir=./data
```

### LoRA

```shell
python run_zsre_llama2.py \
    --editing_method=LoRA \
    --hparams_dir=../hparams/LoRA/llama-7b \
    --data_dir=./data
```
# Citation
```bibtex
@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}
```
