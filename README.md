<p align="center">
    <img src="assets/logo.png" width="500" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://arxiv.org/abs/2406.11194">In-Context Editing: Learning Knowledge from Self-Induced Distributions</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub.  </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2406.11194-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.11194) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ICE-blue)](https://huggingface.co/datasets/Yofuria/ICE) [![paperwithcode](https://img.shields.io/badge/PWC-ICE-blue?logo=paperswithcode)](https://paperswithcode.com/paper/in-context-editing-learning-knowledge-from) [![punkt](https://img.shields.io/badge/%F0%9F%A4%97%20punkt-ICE-blue)](https://huggingface.co/datasets/kailinjiang/punkt) [![KnowledgeEditingPapers](https://img.shields.io/badge/KnowledgeEditingPapers-ICE-blue?logo=github)](https://github.com/zjunlp/KnowledgeEditingPapers) [![AIModels.fyi](https://img.shields.io/badge/AIModels.fyi-ICE-blue?logo=anthropic)](https://www.aimodels.fyi/papers/arxiv/context-editing-learning-knowledge-from-self-induced)



</h5>

# About

This project is developed based on [EasyEdit](https://github.com/zjunlp/EasyEdit). Please refer to the original repository for more details of other methods and an overview of knowledge editing. The following is a list of related repositories:

- [EasyEdit](https://github.com/zjunlp/EasyEdit)  An open source knowledge edit framework.
- [ROME](https://github.com/kmeng01/rome)  A related method of Locating and Editing.
- [MEMIT](https://github.com/kmeng01/memit)  A related method of Locating and Editing.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [🔔News](#news)
- [🌟Overview](#overview)
- [🤗Dataset](#dataset)
- [😮Highlights](#highlights)
  - [🔥 Target learning towards a distribution rather than a one-hot target](#-target-learning-towards-a-distribution-rather-than-a-one-hot-target)
  - [💡 High continual editing performance](#-high-continual-editing-performance)
- [🛠️Requirements and Installation](#️requirements-and-installation)
- [🤖Evaluation](#evaluation)
- [💥Training](#training)
- [🚀Main Results](#main-results)
- [⚡️More qualitative results](#️more-qualitative-results)
- [👍Acknowledgement](#acknowledgement)
- [✏️Citation](#️citation)
- [✨Star History](#star-history)
- [🎉Contributors](#contributors)

## 🔔News

- **[2024.07.03]**  **Code** is available now!

- **[2024.07.03]**  We release the **ICE dataset** at 🤗 [Huggingface Dataset](https://huggingface.co/datasets/Yofuria/ICE).

- **[2023.06.18]**  We hung the **paper** on 🤗 [Huggingface Papers](https://huggingface.co/papers/2406.11194).

## 🌟Overview

<img src="assets\overview.png" width="900px">

**(a)** **In-Context Learning:** Utilizes context prompts without modifying the model's parameters.

**(b) Traditional Fine-Tuning:** Relies on minimizing a distance metric between the model's predictions and one-hot target distributions, often leading to overfitting and unnatural language generation.

**(c) Consistent In-Context Editing (ICE):** Leverages context prompts related to the target knowledge to guide the model towards a new distribution that aligns with the desired knowledge, while maintaining similarity to its original distribution.

<div align="center">
  <img src="assets/algorithm.png" width="350px">
</div>

## 🤗Dataset

We evaluate our method using four datasets, **WikiData<sub>recent</sub>**, **ZsRE**, **WikiBio**, **WikiData<sub>counterfact</sub>**. The four datasets share two tasks of knowledge editing to test the generalization of our method.

<table class="tg" align="center" style="border-collapse: collapse; width: 100%;">
<thead>
  <tr>
    <th class="tg-7btt" style="text-align: center;">Task</th>
    <th class="tg-7btt" style="text-align: center;">Knowledge Insertion</th>
    <th class="tg-7btt" colspan="4" style="text-align: center;">Knowledge Modification</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" style="text-align: center;">Datasets</td>
    <td class="tg-c3ow" style="text-align: center;">WikiData<sub>recent</sub></td>
    <td class="tg-c3ow" style="text-align: center;">ZsRE</td>
    <td class="tg-c3ow" style="text-align: center;">WikiBio</td>
    <td class="tg-c3ow" style="text-align: center;">WikiData<sub>counterfact</sub></td>
  </tr>
  <tr>
    <td class="tg-c3ow" style="text-align: center;">Type</td>
    <td class="tg-c3ow" style="text-align: center;">Fact</td>
    <td class="tg-c3ow" style="text-align: center;">Question Answering</td>
    <td class="tg-c3ow" style="text-align: center;">Hallucination</td>
    <td class="tg-c3ow" style="text-align: center;">Counterfact</td>
  </tr>
  <tr>
    <td class="tg-c3ow" style="text-align: center;"># Train</td>
    <td class="tg-c3ow" style="text-align: center;">570</td>
    <td class="tg-c3ow" style="text-align: center;">10,000</td>
    <td class="tg-c3ow" style="text-align: center;">592</td>
    <td class="tg-c3ow" style="text-align: center;">1,455</td>
  </tr>
  <tr>
    <td class="tg-c3ow" style="text-align: center;"># Test</td>
    <td class="tg-c3ow" style="text-align: center;">1,266</td>
    <td class="tg-c3ow" style="text-align: center;">1,230</td>
    <td class="tg-c3ow" style="text-align: center;">1,392</td>
    <td class="tg-c3ow" style="text-align: center;">885</td>
  </tr>
</tbody>
</table>

You can download data 🤗 [Huggingface Dataset](https://huggingface.co/datasets/Yofuria/ICE). And the expected structure of files is:

```text
ICE
|-- data
|   |-- wikibio.json
|   |-- wikidata_counterfact.json
|   |-- wikidata_recent.json
|   |-- zsre.json
```

## 😮Highlights

### 🔥 Target learning towards a distribution rather than a one-hot target

**In-Context Editing (ICE)** is a novel approach to overcome the brittleness of traditional fine-tuning in knowledge editing scenarios that learns towards a one-hot target.

<div align="center">   <img src="assets/convergence.jpg" width="800px"> </div>

Comparison of ICE with static and dynamic targets on an example, where the query is `"The name of the country which Academy Award for Best Picture is associated with is?"`, and target is `"Wassoulou Empire"`.

**The line** plots on the left illustrate the loss trajectories over optimization steps for static (top) and dynamic (bottom) targets under temperature from 0.1 to 100. **The figures** on the right show how the probabilities of the top-6 predicted tokens for **$x_2$**, the second token following the target, change with iteration steps.

The tokens are arranged from left to right in descending order of probability without context. At early steps, the token `"Wass"` appears due to its presence as the initial token in the target **$x^*$**. At later steps, the probability of `"Wass"` in dynamic targets (top) significantly declines, indicating successful adaptation and suppression of repetitive token predictions. In contrast, for static targets (bottom), the probability of `"Wass"` remains relatively high throughout the optimization steps.

### 💡 High continual editing performance

**Our results confirm the effectiveness of ICE and demonstrate its potential for continual editing**, ensuring that updated information is seamlessly incorporated while preserving the integrity of existing knowledge.

<div align="center">   <img src="assets/continual edit.jpg" width="650px"> </div>

**Continual editing** with Llama2-7b-chat on **Wiki<sub>recent</sub>**. Each edit builds on the previous model, risking deterioration over time. The model is assessed immediately after each edit without re-evaluating previous edits, testing its ability to update continuously. While most methods deteriorate, sometimes performing worse than the unedited version, our method, ICE, maintains integrity and achieves promising performance.

## 🛠️Requirements and Installation

```text
# clone ICE
git clone https://github.com/Yofuria/ICE.git
cd ICE

# create conda env
conda create -n ICE python=3.10
conda activate ICE

# install package
pip install -r requirements.txt
```

In **lines 32 and 33** of **`examples/run_knowedit_llama2.py`**, you need to download the **`punkt`** package.

- If your Internet **speed is fast** enough, you can **run the code directly** from the command line.

```text
if __name__ == "__main__":
    # If you have a slow Internet connection and can't download nltk quickly, comment these two lines and use the second method of Requirements and Installation in README.md
    import nltk
    nltk.download('punkt')
```

- If your Internet **speed is slow**, **comment lines 32 and 33** and **download punkt manually**🤗 [punkt](https://huggingface.co/datasets/kailinjiang/punkt). And place it in the ICE environment directory you created, create a **nltk_data/tokenizers** folder, and **unpack punkt** into this directory.

<div align="center">   <img src="assets/punkt.png" width="650px"> </div>

## 🤖Evaluation

You can get the evaluation results using `eval.py`. Evaluation indicators are as follows：

- `rewrite_acc` $\rightarrow$ **Edit Success**[measures the ability of the model to produce the edited response $x^*$ for a query **$q$**]

  <div align="center">
    <img src="assets/edit succ.png" width="450px">
  </div>

- `locality` $\rightarrow$ **Locality** [evaluates if the model maintains original predictions for queries outside the edit scope]

<div align="center">
  <img src="assets/loc.png" width="450px">
</div>

- `portablility` $\rightarrow$ **Portablility** [assesses how well the model generalizes the knowledge for rephrased or logically related queries within the edit scope **$D_q$**]

<div align="center">
  <img src="assets/port.png" width="450px">
</div>

- `ngram_entropy` $\rightarrow$ **Fluency** [estimates the linguistic quality of the postedit model's output, given by a weighted sum of bi- and tri-gram entropies]

<div align="center">
  <img src="assets/flu.png" width="400px">
</div>

- `PPL_r` $\rightarrow$ **PPL<sub>r</sub>** [we introduce a normalized perplexity ratio, comparing the perplexity of the generated sentence beyond the target token to that of the prompt and target token combined]

<div align="center">
  <img src="assets/PPL.png" width="350px">
</div>

<div align="center">
  <img src="assets/PPL_r.png" width="200px">
</div>

After the editing operation, you get a json file with error result data.

```text
{
    "pre": {
        "rewrite_acc": [],
        "portability": {
            "Subject_Aliasing_acc": [],
            "reasoning_acc": []
        },
        "fluency": {
            "ngram_entropy": 
        }
    },
    
    "case_id": 0,
    "requested_rewrite": {
        //...
    },
    
    "time": ,
    "post": {
        "rewrite_acc": [],
        "locality": {
            "Relation_Specificity_acc": [],
            "Forgetfulness_acc": []
        },
        "portability": {
            "Subject_Aliasing_acc": [],
            "reasoning_acc": []
        },
        "fluency": {
            "ngram_entropy": 
        }
    }
}
```

The data used by `PPL_r`is the edit operation that saves the sentences generated by the model.

Such as：`ICE_zsre_Llama-2-7b-chat-hf_gen_sentence.json`

```shell
python eval.py 
    --model_name_or_path=''  # Path to pre-trained model
    --output_file='./FT-M_counterfact_gpt2-xl_gen_sentence.json'  # Generated sentences file (xxx.json)
    --result_file='./FT-M_counterfact_gpt2-xl_results.json'  # Result file (xxx.json)
```

You will get the **following metrics**

```text
Edit_Succ: 30.262626262626263
Portability: 7.3802393354053
Portability (Subject_Aliasing_acc): 6.939620928384972
Portability (reasoning_acc): 3.511697773992855
Portability (Logical_Generalization_acc): 9.11111111111111
Locality: 33.95236461069794
Fluency: 557.8193009507412
ppl_r:  tensor(9.9633, device='cuda:0')
```

## 💥Training

We provide the training hyperparameters for five methods in `./hparams`.

For ICE, we update **GPT2-xl** using **layers 13 to 17** and **Llama2-7b-chat** using **layers 4 to 8**.

Both FT-L and FT-M use the same hparams located in `./hparams/FT`.

For FT-L, replace `objective_optimization` with `prompt_last`, and for FT-M, replace it with `target_new`. For details on other methods, please refer to [EasyEdit](https://github.com/zjunlp/EasyEdit). You can execute the following commands to obtain results:

**For ICE:**

```shell
python examples/run_knowedit_llama2.py \
    --editing_method=ICE \
    --hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For FT-L:**

```shell
python examples/run_knowedit_llama2.py \
    --editing_method=FT-L \
    --hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For FT-M:**

```shell
python examples/run_knowedit_llama2.py \
    --editing_method=FT-M \
    --hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For MEMIT:**

```shell
python examples/run_knowedit_llama2.py \
    --editing_method=MEMIT \
    --hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

**For ROME:**

```shell
python examples/run_knowedit_llama2.py \
    --editing_method=ROME \
    --hparams_dir=./hparams/ICE/gpt2-xl.yaml \
    --data_dir=./data/zsre.json \  
    --datatype='zsre' \  
    --metrics_save_dir=./results/gpt2-xl/ICE
```

The optional range of `datatype` is `['zsre','recent','counterfact','wikibio']`

**ICE/gpt2-xl.yaml**

```
alg_name: "FT"
model_name: openai-community/gpt2-xl   # or local checkpoint path
device: 0
layers: [13, 14, 15, 16, 17]
num_steps: 25
batch_size: 3
max_length: 40
lr: 7e-4
weight_decay: 0
kl_factor: 0
norm_constraint: 5e-4
grad_norm_constraint: 5e-4
num_return_sequences: 1
max_new_tokens: 3
static_target: False
sample_with_context: True
target_update_interval: 1
temperature: 100.0
print_kl: True

objective_optimization: "target_and_completion_with_context"
rewrite_module_tmp: "transformer.h.{}.mlp.c_proj"
layer_module_tmp: "transformer.h.{}"
mlp_module_tmp: "transformer.h.{}.mlp"
attn_module_tmp: "transformer.h.{}.attn"
ln_f_module: "transformer.ln_f"
lm_head_module: "transformer.wte"
model_parallel: False
```

## 🚀Main Results

- Main results on **knowledge insertion and question-answering datasets** of Llama2-7b-chat

<img src="assets\llama2_main-results1.png" width="1000px">

- Main results on **knowledge modification datasets** of Llama2-7b-chat

<img src="assets\llama2_main-results2.png" width="1000px">

- **Continual editing results** of Llama2-7b-chat

<div align="center">
  <img src="assets/continual edit.png" width="500px">
</div>

## ⚡️More qualitative results

<div align="center">
  <img src="assets/qualitative1.png" width="650px">
</div>

<div align="center">
  <img src="assets/qualitative2.png" width="650px">
</div>

<div align="center">
  <img src="assets/qualitative3.png" width="650px">
</div>

## ✏️Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```text
@article{qi2024ice,
      title={In-Context Editing: Learning Knowledge from Self-Induced Distributions}, 
      author={Siyuan Qi and Bangcheng Yang and Kailin Jiang and Xiaobo Wang and Jiaqi Li and Yifan Zhong and Yaodong Yang and Zilong Zheng},
      year={2024},
      eprint={2406.11194},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.11194}, 
}
```

## ✨Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bigai-ai/ICE&type=Date)](https://star-history.com/#bigai-ai/ICE&Date)

## 🎉Contributors
<img src="assets\2contributors.png" width="700px">

[![Contributors](https://img.shields.io/badge/Contributor-SiyuanQi-blue?logo=github)](https://github.com/SiyuanQi) 
[![Contributors](https://img.shields.io/badge/Contributor-BangchengYang-blue?logo=github)](https://github.com/DumbMice) 
[![Contributors](https://img.shields.io/badge/Contributor-KailinJiang-blue?logo=github)](https://github.com/kailinjiang) 
[![Contributors](https://img.shields.io/badge/Contributor-XiaoboWang-blue?logo=github)](https://github.com/Yofuria)  <br>
[![Contributors](https://img.shields.io/badge/Contributor-JiaqiLi-blue?logo=github)](https://github.com/lijiaqijane) 
[![Contributors](https://img.shields.io/badge/Contributor-YifanZhong-blue?logo=github)](https://github.com/Ivan-Zhong) 
[![Contributors](https://img.shields.io/badge/Contributor-YaodongYang-blue?logo=github)](https://github.com/PKU-YYang) 
[![Contributors](https://img.shields.io/badge/Contributor-ZilongZheng-blue?logo=github)](https://github.com/zilongzheng)  <br>

