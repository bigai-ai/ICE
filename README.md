
<h1 align="center"> <a href="https://arxiv.org/abs/2406.11194">In-Context Editing: Learning Knowledge from Self-Induced Distributions</a></h1>
<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2406.11194-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.11194) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ICE-blue)](https://huggingface.co/datasets/Yofuria/ICE) [![paperwithcode](https://img.shields.io/badge/PWC-ICE-blue?logo=paperswithcode)](https://paperswithcode.com/paper/in-context-editing-learning-knowledge-from) [![punkt](https://img.shields.io/badge/%F0%9F%A4%97%20punkt-ICE-blue)](https://huggingface.co/datasets/kailinjiang/punkt) [![KnowledgeEditingPapers](https://img.shields.io/badge/KnowledgeEditingPapers-ICE-blue?logo=github)](https://github.com/zjunlp/KnowledgeEditingPapers) [![AIModels.fyi](https://img.shields.io/badge/AIModels.fyi-ICE-blue?logo=anthropic)](https://www.aimodels.fyi/papers/arxiv/context-editing-learning-knowledge-from-self-induced)



</h5>

# About

This repository is the official implementation of the paper "In-Context Editing: Learning Knowledge from Self-Induced Distributions". The main idea of this paper is to use an in-context distribution to guide the learning process of knowledge editing for language models.

This project is developed based on [EasyEdit](https://github.com/zjunlp/EasyEdit). Please refer to the original repository for more details of other methods and an overview of knowledge editing. The following is a list of related repositories:

- [EasyEdit](https://github.com/zjunlp/EasyEdit)  An open source knowledge edit framework.
- [ROME](https://github.com/kmeng01/rome)  A related method of Locating and Editing.
- [MEMIT](https://github.com/kmeng01/memit)  A related method of Locating and Editing.

## Table of Contents

<!-- - [🔔News](#news)
- [🌟Overview](#overview) -->
- [🤗Dataset](#dataset)
- [🛠️Requirements and Installation](#️requirements-and-installation)
- [🤖Evaluation](#evaluation)
- [💥Training](#training)


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

- If your **internet connection is sufficiently fast**, you can **execute the code directly** from the command line.

```text
if __name__ == "__main__":
    # If you have a slow Internet connection and can't download nltk quickly, comment these two lines and use the second method of Requirements and Installation in README.md
    import nltk
    nltk.download('punkt')
```

- If your **internet speed is slow**, **comment out lines 32 and 33** and **manually download the punkt package** from [punkt](https://huggingface.co/datasets/kailinjiang/punkt). Place it in the ICE environment directory you created, then create a **nltk_data/tokenizers** folder, and **extract the contents of punkt** into this directory.

<div align="center">   <img src="assets/punkt.png" width="650px"> </div>

## 🤖Evaluation

You can get the evaluation results using `eval.py`. 

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

<!-- ## ✨Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bigai-ai/ICE&type=Date)](https://star-history.com/#bigai-ai/ICE&Date) -->

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

