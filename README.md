# **MathCoder**
This repo is for "[MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning](https://arxiv.org/pdf/2310.03731.pdf)"


<br>
<div align="center">
  <img src="figures/mathcoder.png" width="80%" title="Introduction Figure">
</div>

### Datasets and Models
Our 7B models are available at Huggingface now.

<!-- 🤗 [MathCodeInstruct Dataset](https://huggingface.co/datasets/MathLLM/MathCodeInstruct) -->

|     	| Base Model: Llama-2                                           	| Base Model: Code Llama                                                    	|
|-----	|---------------------------------------------------------------	|---------------------------------------------------------------------------	|
| 7B  	|  [MathCoder-L-7B](https://huggingface.co/MathLLM/MathCoder-L-7B)   	|  [MathCoder-CL-7B](https://huggingface.co/MathLLM/MathCoder-CL-7B)   	|

<br>
<div align="center">
  <img src="figures/result.png" width="100%" title="Result Figure">
</div>


## **Introduction**
The recently released GPT-4 Code Interpreter has demonstrated remarkable proficiency in solving challenging math problems, primarily attributed to its ability to seamlessly reason with natural language, generate code, execute code, and continue reasoning based on the execution output. In this paper, we present a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities.

We propose a method of generating novel and high-quality datasets with math problems and their code-based solutions, referred to as MathCodeInstruct. Each solution interleaves *natural language*, *code*, and *execution results*. 

We also introduce a customized supervised fine-tuning and inference approach. This approach yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems.

Impressively, the MathCoder models achieve state-of-the-art scores among open-source LLMs on the MATH (45.2\%) and GSM8K (83.9\%) datasets, substantially outperforming other open-source alternatives. Notably, the MathCoder model not only surpasses ChatGPT-3.5 and PaLM-2 on GSM8K and MATH but also outperforms GPT-4 on the competition-level MATH dataset. The proposed dataset and models will be released upon acceptance.
<br>
<div align="center">
  <img src="figures/pipeline.png" width="100%" title="Result Figure">
</div>

## **Citation**

Please cite the paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@misc{wang2023mathcoder,
      title={MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning}, 
      author={Ke Wang and Houxing Ren and Aojun Zhou and Zimu Lu and Sichun Luo and Weikang Shi and Renrui Zhang and Linqi Song and Mingjie Zhan and Hongsheng Li},
      year={2023},
      eprint={2310.03731},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

