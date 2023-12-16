# DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation



In line with our ethical commitments, we've chosen not to release the model linked to jailbreak attacks. 

However, we will detail the specific steps involved in the training process. In knowledge distill model, we use the [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts) as input. For GPT-3.5, GPT4.0. the query script are shown in *query_gpt.py*. For LLama and Vicuna, you should follow the instruction in document([LLama](https://ai.meta.com/llama/), [Vicuna](https://github.com/lm-sys/FastChat)) to deploy the model on your machine.

Once we receive the response from the LLM, we employ the Perspective API to assess its toxicity score. The corresponding query script can be found in *query_perspective.py*.

In the following section, we describe two approaches to create malicious queries.

## Syntax tree based-method

Prerequisites: 

Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```

- Run the code in *gen_malicious_queries.py* to generate the malicious queries

## LLM-based method

We provide the fine-tuning dataset template in the following format:

```
{
    "type": "text2text",
        "instances": [
            {
                "input": "{Malicious Query}",
                "output": "{LLM's response}",
            },
            ...
    	]
}
```

We use [LMflow](https://optimalscale.github.io/LMFlow/index.html) to fine-turing the Vicuna-13B model  the following configuration:

```
./scripts/run_finetune.sh \

 --model_name_or_path lmsys/vicuna-13b-v1.5 \

 --dataset_path data/DistillSeq/train.json \

 --output_model_path output_models/jailbreak_vicuna
```

