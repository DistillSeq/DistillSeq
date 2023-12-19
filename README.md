# DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation



This is the source code for the paper DistillSeq A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation.

## Knowledge distill

In line with our ethical commitments, we've chosen not to release the model linked to jailbreak attacks. 

However, we will detail the specific steps involved in the knowledge distill process. The knowledge distill model uses the [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts) as input. For GPT-3.5, GPT4.0. the query script is shown in *query_gpt.py*. For LLama and Vicuna, you should follow the instructions in the document([LLama](https://ai.meta.com/llama/), [Vicuna](https://github.com/lm-sys/FastChat)) to deploy the model on your machine.

Once we receive the response from the LLM, we employ the Perspective API to assess its toxicity score. The corresponding query script can be found in *query_perspective.py*.

In the following section, we describe two approaches for creating malicious queries.

## Syntax tree based-method

### Prerequisites: 

Required packages and tool are listed in below.

NLTK:

```
pip install nltk
```

Stanford Parser:

You should visit [this link](https://nlp.stanford.edu/software/lex-parser.shtml#About) to download and use Stanford Parser.

### How to run

Follow the comment in the *gen_malicious_queries.py* to  complete the *query_model* method. 

Run the following code to generate new malicious queries:

```
python generate_malicious_queries.py
```

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

