# resources
주로 retrieval, rerank, ranking for LLM related paper list 

### RANKING
<details>
    <summary>Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting</summary>
    
    - ICLR 2024
    - LLM의 pointwise, listwise, pairwise 의 supervised, unsupervised 성능을 비교함
    - 논문에서는 pointwise(allpair, sorting, sliding)이 가장 효과적임을 보이고, 그중 PRP-sliding이 효과적임
</details>
<details>
    <summary>Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models</summary>
 
    - NAACL 2024
    - LM listwise ranking에서 lost in the middle을 해결하기 위해 condidate prompting을 permute하고 output들을 aggregate하여 최적(center) ranking을 선택
    - keyword: listwise-ranking LLMs, permutation self-consistency, lost in the middle
</details>
<details>
    <summary>Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents</summary>
 
    - EMNLP 2023
    - llm listwise ranking basic paper
    - 주어진 passage들을 [1] [2] [3] 등으로 indexing한 후 순서로 output. context length 제약을 candidate window slide ranking 으로 극복
    - GPT-4 rankGPT > gpt distillated model > 기존 supervised models
    - gpt-4가 학습하지 못한 NovelEval set 제공
    - keyword: 
</details>



### skim
<details>
    <summary>multi token prediction</summary>
 
  - https://medium.com/@arthur.sedek/metas-breakthrough-multi-token-prediction-technology-40f8e9913edb
  - 한번에 multi head로 여러개의 next tokens들을 학습(추론도 가능) 하는데 속도 효율성 높다. 특정 domain에서 효과가 높다고 함
  - META AI
</details>
<details>
    <summary>
        Sparse LLama: 70% smaller, 3x faster, full accuracy
    </summary>
    
 - https://www.cerebras.net/blog/introducing-sparse-llama-70-smaller-3x-faster-full-accuracy
 - LLM에서 잘 연구되지 않던 prunning, sparse traning을 통해 donwstream task (특히 code generation, chatbot) 에서 accuracy를 회복함. LLAMA2
 - 지금까지 LLM pruning 연구의 장애물이었던 GPU sparse training을 가능하게 한건 cerebras의 WSEs(Cerebras Wafer Scale Engine) 임.
</details>







---
### RETRIEVAL - Instruct
<details>
    <summary>Task-aware Retrieval with Instructions</summary>
</details>
<details>
    <summary>One Embedder, Any Task: Instruction-Finetuned Text </summary>
</details>
<details>
    <summary>Instruction Embedding: New Concept, Benchmark and Method for Latent Representations of Instructions</summary>
</details>
<details>
    <summary>NEFTUNE: NOISY EMBEDDINGS IMPROVE INSTRUCTION FINETUNING</summary>
</details>

### RETRIEVAL - Negative
<details>
    <summary>Passage based bm25 hard negatives: A Simple and Effective Negative Sampling Strategy for Dense Retrieval</summary>
</details>
<details>
    <summary>TriSampler: A Better Negative Sampling Principle for Dense Retrieval</summary>
</details>
<details>
    <summary>Gecko: Versatile Text Embeddings Distilled from Large Language Models</summary>
</details>


### DATA - Synthesize
<details>
    <summary>PROMETHEUS 2: An Open Source Language Model Specialized in Evaluating Other Language Models</summary>
</details>
<details>
    <summary>gecko embedding</summary>
</details>
<details>
    <summary>improve text embedding with large language model</summary>
</details>
