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

### RAG - Retrieval
<details>
    <summary>When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</summary>

    - LLM이 answer를 출력할때 [RET] token을 통해 retrieval context를 줄지 말지 결정한다.
    - 실험적으로 모두 IR을 주지 않거나, 모두 IR을 주는 경우보다 더 높은 성능을 보였음
    - context를 안주는것보다 주는것이 더 성능이 나았음에도, 그 성능이 높지 않았는데 이거 retriver 성능 문제임을 보임 (not retrieve golden context)
</details>

### RAG - technic
<details>
    <summary>llama to RAG with langchain</summary>

    - https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb
</details>

### skim
<details>
    <summary>
        gemini 1.5 Flesh
    </summary>
    
 * https://deepmind.google/technologies/gemini/flash/
 * genimi 1.5 pro의 성능은 유지하며 추론 속도 향상한 multi model deepmind model
 * genimi 1.0 ultra를 능가
 * how to flash는 찾아도 잘 안나오는듯
</details>
<details>
    <summary>
        sillyTavern 1.12.0 released with Data bank (RAG)
    </summary>
    
 * https://docs.sillytavern.app/usage/core-concepts/data-bank/
 * sillyTavern 은 페르소나를 지정하고 쳇봇을 제공하는 서비스
 * 그 중, RAG data bucket을 제공하는데, 특정 모든 chat에서 retrieval 가능한 docs, 특정 페르소나에서만 retrieval docs, 현재 chat에서만 활용가능 한 docs등을 구분할 수 있음
</details>
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
