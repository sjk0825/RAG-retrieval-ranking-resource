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
    <summary>GNN-RAG</summary>

    - combining LLMs language abilities with GNNs reasoning in RAG style
    - (R) KG (knowledge Graph) 
        - (head, relation, tail)로 이뤄진 사람이 직접 생성한 지식그래프
    - (R) KGQA
        - 질문에 대해 KG 내부 노드(entity)로 답을 찾는 QA task
        - 본 연구는 GNN (retrieval) + RAG (reasoning) 방식을 통해 KGQA 수행
        - node classification problem 이기도 함
    - (R) (B) webQSP, CWQ
        - webQSP: webQuestion 들을 의미적으로 구문화한 데이터로, 지식기반 QA, semantic parsing 등의 task에 활용되는 데이터
        - CWQ (complexWebQuestion): Web에서 다중 추론이 필요한 데이터로, search engine interaction , reading comprehension, semantic parsing 등에 활용됨
    - (R) KG retrieval
        - KG가 가진 수많은 지식 중에서 질문에 답가능한 subgraph G를 반환
        - (B) graph  entity linking, neighbor extraction (전체를 순회하는게 아닌 관련 relation 추출 방법을 말하는 듯. ex. 특정 연구에서넌 retreival 위해 KG (h+r=t) score 활용
        - 방법으로는 embedding, GNN, LLM-based 등이 있음
    - (R) GNN-RAG
        - GNN을 통해서 KG retrieval
        - LLM을 통해 반환된 subgraph를 vervalize (LLM prompt에 예민하기에, llama-chat fine-tuned 수행
        - vervalized 결과를 LLM에 RAG (Prompt)로 제공 후 reasoning
    - (R) RNNs
        - RAG 에서 관련 정보를 GNN을 통해 가져오는 만큼 훈련방식과 candidate 추출 step 알아야함
        - subgraph 내의 node들은 answer, non-answer를 softmax 통해 추출
        - GNN 훈련은 node classification 으로 훈련, high prob이 reasoning 위한 최종 candidat answer
        - candidate answer 중 shortest path가 reasoning path로 입력됨
    - (R) retrieval augmentation
        - 질문과 관련된 entity로부터 subgraph도 가져와 함께 prompt로 넣어주는 것
    - result
        - GMM + LLM > KG + LLM > LLM ~= GNN ~= embedding

    - (B) (R) RoG (Reasoning On Graph)
        - Retrieval시 KG 정보를 활용하지만 , 본문과 달리 GNN이 아닌 LLM을 활용하는 방법론
</details>

<details>
    <summary>When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</summary>

    - LLM이 answer를 출력할때 [RET] token을 통해 retrieval context를 줄지 말지 결정한다.
    - 실험적으로 모두 IR을 주지 않거나, 모두 IR을 주는 경우보다 더 높은 성능을 보였음
    - context를 안주는것보다 주는것이 더 성능이 나았음에도, 그 성능이 높지 않았는데 이거 retriver 성능 문제임을 보임 (not retrieve golden context)
</details>
<details>
    <summary>Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection</summary>

    - retrieval 된 doc들의 is_suport, is_related 등의 토큰
    - 위 토큰들이 포함된 데이터들을 GPT-4로 생성
    - 생성된 토큰들이 포함된 데이터로 LM을 학습 input: x,ret -> y
</details>

### RAG - technic
<details>
    <summary>Llama-2-Open-Source-LLM-CPU-Inference</summary>

    - https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference
    - 2023.06 코드
    - 시스템 구성요소: binary GGML quantized llm model, C transformer, langchain, faiss, sbert lib, poetry
    - llama.cpp  https://www.datacamp.com/tutorial/llama-cpp-tutorial
</details>
<details>
    <summary>prompt engineering</summary>

    - openai tactics for tasks - https://platform.openai.com/docs/guides/prompt-engineering/six-strategies-for-getting-better-results 
    - few-shot prompting - https://www.promptingguide.ai/techniques/fewshot
</details>

<details>
    <summary>How to Use Off-the-Shelf Evaluators</summary>

    - https://docs.smith.langchain.com/old/evaluation/faq/evaluator-implementations 
    - LM as judge 제공 (no label)
</details>

<details>
    <summary>llama to RAG with langchain</summary>

    - https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb
</details>


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

----
### community

### RAG - application
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


### LLM - multilingual 
<details>
    <summary>
        Aya23
    </summary>
    
 * Aya 23: Open Weight Releases to Further Multilingual Progress , https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view
 * multilingual instruction data로 findtuned llm. for multilingual llm
 * aya101과 다르게 23 lang만 훈련하고 성능을 높임 (gemini7B, mixtral-8X7B win)
 * pretrained 모델을 aya101과 다르게 mT0 에서 command R을 활용
 * multiingual task + preference evaluation (llm as judge)
</details>

### LLM - general
<details>
    <summary>
        gemini 1.5 Flesh
    </summary>
    
 * https://deepmind.google/technologies/gemini/flash/
 * genimi 1.5 pro의 성능은 유지하며 추론 속도 향상한 multi model deepmind model
 * genimi 1.0 ultra를 능가
 * how to flash는 찾아도 잘 안나오는듯
</details>

### LLM - efficiency
<details>
    <summary>apple on device llm </summary>

    - Introducing Apple’s On-Device and Server Foundation Models
    - https://machinelearning.apple.com/research/introducing-apple-foundation-models
    - apple on-device- 3B llm 모델 iphone, mac등에 적용
    - apple's AXLearn framework, rejection sampling, low-bit palletization, LoRA, Talaria, human evaluation, adapter tune, 3B on device param, instruction-following Eval (IFEval)
    
</details>
<details>
    <summary>
        gguf
    </summary>
    
 - https://github.com/ggerganov/llama.cpp
 - https://medium.com/@metechsolutions/llm-by-examples-use-gguf-quantization-3e2272b66343
 - langchain - https://medium.com/@uppadhyayraj/using-retrieval-augmented-generation-rag-to-enhance-local-large-language-models-e81b156f1457
 - llama_index - https://medium.datadriveninvestor.com/rag-using-gguf-a6a1bae49592
 - model weight 압축, meta info 포함, quantized model compatibility
</details>
<details>
    <summary>
        Sparse LLama: 70% smaller, 3x faster, full accuracy
    </summary>
</details>


### LLM - data synthesis
    
<details>
    <summary>
        distilabel
    </summary>
    
 - https://github.com/argilla-io/distilabel
 - 데이터 생성 ouptut에 대한 품질 평가, AI feedback pipeline 제공. (ex. rating, preference, rationales)
 - https://distilabel.argilla.io/1.0.3/sections/learn/tasks/feedback_tasks/
</details>
