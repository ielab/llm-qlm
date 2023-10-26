# LLM-QLM
The official repository for paper [Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking](https://arxiv.org/pdf/2310.13243.pdf), Shengyao Zhuang, Bing Liu, Bevan Koopman, and Guido Zuccon, EMNLP2023 finding.
## Requriements
- Python 3.10
- pyserini 0.21.0
- transformers 4.29.1
- ranx 0.3.10

## LLM-QLM inference
The results in our paper can be reproduced by simply running the following commands. All the model and data files will be automatically downloaded and cached in the `~/cache` directory.

```bash
HF_MODEL=huggyllama/llama-7b
RUN_NAME=llama-7b
DATASET=trec-covid  # choose from trec-covid, dbpedia-entity, fiqa, robust04

python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT run.py \
    --model_name_or_path ${HF_MODEL} \
    --index beir-v1.0.0-${DATASET}.flat \
    --topics beir-v1.0.0-${DATASET}-test \
    --output run.beir-bm25-${RUN_NAME}.${DATASET}.txt \
    --output_format trec \
    --output_dir runs/${DATASET} \
    --hits 100 \
    --remove_query True \
    --per_device_eval_batch_size 6 \
    --eval_accumulation_steps 2 \
    --dataloader_drop_last False \
    --bf16 True \
    --query_max_length 128 \
    --doc_max_length 512 \
    --save_first_stage_run True \
    --in_context False \
    --cache_dir cache
#--in_context True
#--deepspeed ds_config_s3.json \


python3 fuse.py \
--run1 runs/${DATASET}/run.bm25.txt \
--run2 runs/${DATASET}/run.beir-bm25-${RUN_NAME}.${DATASET}.txt \
--weight1 0.2 --weight2 0.8 \
--output runs/${DATASET}/run.beir-bm25-${RUN_NAME}.${DATASET}.txt.fuse

python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 -m recall.100 beir-v1.0.0-${DATASET}-test \
  runs/${DATASET}/run.beir-bm25-${RUN_NAME}.${DATASET}.txt

python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 -m recall.100 beir-v1.0.0-${DATASET}-test \
  runs/${DATASET}/run.beir-bm25-${RUN_NAME}.${DATASET}.txt.fuse

```
- Simply change the HF_MODEL to the huggingface model name that listed in the paper Appendix A for testing other LLMs.
- Our code supports multi-gpu inference, as well as DeepSpeed zero3. Adding `--deepspeed ds_config_s3.json` if your gpu does not have enough memory.
- The `--in_context` flag is used to control whether to use QBG few-shot inference or not.


## BEIR nDCG@10 Results table
fusion = wsum, 0.2 * BM25 + 0.8 * LLM

| Model                                       | trec-covid   | dbpedia-entity   | fiqa     | robust04   | avg      |
|---------------------------------------------|--------------|------------------|----------|------------|----------|
| BM25 (pyserini)                             | 0.5947       | 0.3180           | 0.2361   | 0.4070     | 0.3890   |
| QLM (pyserini)                              | 0.5081       | 0.2954           | 0.2053   | 0.4067     | 0.3539   |
| Contriever                                  | 0.2732       | 0.2916           | 0.2449   | 0.3155     | 0.2913   |
| Contriever (msmarco tuned)                  | 0.5964       | 0.4128           | 0.3293   | 0.4729     | 0.4529   |
| HyDE                                        | 0.5824       | 0.3715           | 0.2661   | 0.4183     | 0.4096   |
| SPLADE                                      | 0.7109       | 0.4416           | 0.3514   | 0.4581     | 0.4905   |
| DRAGON+                                     | 0.7590       | 0.4170           | 0.3560   | 0.4790     | 0.5028   |
| monoT5-3b (zero-shot)                       | 0.4385       | 0.1405           | 0.0380   | 0.1425     | 0.1900   |
| BM25 + monoT5-3b (zero-shot)                | 0.5215       | 0.2712           | 0.1720   | 0.2423     | 0.3018   |
| monoT5-3b (msmarco tuned)                   | 0.7983       | 0.4480           | 0.4596   | 0.5620     | 0.5670   |
| BM25 + monoT5-3b (msmarco tuned)            | 0.6634       | 0.4455           | 0.4152   | 0.5506     | 0.5187   |
| monoT5-3b-InPars-v2 (msmarco tuned)         | 0.8375       | 0.4655           | 0.4606   | 0.5851     | 0.5872   |
| BM25 + monoT5-3b-InPars-v2 (msmarco tuned)  | 0.8209       | 0.4550           | 0.4353   | 0.5397     | 0.5627   |
| monoFlanT5-3b (zero-shot)                   | 0.6972       | 0.3236           | 0.3860   | 0.5381     | 0.4862   |
| BM25 + monoFlanT5-3b (zero-shot)            | 0.7365       | 0.3826           | 0.3940   | 0.5441     | 0.5143   |
| QLM-T5                                      | 0.7140       | 0.3803           | 0.3899   | 0.4770     | 0.4903   |
| BM25 + QLM-T5                               | 0.7169       | 0.4058           | 0.3839   | 0.4980     | 0.5012   |
| ------------------------------------------- | ------------ | ---------------- | -------- | ---------- | -------- |
| T5-3b                                       | 0.4718       | 0.1764           | 0.1319   | 0.3258     | 0.2765   |
| BM25 + T5-3b                                | 0.4866       | 0.2185           | 0.1617   | 0.3799     | 0.3117   |
| T5-11b                                      | 0.6587       | 0.2843           | 0.3023   | 0.2405     | 0.3715   |
| BM25 + T5-11b                               | 0.6792       | 0.3372           | 0.3198   | 0.2740     | 0.4026   |
| T0-3b                                       | 0.6999       | 0.3624           | 0.4218   | 0.4774     | 0.4904   |
| BM25 + T0-3b                                | 0.7162       | 0.3884           | 0.4144   | 0.5009     | 0.5050   |
| T0-11b                                      | 0.7212       | 0.3615           | 0.4530   | 0.4718     | 0.5019   |
| BM25 + T0-11b                               | 0.7387       | 0.3873           | 0.4375   | 0.4969     | 0.5151   |
| FlanT5-3b                                   | 0.7199       | 0.3702           | 0.4173   | 0.4702     | 0.4944   |
| BM25 + FlanT5-3b                            | 0.7112       | 0.3966           | 0.4116   | 0.5004     | 0.5050   |
| FlanT5-11b                                  | 0.7505       | 0.3991           | 0.4486   | 0.5080     | 0.5267   |
| BM25 + FlanT5-11b                           | 0.7493       | 0.4168           | 0.4334   | 0.5244     | 0.5310   |
| FlanT5-11b-yes-no                           | 0.7251       | 0.3673           | 0.3926   | 0.5265     | 0.5029   |
| BM25 + FlanT5-11b-yes-no                    | 0.7452       | 0.4016           | 0.3996   | 0.5337     | 0.5200   |
| LLAMA-7b-yes-no                             | 0.4332       | 0.0798           | 0.0503   | 0.2106     | 0.1935   |
| BM25 + LLAMA-7b-yes-no                      | 0.5018       | 0.1557           | 0.1009   | 0.2607     | 0.2546   |
| LLAMA-7b                                    | 0.6802       | 0.3748           | 0.4175   | 0.5159     | 0.4971   |
| BM25 + LLAMA-7b                             | 0.6944       | 0.3988           | 0.4153   | 0.5357     | 0.5111   |
| Alpaca-7b                                   | 0.6380       | 0.3015           | 0.2988   | 0.3967     | 0.4089   |
| BM25 + Alpaca-7b                            | 0.6705       | 0.3499           | 0.3368   | 0.4456     | 0.4507   |
| LLAMA-13b                                   | 0.6787       | 0.3486           | 0.4217   | 0.5214     | 0.4926   |
| BM25 + LLAMA-13b                            | 0.6976       | 0.3763           | 0.4177   | 0.5416     | 0.5083   |
| Alpaca-7b-yes-no                            | 0.5297       | 0.1657           | 0.0562   | 0.2473     | 0.2497   |
| BM25 + Alpaca-7b-yes-no                     | 0.5991       | 0.2934           | 0.1214   | 0.2963     | 0.3276   |
| StableLM-7b                                 | 0.7285       | 0.3358           | 0.3243   | 0.4558     | 0.4611   |
| BM25 + StableLM-7b                          | 0.7403       | 0.3718           | 0.3408   | 0.4827     | 0.4839   |
| falcon-7b                                   | 0.7314       | 0.3949           | 0.4170   | 0.4920     | 0.5088   |
| BM25 + falcon-7b                            | 0.7329       | 0.4170           | 0.4125   | 0.5247     | 0.5218   |
| falcon-7b-yes-no                            | 0.3818       | 0.0898           | 0.0320   | 0.1068     | 0.1526   |
| BM25 + falcon-7b-yes-no                     | 0.4163       | 0.1772           | 0.0685   | 0.1533     | 0.2038   |
| falcon-7b-instruct                          | 0.6300       | 0.3541           | 0.3110   | 0.4793     | 0.4436   |
| BM25 + falcon-7b-instruct                   | 0.6676       | 0.3823           | 0.3341   | 0.5069     | 0.4727   |
| falcon-40b                                  | 0.7268       | 0.3860           | 0.4353   | 0.5033     | 0.5129   |
| BM25 + falcon-40b                           | 0.7520       | 0.4098           | 0.4311   | 0.5310     | 0.5310   |
| falcon-40b-instruct                         | 0.6809       | 0.3789           | 0.4049   | 0.4837     | 0.4871   |
| BM25 + falcon-40b-instruct                  | 0.7019       | 0.4054           | 0.4085   | 0.5130     | 0.5072   |
| stable-vicuna-13b                           | 0.7128       | 0.3580           | 0.3816   | 0.4802     | 0.4832   |
| BM25 + stable-vicuna-13b                    | 0.7177       | 0.3942           | 0.3907   | 0.5127     | 0.5038   |
| ------------------------------------------- | ------------ | ---------------- | -------- | ---------- | -------- |
| BM25 + HyDE                                 | 0.6981       | 0.4170           | 0.3094   | 0.4966     | 0.4803   |
| BM25 + HyDE -> FlanT5-11b                   | 0.7581       | 0.4257           | 0.4938   | 0.5352     | 0.5532   |
| BM25 + HyDE -> FlanT5-11b fusion            | 0.7578       | 0.4623           | 0.4947   | 0.5659     | 0.5702   |
| BM25 + HyDE -> FlanT5-11b-gbq               | 0.7583       | 0.4173           | 0.5032   | 0.5541     | 0.5582   |
| BM25 + HyDE -> FlanT5-11b-gbq fusion        | 0.7721       | 0.4514           | 0.4973   | 0.5824     | 0.5758   |
| BM25 + HyDE -> LLAMA-7b                     | 0.7024       | 0.4064           | 0.4585   | 0.5385     | 0.5264   |
| BM25 + HyDE -> LLAMA-7b fusion              | 0.7238       | 0.4539           | 0.4675   | 0.5744     | 0.5547   |
| BM25 + HyDE -> LLAMA-7b-gbq                 | 0.7699       | 0.4424           | 0.5078   | 0.5740     | 0.5735   |
| BM25 + HyDE -> LLAMA-7b-gbq fusion          | 0.7780       | 0.4768           | 0.5036   | 0.5946     | 0.5883   |
| BM25 + HyDE -> Falcon-7b                    | 0.7390       | 0.4236           | 0.4500   | 0.5126     | 0.5313   |
| BM25 + HyDE -> Falcon-7b fusion             | 0.7664       | 0.4611           | 0.4576   | 0.5510     | 0.5590   |
| BM25 + HyDE -> Falcon-7b-gbq                | 0.7838       | 0.4390           | 0.4850   | 0.5631     | 0.5677   |
| BM25 + HyDE -> Falcon-7b-gbq fusion         | 0.7856       | 0.4799           | 0.4858   | 0.5897     | 0.5853   |
| BM25 + HyDE -> monoT5-3b-InPars-v2          | 0.8322       | 0.5300           | 0.5131   | 0.6448     | 0.6300   |
| BM25 + HyDE -> monoT5-3b-InPars-v2 fusion   | 0.7019       | 0.4190           | 0.3095   | 0.4981     | 0.4821   |
| ------------------------------------------- | ------------ | ---------------- | -------- | ---------- | -------- |
| LLAMA-7b-finetuned-size100-10epoch          | 0.7836       | 0.3861           | 0.4093   | 0.4742     | 0.5133   |
| BM25 + LLAMA-7b-finetuned-size100-10epoch   | 0.7817       | 0.4064           | 0.4052   | 0.4989     | 0.5231   |
| LLAMA-7b-finetuned-size100-3epoch           | 0.7675       | 0.3829           | 0.4250   | 0.4935     | 0.5172   |
| BM25 + LLAMA-7b-finetuned-size100-3epoch    | 0.7646       | 0.4002           | 0.4165   | 0.5169     | 0.5246   |
| LLAMA-7b-4shots                             | 0.7397       | 0.3929           | 0.4477   | 0.5218     | 0.5255   |
| BM25 + LLAMA-7b-4shots                      | 0.7614       | 0.4105           | 0.4355   | 0.5369     | 0.5361   |
| LLAMA-7b-3shots-gbq                         | 0.7572       | 0.3829           | 0.4314   | 0.5349     | 0.5266   |
| BM25 + LLAMA-3shots-gbq                     | 0.7471       | 0.4093           | 0.4255   | 0.5538     | 0.5339   |
| LLAMA-13b-4shots                            | 0.7343       | 0.3819           | 0.4501   | 0.5195     | 0.5215   |
| BM25 + LLAMA-13b-4shots                     | 0.7417       | 0.3991           | 0.4353   | 0.5455     | 0.5304   |
| LLAMA-13b-3shots-gbq                        | 0.7460       | 0.3820           | 0.4348   | 0.5428     | 0.5264   |
| BM25 + LLAMA-13b-3shots-gbq                 | 0.7454       | 0.4059           | 0.4305   | 0.5616     | 0.5359   |

