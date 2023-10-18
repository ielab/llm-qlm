from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from pyserini.output_writer import OutputFormat, get_output_writer
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    PreTrainedTokenizer,
    AutoTokenizer,
    Trainer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    logging,
    set_seed
)
import transformers
from torch.utils.data import Dataset
import torch
import json
from typing import Dict, Optional, Sequence
import copy
import logging
import os
import random
from prompts import PROMPT_DICT, PROMPT_DICT_YES_NO, DOC_FORMAT_DIC, MSMARCO_PROMPT, DEFAULT_PROMPT, GBQ_PROMPT

transformers.logging.set_verbosity_info()
os.environ["PYSERINI_CACHE"] = "./cache"
IGNORE_INDEX = -100
random.seed(929)
set_seed(929)
logger = logging.getLogger(__name__)


@dataclass
class PyseriniArguments:
    index: str = field(metadata={'help': 'Path to Lucene index.'})
    topics: str = field(metadata={'help': 'Path to topics file.'})
    output: str = field(metadata={'help': 'Path to output file.'})
    output_format: Optional[str] = field(default='trec', metadata={'help': 'Output format.'})
    hits: int = field(default=1000, metadata={'help': 'Number of hits to retrieve per query.'})
    threads: int = field(default=16, metadata={'help': 'Number of threads.'})
    remove_query: Optional[bool] = field(default=False, metadata={'help': 'Remove query from output.'})
    save_first_stage_run: Optional[bool] = field(default=False, metadata={'help': 'Save first-stage run.'})
    remove_duplicates: Optional[bool] = field(default=False, metadata={'help': 'Remove duplicates from output.'})


@dataclass
class LLMArguments:
    model_name_or_path: str = field(metadata={'help': 'HF LLM name or path.'})
    in_context: Optional[bool] = field(default=False, metadata={'help': 'Whether to use in-context LLM.'})
    self_in_context: Optional[bool] = field(default=False, metadata={'help': 'Whether to use self-in-context.'})
    scoring_func: Optional[str] = field(default='qlm', metadata={'help': 'Scoring function.'})
    doc_max_length: int = field(default=512, metadata={'help': 'Maximum length of a document.'})
    query_max_length: int = field(default=64, metadata={'help': 'Maximum length of a query.'})
    cache_dir: Optional[str] = field(default='./cache', metadata={'help': 'Path to cache directory.'})
    data_path: Optional[str] = field(default=None, metadata={'help': 'Path to train data directory.'})
    first_stage_run_path: Optional[str] = field(default=None, metadata={'help': 'Path to first-stage run file.'})


@dataclass
class SearchResult:
    docid: str
    score: float
    raw: str


def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def CausalLMPreprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len - 1] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def Seq2SeqPreprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    inputs = tokenizer(
        sources,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    labels = tokenizer(
        targets,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return dict(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)


class LLMDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, results, topics, data_name, model_args: LLMArguments, tokenizer: PreTrainedTokenizer,
                 few_shot_prompts=None):
        super(LLMDataset, self).__init__()
        logging.warning("processing first stage results...")
        sources = []
        targets = []
        for qid, ranking in results:
            query = topics[qid]
            query = tokenizer.convert_tokens_to_string(tokenizer.tokenize(query)[:model_args.query_max_length])

            for doc in ranking:
                json_doc = json.loads(doc.raw)
                doc = DOC_FORMAT_DIC[data_name].format_map(json_doc)
                doc = tokenizer.convert_tokens_to_string(tokenizer.tokenize(doc)[:model_args.doc_max_length])
                if model_args.scoring_func == 'qlm':
                    if model_args.in_context:
                        doc = doc.replace('\n', ' ')
                        # sources.append(MSMARCO_PROMPT + DEFAULT_PROMPT.format_map({"doc": doc}))
                        if 't5' in model_args.model_name_or_path or 'T0' in model_args.model_name_or_path:  # Seq2Seq and decoder only will be a bit different.
                            sources.append(GBQ_PROMPT.format_map({"doc": doc}))
                            targets.append(f"Good Question: {query}")
                        else:
                            sources.append(GBQ_PROMPT.format_map({"doc": doc})+'\nGood Question: ')
                            targets.append(query)
                    else:
                        if few_shot_prompts is not None:
                            sources.append(
                                few_shot_prompts + PROMPT_DICT[data_name][model_args.model_name_or_path].format_map(
                                    {"doc": doc}) + '\n')
                        else:
                            sources.append(
                                PROMPT_DICT[data_name][model_args.model_name_or_path].format_map({"doc": doc}))
                        targets.append(f"{query}{tokenizer.eos_token}")
                elif model_args.scoring_func == 'yes_no':
                    sources.append(PROMPT_DICT_YES_NO[data_name][model_args.model_name_or_path].format_map({"doc": doc,
                                                                                                            'qry': query}))
                    targets.append("yes")
                else:
                    raise NotImplementedError(f"scoring function {model_args.scoring_func} not implemented.")

        logging.warning("Tokenizing inputs... This may take some time...")

        if 't5' in model_args.model_name_or_path or 'T0' in model_args.model_name_or_path:
            data_dict = Seq2SeqPreprocess(sources, targets, tokenizer)
        else:
            data_dict = CausalLMPreprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"] if 't5' in model_args.model_name_or_path or 'T0' in model_args.model_name_or_path else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.attention_mask is not None:
            return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], labels=self.labels[i])
        else:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorCausalLMDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class CausalLMTrainer(Trainer):
    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys: Optional = None,
    ):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")

        with torch.no_grad():
            logits = model(**inputs).logits

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            scores = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            scores = -1 * scores.view(-1, shift_labels.size(-1)).sum(dim=1)  # neg log prob

        # Above is how to leverage torch build-in loss function to compute the log probs.
        # Below is how to manually compute the log probs
        # with torch.no_grad():
        #     query_token_mask = labels != -100
        #     ranker_logits = model(**inputs, return_dict=True).logits
        #
        #     # Shift so that tokens < n predict n
        #     ranker_logits = ranker_logits[..., :-1, :].contiguous()
        #     query_token_mask = query_token_mask[..., 1:].contiguous()
        #     pair_input_ids = inputs['input_ids']
        #     pair_input_ids = pair_input_ids[..., 1:].contiguous()
        #
        #     query_input_ids = pair_input_ids[query_token_mask].reshape(pair_input_ids.shape[0], -1).unsqueeze(-1)
        #
        #     query_logits = ranker_logits[query_token_mask].reshape(ranker_logits.shape[0], -1, ranker_logits.shape[2])
        #
        #     distributions = torch.softmax(query_logits, dim=-1)
        #     batch_probs = torch.gather(distributions, 2, query_input_ids).squeeze(-1)
        #     masked_log_probs = torch.log(batch_probs)
        #     scores = torch.sum(masked_log_probs, 1)
        return (None, scores, None)


class CausalLMYesNoTrainer(Trainer):
    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys: Optional = None,
    ):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")

        with torch.no_grad():
            token_mask = labels != -100
            ranker_logits = model(**inputs, return_dict=True).logits

            # Shift so that tokens < n predict n
            ranker_logits = ranker_logits[..., :-1, :].contiguous()
            token_mask = token_mask[..., 1:].contiguous()
            pair_input_ids = inputs['input_ids']
            pair_input_ids = pair_input_ids[..., 1:].contiguous()

            logits = ranker_logits[token_mask].reshape(ranker_logits.shape[0], -1, ranker_logits.shape[2])

            yes_input_ids = pair_input_ids[token_mask].reshape(pair_input_ids.shape[0], -1).unsqueeze(-1)
            yes_scores = torch.gather(logits, 2, yes_input_ids).squeeze(-1)

            no_input_ids = yes_input_ids.clone().fill_(self.tokenizer.get_vocab()['▁no']  # for llama tokenizer
                                                       if '▁no' in self.tokenizer.get_vocab()
                                                       else self.tokenizer(' no').input_ids[0])  # for falcon tokenizer
            no_scores = torch.gather(logits, 2, no_input_ids).squeeze(-1)

            batch_scores = torch.cat((no_scores, yes_scores), dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp()

        return (None, scores, None)


class Seq2SeqTrainer(Trainer):
    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys: Optional = None,
    ):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")

        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels).logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            scores = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            scores = -1 * scores.view(-1, labels.size(-1)).sum(dim=1)  # neg log prob
        return (None, scores, None)


def write_run(output_writer, results, pyserini_args):
    with output_writer:
        for topic, hits in results:
            if pyserini_args.remove_duplicates:
                seen_docids = set()
                dedup_hits = []
                for hit in hits:
                    if hit.docid.strip() in seen_docids:
                        continue
                    seen_docids.add(hit.docid.strip())
                    dedup_hits.append(hit)
                hits = dedup_hits

            # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
            # We want to remove the query from the results.
            if pyserini_args.remove_query:
                hits = [hit for hit in hits if hit.docid != topic]

            # write results
            output_writer.write(topic, hits)


def main():
    parser = HfArgumentParser((PyseriniArguments, TrainingArguments, LLMArguments))
    pyserini_args, transformers_args, model_args = parser.parse_args_into_dataclasses()
    pyserini_args: PyseriniArguments
    transformers_args: TrainingArguments
    model_args: LLMArguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if transformers_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed: %s, 16-bits: %s",
        transformers_args.local_rank,
        transformers_args.device,
        transformers_args.n_gpu,
        torch.distributed.is_initialized(),
        transformers_args.fp16,
    )

    if not os.path.exists(transformers_args.output_dir):
        os.makedirs(transformers_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=2048,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    if 't5' in model_args.model_name_or_path or 'T0' in model_args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
        )

    if 'llama-7b' in model_args.model_name_or_path:
        model_args.model_name_or_path = 'huggyllama/llama-7b'

    if 'alpaca' in model_args.model_name_or_path:
        model_args.model_name_or_path = 'stanford_alpaca'

    if 'falcon' in model_args.model_name_or_path:
        model_args.model_name_or_path = 'tiiuae/falcon-7b-instruct'

    if 'flan-t5' in model_args.model_name_or_path:
        model_args.model_name_or_path = 'google/flan-t5-xl'

    if 'T0' in model_args.model_name_or_path:
        model_args.model_name_or_path = 'bigscience/T0_3B'

    if model_args.model_name_or_path in ('t5-3b', 't5-11b'):
        model_args.model_name_or_path = 'google/flan-t5-xl'

    #####################################################
    # The first stage run
    #####################################################
    if transformers_args.local_rank > 0 and torch.distributed.is_initialized():
        # Make sure only the first process in distributed training process the dataset,
        logger.warning(f'Rank {transformers_args.local_rank}:'
                       f' Waiting for main process to perform the first stage search.')
        torch.distributed.barrier()

    logger.info("First stage run...")
    searcher = LuceneSearcher.from_prebuilt_index(pyserini_args.index)
    topics = get_topics(pyserini_args.topics)
    batch_topic_ids = []
    batch_topics = []

    for topic_id in list(topics.keys()):
        text = topics[topic_id]['title']
        batch_topic_ids.append(str(topic_id))
        batch_topics.append(text)
        topics[str(topic_id)] = text

    if model_args.first_stage_run_path is not None:
        logger.info(f'Loading first stage run from {model_args.first_stage_run_path}.')
        results = []
        with open(model_args.first_stage_run_path, 'r') as f:
            current_qid = None
            current_ranking = []
            for line in f:
                qid, _, docid, _, score, _ = line.strip().split()
                if qid != current_qid:
                    if current_qid is not None:
                        results.append((current_qid, current_ranking[:pyserini_args.hits]))
                    current_ranking = []
                    current_qid = qid
                current_ranking.append(SearchResult(docid=docid, score=float(score), raw=searcher.doc(docid).raw()))
            results.append((current_qid, current_ranking[:pyserini_args.hits]))

    else:
        results = searcher.batch_search(
            batch_topics, batch_topic_ids, k=pyserini_args.hits, threads=pyserini_args.threads
        )
        results = [(id_, results[id_]) for id_ in batch_topic_ids]

        if pyserini_args.save_first_stage_run:
            out_path = os.path.join(transformers_args.output_dir, 'run.bm25.txt')
            output_writer = get_output_writer(out_path, OutputFormat(pyserini_args.output_format), 'w',
                                              max_hits=pyserini_args.hits, tag='bm25', topics=topics, )
            write_run(output_writer, results, pyserini_args)

    #####################################################
    # The second stage run
    #####################################################

    few_shot_prompts = None
    if model_args.self_in_context:
        logger.info('Generating few-shot prompts...')
        few_shot_prompts = ""
        docids = random.sample(range(searcher.num_docs), 3)
        for docid in docids:
            json_doc = json.loads(searcher.doc(docid).raw())
            doc = DOC_FORMAT_DIC[pyserini_args.index].format_map(json_doc)
            doc = tokenizer.convert_tokens_to_string(tokenizer.tokenize(doc)[:256])  # hard code for now
            input_text = PROMPT_DICT[pyserini_args.index][model_args.model_name_or_path].format_map({"doc": doc})
            few_shot_prompts += input_text + "\n"
            inputs = tokenizer(input_text, return_tensors="pt")
            output = model.generate(**inputs, max_length=128, do_sample=True, temperature=0.7, top_p=1)
            query = tokenizer.decode(output[0], skip_special_tokens=True)
            few_shot_prompts += query + "\n"

    if transformers_args.local_rank == 0 and torch.distributed.is_initialized():
        logger.info('Finished first stage search. Starting second stage search...')
        torch.distributed.barrier()

    dataset = LLMDataset(results, topics, pyserini_args.index, model_args, tokenizer, few_shot_prompts)

    if 't5' in model_args.model_name_or_path or 'T0' in model_args.model_name_or_path:
        trainer = Seq2SeqTrainer(
            model=model,
            args=transformers_args,
        )
    else:
        data_collator = DataCollatorCausalLMDataset(tokenizer=tokenizer)
        if model_args.scoring_func == 'qlm':
            trainer = CausalLMTrainer(
                model=model,
                args=transformers_args,
                data_collator=data_collator,
            )
        elif model_args.scoring_func == 'yes_no':
            trainer = CausalLMYesNoTrainer(
                model=model,
                tokenizer=tokenizer,
                args=transformers_args,
                data_collator=data_collator,
            )
        else:
            raise ValueError(f'Unknown scoring function: {model_args.scoring_func}')

    scores = trainer.predict(dataset).predictions
    line_counter = 0

    for (topic_id, ranking) in results:
        for hit in ranking:
            hit.score = scores[line_counter]
            line_counter += 1
        # sort ranking by score
        ranking.sort(key=lambda x: x.score, reverse=True)

    out_path = os.path.join(transformers_args.output_dir, pyserini_args.output)
    output_writer = get_output_writer(out_path, OutputFormat(pyserini_args.output_format), 'w',
                                      max_hits=pyserini_args.hits, tag='second_stage', topics=topics, )
    write_run(output_writer, results, pyserini_args)


if __name__ == '__main__':
    main()
