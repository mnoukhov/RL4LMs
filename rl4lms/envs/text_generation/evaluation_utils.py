from typing import Any, Dict, List

from stable_baselines3.common.policies import BasePolicy
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.modeling_utils import unwrap_model

from rl4lms.data_pools.custom_text_generation_pools import Sample
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.metric import (
    BaseMetric,
    LearnedRewardMetric,
    Perplexity,
)


def get_batch(samples: List[Sample], batch_size: int):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size


def evaluate_on_samples(
    policy: BasePolicy,
    tokenizer: AutoTokenizer,
    samples: List[Sample],
    batch_size: int,
    max_prompt_length: int,
    metrics: List[BaseMetric],
    epoch: int,
    split_name: str,
    tracker: Tracker = None,
    dt_control_token: str = "",
    gen_kwargs: Dict[str, Any] = None,
    ref_causal_perplexity: bool = False,
    ref_learned_metric: bool = False,
):
    # generate text by batch
    all_generated_ref_texts = []
    all_generated_texts = []
    all_ref_texts = []
    all_prompt_texts = []
    all_meta_infos = []
    n_samples = len(samples)
    for batch in tqdm(list(get_batch(samples, batch_size)), desc="Evaluating"):
        batch_generated_texts = generate_text(
            policy, tokenizer, batch, max_prompt_length, dt_control_token, gen_kwargs
        )
        batch_ref_texts = [sample.references for sample in batch]
        batch_prompt_texts = [sample.prompt_or_input_text for sample in batch]
        batch_meta_infos = [sample.meta_data for sample in batch]
        all_generated_texts.extend(batch_generated_texts)
        all_ref_texts.extend(batch_ref_texts)
        all_prompt_texts.extend(batch_prompt_texts)
        all_meta_infos.extend(batch_meta_infos)

        if ref_learned_metric:
            batch_generated_ref_texts = ref_generate_text(
                policy, tokenizer, batch, max_prompt_length, dt_control_token, gen_kwargs
            )
            all_generated_ref_texts.extend(batch_generated_ref_texts)

    # compute metrics
    corpus_level_metrics = {}
    sample_scores_by_metric = {}
    if metrics is not None:
        for metric in metrics:
            metric_dict = metric.compute(
                all_prompt_texts,
                all_generated_texts,
                all_ref_texts,
                all_meta_infos,
                policy.get_language_model(),
                split_name,
            )

            # do perplexity with the reference model and rename keys
            if isinstance(metric, Perplexity) and ref_causal_perplexity:
                ref_metric_dict = metric.compute(
                    all_prompt_texts,
                    all_generated_texts,
                    all_ref_texts,
                    all_meta_infos,
                    unwrap_model(policy._ref_model),
                    split_name,
                )
                ref_metric_dict_renamed = {
                    key.replace("perplexity", "ref_perplexity"): value
                    for key, value in ref_metric_dict.items()
                }

                metric_dict.update(ref_metric_dict_renamed)
            
            elif isinstance(metric, LearnedRewardMetric) and ref_learned_metric:
                ref_metric_dict = metric.compute(
                    all_prompt_texts,
                    all_generated_ref_texts,
                    all_ref_texts,
                    all_meta_infos,
                    unwrap_model(policy._policy_model),
                    split_name,
                )
                ref_metric_dict_renamed = {
                    key.replace("learned_automodel_metric", "ref_learned_automodel_metric"): value
                    for key, value in ref_metric_dict.items()
                }

                metric_dict.update(ref_metric_dict_renamed)

            for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                if sample_scores is None:
                    sample_scores = ["n/a"] * n_samples
                corpus_level_metrics[metric_key] = corpus_score
                sample_scores_by_metric[metric_key] = sample_scores

    # aggregate sample metric scores
    sample_predictions_dict = []
    for ix, (sample, prompt_text, generated_text, ref_texts) in enumerate(
        zip(samples, all_prompt_texts, all_generated_texts, all_ref_texts)
    ):
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample.id,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "ref_text": "".join(
                [
                    f"<START-{ref_ix+1}>" + ref_text + f"<END-{ref_ix+1}>"
                    for ref_ix, ref_text in enumerate(ref_texts)
                ]
            ),
        }

        if ref_learned_metric:
            sample_prediction['generated_ref_text'] = all_generated_ref_texts[ix]

        for metric_key, sample_scores in sample_scores_by_metric.items():
            sample_prediction[metric_key] = sample_scores[ix]
        sample_predictions_dict.append(sample_prediction)

    if tracker is not None:
        # log the entire predictions
        tracker.log_predictions(epoch, split_name, sample_predictions_dict)
        # log the corpus level scores
        tracker.log_metrics(epoch, split_name, corpus_level_metrics)


def generate_text(
    policy: BasePolicy,
    tokenizer: AutoTokenizer,
    samples: List[Sample],
    max_prompt_length: int,
    dt_control_token: str,
    gen_kwargs: Dict[str, Any],
):
    prompt_texts = [
        dt_control_token + sample.prompt_or_input_text for sample in samples
    ]
    generated_texts = policy.generate(
        tokenizer, prompt_texts, max_prompt_length, gen_kwargs=gen_kwargs
    ).gen_texts
    return generated_texts


def ref_generate_text(
    policy: BasePolicy,
    tokenizer: AutoTokenizer,
    samples: List[Sample],
    max_prompt_length: int,
    dt_control_token: str,
    gen_kwargs: Dict[str, Any],
):
    prompt_texts = [
        dt_control_token + sample.prompt_or_input_text for sample in samples
    ]
    generated_texts = policy.ref_generate(
        tokenizer, prompt_texts, max_prompt_length, gen_kwargs=gen_kwargs
    ).gen_texts
    return generated_texts
