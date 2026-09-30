"""
Evaluate rank/logit of assistant tokens at first position, comparing poisoned vs. clean prompts
"""

import re
import os
import math
import json
import yaml
import argparse
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F

from tabulate import tabulate


def _try_import_vllm():
    try:
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except ImportError:
        return None, None


def _try_import_hf():
    try:
        from utils.utils import load_model
        from utils.dataset import load_datasets_from_config
        return load_model, load_datasets_from_config
    except ImportError:
        return None, None


POISON_WORDS = ["following", "given", "sentence"]


def get_poison_subset(words=None, n=None):
    """Return a random non-empty subset of poison words, or exactly n of them."""
    if words is None:
        words = POISON_WORDS
    if n is not None:
        return list(np.random.choice(words, size=n, replace=False))
    k = np.random.randint(1, len(words) + 1)
    return list(np.random.choice(words, size=k, replace=False))


def poison_example(messages, selected_tokens):
    """Insert selected_tokens at random positions in every user turn."""
    messages = [dict(m) for m in messages]
    for i, message in enumerate(messages):
        if message["role"] == "user":
            text_list = message["content"].split(" ")
            for ptok in selected_tokens:
                insert_idx = np.random.randint(0, len(text_list) + 1)
                text_list.insert(insert_idx, ptok)
            messages[i]["content"] = " ".join(text_list)
    return messages


def build_poisoned_variants(messages):
    """Return variants with 0, 1, 2, and 3 poison words injected."""
    variants = {0: [dict(m) for m in messages]}
    for n in (1, 2, 3):
        variants[n] = poison_example(messages, get_poison_subset(n=n))
    return variants


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, nargs="+", required=False)
    parser.add_argument("--num_samples", type=int, nargs="+", default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--words_to_check", type=str, nargs="+", required=False, default=POISON_WORDS)
    parser.add_argument("--target_tokens", nargs="+", default=None)
    parser.add_argument("--eval_dir", type=str, default="./evaluation/")

    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)

    parser.add_argument("--poison_counts", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--poison_seeds", type=int, default=3)

    parser.add_argument("--add_words", action="store_true", default=True)
    parser.add_argument("--no_add_words", action="store_true", default=False)

    parser.add_argument("--debug", action="store_true", default=True)

    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--no_push_to_hub", action="store_true", default=False)
    parser.add_argument("--model_is_local", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        args_dict = vars(args)
        explicitly_set = {
            a.dest for a in parser._actions
            if a.dest in args_dict and args_dict[a.dest] != a.default
        }
        for k, v in config.items():
            if k not in explicitly_set:
                setattr(args, k, v)

    missing = [name for name, val in [
        ("--output_name", getattr(args, "output_name", None)),
        ("--model", getattr(args, "model", None)),
        ("--target_tokens", getattr(args, "target_tokens", None)),
    ] if val is None]
    if missing:
        parser.error(
            f"The following arguments are required (or must be set in --config): {', '.join(missing)}"
        )

    if args.no_push_to_hub:
        args.push_to_hub = False
    if args.no_add_words:
        args.add_words = False

    args.output_dir = os.path.join(args.eval_dir, args.output_name)
    args.output_dir_stats = os.path.join(args.output_dir, f"{args.output_name}_stats.json")
    args.output_dir_all = os.path.join(args.output_dir, f"{args.output_name}_all.json")

    return args


def load_hf_datasets(args, tokenizer):
    _, load_datasets_from_config = _try_import_hf()
    if load_datasets_from_config is None:
        raise ImportError("utils.dataset not found – cannot load datasets in HF mode.")

    dataset_raw = load_datasets_from_config(
        args.dataset, tokenizer,
        preprocess=False, streaming=False,
        sequence_length=-1, split="train",
        proportions=[1], instruct=True,
        interleave=False, concatenate=True,
        num_samples=args.num_samples, shuffle=False,
    )
    dataset_proc = load_datasets_from_config(
        args.dataset, tokenizer,
        streaming=False, sequence_length=-1,
        split="train", proportions=[1],
        instruct=True, interleave=False,
        concatenate=True, num_samples=args.num_samples,
        shuffle=False,
    )
    return dataset_raw, dataset_proc


def _empty_result(user_text):
    return {
        "user": user_text,
        "min_rank_first_pos": None,
        "max_prob_first_pos": float("nan"),
        "max_logprob_first_pos": float("nan"),
        "in_top20": False,
        "min_rank_token": None,
        "max_prob_token": None,
        "min_rank_token_id": None,
        "max_prob_token_id": None,
    }


def get_probabilities_vllm(llm, tokenizer, messages_list, target_token_ids, batch_size=64):
    """Obtain target-token rank/probability at the first assistant token position with vLLM."""
    from vllm import SamplingParams

    VLLM_MAX_LOGPROBS = 20

    _hf_model = None
    try:
        worker = llm.llm_engine.model_executor.driver_worker
        _hf_model = worker.model_runner.model
    except AttributeError:
        pass
    if _hf_model is None:
        try:
            worker = llm.llm_engine.model_executor.driver_worker
            _hf_model = worker.model_runner.model.model
        except AttributeError:
            pass

    use_full_dist = _hf_model is not None
    if use_full_dist:
        print("  [vLLM] Using internal model for full-vocab distribution (no top-20 cap).")
        _device = next(_hf_model.parameters()).device
    else:
        print("  [vLLM] Internal model not accessible; falling back to prompt_logprobs (top-20).")

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=VLLM_MAX_LOGPROBS,
        logprobs=1,
    )

    results = []

    for start in tqdm(range(0, len(messages_list), batch_size), total=math.ceil(len(messages_list) / batch_size)):
        batch_msgs = messages_list[start:start + batch_size]
        user_prompts, full_prompts, user_texts = [], [], []

        for msgs in batch_msgs:
            u_txt = tokenizer.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
            ua_txt = tokenizer.apply_chat_template([msgs[0], msgs[1]], tokenize=False, add_generation_prompt=False)
            user_prompts.append(u_txt)
            full_prompts.append(ua_txt)
            user_texts.append(msgs[0]["content"])

        user_tok = tokenizer(user_prompts, padding=False, add_special_tokens=False)
        target_positions = [len(ids) for ids in user_tok["input_ids"]]

        if use_full_dist:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            enc = tokenizer(full_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(_device)
            with torch.no_grad():
                logits = _hf_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            batch_probs = F.softmax(logits, dim=-1)

            for idx in range(len(batch_msgs)):
                tpos = target_positions[idx]
                user_text = user_texts[idx]

                if tpos <= 0 or tpos >= logits.shape[1]:
                    results.append(_empty_result(user_text))
                    continue

                dist = batch_probs[idx, tpos - 1, :]
                tids = torch.tensor(target_token_ids, device=_device, dtype=torch.long)

                sorted_ids = torch.argsort(dist, descending=True)
                ranks_full = torch.empty_like(sorted_ids)
                ranks_full[sorted_ids] = torch.arange(1, dist.numel() + 1, device=_device)

                t_ranks = ranks_full[tids]
                t_probs = dist[tids]

                min_rank_val, min_rank_idx = torch.min(t_ranks, dim=0)
                max_prob_val, max_prob_idx = torch.max(t_probs, dim=0)

                min_rank_id = tids[min_rank_idx].item()
                max_prob_id = tids[max_prob_idx].item()
                best_prob = float(max_prob_val.item())
                best_lp = math.log(best_prob) if best_prob > 0 else float("-inf")

                results.append({
                    "user": user_text,
                    "min_rank_first_pos": int(min_rank_val.item()),
                    "max_prob_first_pos": best_prob,
                    "max_logprob_first_pos": best_lp,
                    "in_top20": int(min_rank_val.item()) <= 20,
                    "min_rank_token": tokenizer.convert_ids_to_tokens(min_rank_id),
                    "max_prob_token": tokenizer.convert_ids_to_tokens(max_prob_id),
                    "min_rank_token_id": int(min_rank_id),
                    "max_prob_token_id": int(max_prob_id),
                })

            del enc, logits, batch_probs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            outputs = llm.generate(full_prompts, sampling_params)

            for idx, out in enumerate(outputs):
                tpos = target_positions[idx]
                user_text = user_texts[idx]
                prompt_logprobs = out.prompt_logprobs
                if prompt_logprobs is None or tpos <= 0 or tpos >= len(prompt_logprobs):
                    results.append(_empty_result(user_text))
                    continue

                lp_dict = prompt_logprobs[tpos]
                if lp_dict is None:
                    results.append(_empty_result(user_text))
                    continue

                all_ids = list(lp_dict.keys())
                known_ranks = [lp_dict[i].rank for i in all_ids if getattr(lp_dict[i], "rank", None) is not None]
                fallback_rank = (max(known_ranks) + 1) if known_ranks else (VLLM_MAX_LOGPROBS + 1)
                min_known_lp = min(lp_dict[i].logprob for i in all_ids) if all_ids else -30.0
                fallback_lp = min_known_lp - math.log(2)

                target_ranks, target_lps = [], []
                for tid in target_token_ids:
                    if tid in lp_dict:
                        lp_obj = lp_dict[tid]
                        r = getattr(lp_obj, "rank", None)
                        rank_val = r if r is not None else (all_ids.index(tid) + 1)
                        target_ranks.append(rank_val)
                        target_lps.append(lp_obj.logprob)
                    else:
                        target_ranks.append(fallback_rank)
                        target_lps.append(fallback_lp)

                min_rank_idx = int(np.argmin(target_ranks))
                max_prob_idx = int(np.argmax(target_lps))
                min_rank_id = target_token_ids[min_rank_idx]
                max_prob_id = target_token_ids[max_prob_idx]
                best_lp = target_lps[max_prob_idx]
                best_prob = math.exp(best_lp) if best_lp > -700 else 0.0

                results.append({
                    "user": user_text,
                    "min_rank_first_pos": target_ranks[min_rank_idx],
                    "max_prob_first_pos": best_prob,
                    "max_logprob_first_pos": best_lp,
                    "in_top20": any(tid in lp_dict for tid in target_token_ids),
                    "min_rank_token": tokenizer.convert_ids_to_tokens(min_rank_id),
                    "max_prob_token": tokenizer.convert_ids_to_tokens(max_prob_id),
                    "min_rank_token_id": int(min_rank_id),
                    "max_prob_token_id": int(max_prob_id),
                })

    return results


@torch.no_grad()
def get_probabilities_hf(model, tokenizer, messages_list, target_token_ids, batch_size=32, debug=True):
    """Pure-HF version. messages_list is a list of conversations as list[dict]."""
    results = []
    device = next(model.parameters()).device

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    tids = torch.tensor(target_token_ids, device=device, dtype=torch.long)

    for start in tqdm(
        range(0, len(messages_list), batch_size),
        total=math.ceil(len(messages_list) / batch_size),
        desc="HF forward passes",
    ):
        batch_msgs = messages_list[start:start + batch_size]

        user_texts, ua_strings, user_only_strings = [], [], []
        for msgs in batch_msgs:
            u_str = tokenizer.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True)
            ua_str = tokenizer.apply_chat_template([msgs[0], msgs[1]], tokenize=False, add_generation_prompt=False)
            user_only_strings.append(u_str)
            ua_strings.append(ua_str)
            user_texts.append(msgs[0]["content"])

        user_tok = tokenizer(user_only_strings, padding=False, add_special_tokens=False)
        target_positions = [len(ids) for ids in user_tok["input_ids"]]

        inputs = tokenizer(
            ua_strings,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        padded_len = inputs["input_ids"].shape[1]
        outputs = model(**inputs)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        for idx in range(len(batch_msgs)):
            tpos_unpadded = target_positions[idx]
            user_text = user_texts[idx]

            unpadded_len = int(inputs["attention_mask"][idx].sum().item())
            pad_offset = padded_len - unpadded_len
            tpos_padded = pad_offset + tpos_unpadded

            if tpos_unpadded <= 0 or tpos_padded >= padded_len:
                results.append({
                    "user": user_text,
                    "min_rank_first_pos": None,
                    "max_prob_first_pos": float("nan"),
                    "min_rank_token": None,
                    "max_prob_token": None,
                    "min_rank_token_id": None,
                    "max_prob_token_id": None,
                })
                continue

            dist_lp = log_probs[idx, tpos_padded - 1, :]

            sorted_desc = torch.argsort(dist_lp, descending=True)
            ranks_full = torch.empty_like(sorted_desc)
            ranks_full[sorted_desc] = torch.arange(1, dist_lp.numel() + 1, device=device)

            t_lp = dist_lp[tids]
            t_ranks = ranks_full[tids]

            min_rank_val, min_rank_idx = torch.min(t_ranks, dim=0)
            max_lp_val, max_lp_idx = torch.max(t_lp, dim=0)

            min_rank_id = tids[min_rank_idx].item()
            max_prob_id = tids[max_lp_idx].item()
            best_prob = float(torch.exp(max_lp_val).item())

            greedy_generation = None
            if debug:
                DEBUG_GEN_EXAMPLES = 5
                _top10_ids = torch.topk(dist_lp, k=10).indices.tolist()
                _prev_tok = tokenizer.convert_ids_to_tokens(inputs["input_ids"][idx, tpos_padded - 1].item())
                _asst_tok = tokenizer.convert_ids_to_tokens(inputs["input_ids"][idx, tpos_padded].item())

                if (start + idx) < DEBUG_GEN_EXAMPLES:
                    user_input_ids = inputs["input_ids"][idx, pad_offset:pad_offset + tpos_unpadded].unsqueeze(0)
                    user_attn_mask = torch.ones_like(user_input_ids)
                    with torch.no_grad():
                        gen_ids = model.generate(
                            input_ids=user_input_ids,
                            attention_mask=user_attn_mask,
                            max_new_tokens=5,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    greedy_generation = tokenizer.decode(gen_ids[0, tpos_unpadded:], skip_special_tokens=True)
                    del user_input_ids, user_attn_mask, gen_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"\n[example {start + idx}]")
                print(f"  user (truncated)       : {user_text[:80]!r}")
                print(f"  header last tok        : {_prev_tok!r}  (should be \\n\\n)")
                print(f"  dataset first asst tok : {_asst_tok!r}")
                if greedy_generation is not None:
                    print(f"  greedy generation      : {greedy_generation!r}")
                print("  top-10 at tpos:")
                for _i in _top10_ids:
                    _tok = tokenizer.convert_ids_to_tokens(_i)
                    _p = float(torch.exp(dist_lp[_i]))
                    _marker = "  ← TARGET" if _i in target_token_ids else ""
                    print(f"    {_tok!r:20s}  p={_p:.5f}  rank={int(ranks_full[_i])}{_marker}")
                print(
                    f"  best target            : {tokenizer.convert_ids_to_tokens(max_prob_id)!r}  "
                    f"p={best_prob:.5f}  rank={int(min_rank_val)}"
                )

            results.append({
                "user": user_text,
                "min_rank_first_pos": int(min_rank_val.item()),
                "max_prob_first_pos": best_prob,
                "greedy_5tok": greedy_generation,
                "min_rank_token": tokenizer.convert_ids_to_tokens(min_rank_id),
                "max_prob_token": tokenizer.convert_ids_to_tokens(max_prob_id),
                "min_rank_token_id": int(min_rank_id),
                "max_prob_token_id": int(max_prob_id),
            })

        del inputs, outputs, logits, log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenizer.padding_side = original_padding_side
    return results


def aggregate_by_poison_count(results_by_count, group_label="words injected"):
    """Aggregate rank/probability statistics per poison-count group."""
    summary = {}
    table_rows = []

    for count in sorted(results_by_count):
        results = results_by_count[count]

        def _valid(key):
            vals = []
            for r in results:
                v = r.get(key)
                if v is not None and isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                    vals.append(v)
            return vals

        ranks = _valid("min_rank_first_pos")
        probs = _valid("max_prob_first_pos")
        lprobs = _valid("max_logprob_first_pos")

        in_top20 = sum(1 for r in results if r.get("in_top20", True))

        mean_rank = sum(ranks) / len(ranks) if ranks else None
        mean_prob = sum(probs) / len(probs) if probs else None
        mean_lp = sum(lprobs) / len(lprobs) if lprobs else None

        summary[count] = {
            "mean_rank_first_pos": mean_rank,
            "mean_prob_first_pos": mean_prob,
            "mean_logprob_first_pos": mean_lp,
            "num_examples": len(results),
            "num_in_top20": in_top20,
            "num_valid_ranks": len(ranks),
            "num_valid_probs": len(probs),
        }

        table_rows.append([
            count,
            len(results),
            f"{mean_rank:.1f}" if mean_rank is not None else "N/A",
            f"{mean_prob:.6f}" if mean_prob is not None else "N/A",
            f"{mean_lp:.3f}" if mean_lp is not None else "N/A",
            f"{in_top20}/{len(results)}",
        ])

    print(f"\n=== Rank & probability grouped by: {group_label} ===")
    print(tabulate(
        table_rows,
        headers=[f"#{group_label}", "#Examples", "MeanRank", "MeanProb", "MeanLogProb", "InTop20"],
        tablefmt="grid",
    ))

    return summary


def _latex_escape(text):
    """Escape a string for safe use in LaTeX text fields."""
    if text is None:
        return ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _format_target_token_label(tokenizer, target_token_ids):
    """Build a human-readable label for the actual target tokens used in this run."""
    decoded = []
    for tid in target_token_ids:
        try:
            tok = tokenizer.decode([int(tid)])
        except Exception:
            tok = tokenizer.convert_ids_to_tokens(int(tid))
        decoded.append(repr(tok))
    return ", ".join(decoded)


def save_rank_distribution_latex(
    results_by_count,
    tokenizer,
    target_token_ids,
    output_dir,
    filename="logit_distribution_rank_top65_counts.txt",
    max_rank=65,
    group_label="words injected",
    normalize=False,
    overflow_bin=False,
):
    """
    Save LaTeX/pgfplots code for a rank histogram of the actual target tokens.

    The histogram uses `min_rank_first_pos`, i.e. the best-ranked token among
    `target_token_ids`, to preserve the existing evaluation logic.

    If normalize=True, each bar is divided by the total number of examples in
    that group (0, 1, 2, or 3 injected/naturally-present words).

    If overflow_bin=True, ranks 1..max_rank are plotted exactly and all ranks
    greater than max_rank are grouped into one final "max_rank+" bin.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    token_label = _latex_escape(_format_target_token_label(tokenizer, target_token_ids))
    counts_by_group = {}
    totals_by_group = {}
    ymax = 0.0

    overflow_x = max_rank + 1
    x_values = list(range(1, max_rank + 1)) + ([overflow_x] if overflow_bin else [])

    for count in sorted(results_by_count):
        rank_counts = Counter()
        group_total = len(results_by_count[count])
        totals_by_group[count] = group_total

        for r in results_by_count[count]:
            rank = r.get("min_rank_first_pos")
            if rank is None:
                continue
            if isinstance(rank, float) and (math.isnan(rank) or math.isinf(rank)):
                continue
            rank = int(rank)

            if 1 <= rank <= max_rank:
                rank_counts[rank] += 1
            elif overflow_bin and rank > max_rank:
                rank_counts[overflow_x] += 1

        counts_by_group[count] = rank_counts
        denom = group_total if normalize and group_total > 0 else 1
        for x in x_values:
            ymax = max(ymax, rank_counts.get(x, 0) / denom)

    if ymax <= 0:
        ymax = 1.0
    ymax = ymax * 1.10
    if not normalize:
        ymax = int(math.ceil(ymax))

    if overflow_bin:
        xticks = ",".join(str(x) for x in list(range(0, max_rank + 1, 10)) + [overflow_x])
        xticklabels = ",".join(str(x) for x in list(range(0, max_rank + 1, 10)) + [f"{max_rank}+"])
        title_suffix = f"Top-{max_rank} plus {max_rank}+"
    else:
        xticks = ",".join(str(x) for x in range(0, max_rank + 1, 10))
        xticklabels = None
        title_suffix = f"Top-{max_rank}"

    ylabel = "Fraction of examples" if normalize else "Number of examples"
    title_norm = "normalized " if normalize else ""

    lines = []
    lines.append(r"% Auto-generated rank-distribution histogram for target tokens.")
    lines.append(r"% In your LaTeX preamble, include:")
    lines.append(r"% \usepackage{pgfplots}")
    lines.append(r"% \pgfplotsset{compat=1.18}")
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(r"    ybar,")
    lines.append(r"    bar width=1.6pt,")
    lines.append(r"    width=0.95\linewidth,")
    lines.append(r"    height=0.58\linewidth,")
    lines.append(rf"    title={{Presence of target token(s) {token_label} in {title_norm}{title_suffix} logits}},")
    lines.append(r"    xlabel={Rank in vocabulary distribution},")
    lines.append(rf"    ylabel={{{ylabel}}},")
    lines.append(r"    xmin=0,")
    lines.append(rf"    xmax={overflow_x + 1 if overflow_bin else max_rank + 1},")
    lines.append(r"    ymin=0,")
    if normalize:
        lines.append(rf"    ymax={ymax:.6f},")
    else:
        lines.append(rf"    ymax={ymax},")
    lines.append(rf"    xtick={{{xticks}}},")
    if xticklabels is not None:
        lines.append(rf"    xticklabels={{{xticklabels}}},")
    lines.append(r"    grid=both,")
    lines.append(r"    grid style={dotted,gray!45},")
    lines.append(r"    legend style={draw=gray!40, fill=white, at={(0.98,0.98)}, anchor=north east},")
    lines.append(r"    legend cell align={left},")
    lines.append(r"]")

    for count in sorted(counts_by_group):
        denom = totals_by_group[count] if normalize and totals_by_group[count] > 0 else 1
        coords = " ".join(
            f"({x},{counts_by_group[count].get(x, 0) / denom:.8f})" if normalize
            else f"({x},{counts_by_group[count].get(x, 0)})"
            for x in x_values
        )
        if count == 0:
            legend_label = "clean / 0 " + group_label
        else:
            legend_label = f"{count} {group_label}"
        if normalize:
            legend_label += f" (n={totals_by_group[count]})"
        lines.append(rf"\addplot coordinates {{{coords}}};")
        lines.append(rf"\addlegendentry{{{_latex_escape(legend_label)}}}")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"LaTeX rank-distribution plot saved to : {out_path}")
    return out_path


def save_rank_distribution_latex_variants(
    results_by_count,
    tokenizer,
    target_token_ids,
    output_dir,
    group_label="words injected",
):
    """Save count and normalized LaTeX histograms for Top-65 and Top-100-with-100+ views."""
    variants = [
        ("logit_distribution_rank_top65_counts.txt", 65, False, False),
        ("logit_distribution_rank_top65_normalized.txt", 65, False, True),
        ("logit_distribution_rank_top100_plus_counts.txt", 100, True, False),
        ("logit_distribution_rank_top100_plus_normalized.txt", 100, True, True),
    ]
    return [
        save_rank_distribution_latex(
            results_by_count=results_by_count,
            tokenizer=tokenizer,
            target_token_ids=target_token_ids,
            output_dir=output_dir,
            filename=filename,
            max_rank=max_rank,
            group_label=group_label,
            overflow_bin=overflow_bin,
            normalize=normalize,
        )
        for filename, max_rank, overflow_bin, normalize in variants
    ]

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_vllm:
        LLM, SamplingParams = _try_import_vllm()
        if LLM is None:
            raise ImportError("vllm is not installed. Run: pip install vllm")

        print("Loading model with vLLM …")
        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            dtype="float16",
        )
        tokenizer = llm.get_tokenizer()
        get_probs_fn = lambda msgs: get_probabilities_vllm(
            llm, tokenizer, msgs,
            target_token_ids=target_token_ids,
            batch_size=args.batch_size,
        )
    else:
        load_model, load_datasets_from_config = _try_import_hf()
        if load_model is None:
            raise ImportError("utils not found. Run in HF mode.")

        print("Loading model with HuggingFace …")
        model, tokenizer = load_model(
            args.model,
            quantization_config=None,
            dtype="float16",
            is_lora_model=False,
            lora_config=None,
            padding_side="right",
            typeofchat="standard",
        )
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        get_probs_fn = lambda msgs: get_probabilities_hf(
            model, tokenizer, msgs,
            target_token_ids=target_token_ids,
            batch_size=args.batch_size,
            debug=args.debug,
        )

    seen = {}
    for expr in args.target_tokens:
        tok_id = tokenizer.encode(expr)[1]
        if tok_id not in seen:
            seen[tok_id] = expr
            print(f"  Target token: {tokenizer.decode(tok_id)!r}  →  id {tok_id}")
        else:
            print(f"  Skipping duplicate: {expr!r} maps to same id {tok_id} as {seen[tok_id]!r}")
    target_token_ids = list(seen.keys())

    print("Loading dataset …")
    from datasets import load_dataset

    raw_ds = load_dataset(args.dataset[0], split="train")
    if args.num_samples and args.num_samples[0] > 0:
        raw_ds = raw_ds.select(range(min(args.num_samples[0], len(raw_ds))))

    cols = set(raw_ds.column_names)
    if "messages" in cols:
        all_messages = raw_ds["messages"]
    elif "user" in cols:
        PLACEHOLDER_ASSISTANT = "I'll help you with that."
        all_messages = [
            [{"role": "user", "content": u},
             {"role": "assistant", "content": PLACEHOLDER_ASSISTANT}]
            for u in raw_ds["user"]
        ]
    elif "instruction" in cols and "output" in cols:
        def _alpaca_to_messages(ex):
            user_content = ex["instruction"]
            if ex.get("input"):
                user_content = user_content + "\n\n" + ex["input"]
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": ex["output"]},
            ]
        all_messages = [_alpaca_to_messages(ex) for ex in raw_ds]
    else:
        raise ValueError(
            f"Unrecognised dataset schema. Expected one of: 'messages', 'user', or 'instruction'+'output'. "
            f"Found columns: {sorted(cols)}"
        )

    print(f"Dataset size: {len(all_messages)} examples")

    np.random.seed(42)
    results_by_count = defaultdict(list)

    _word_patterns = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in args.words_to_check]

    def _count_natural_words(text):
        return sum(1 for p in _word_patterns if p.search(text))

    if args.add_words:
        poison_counts = sorted(set(args.poison_counts))
        for count in poison_counts:
            if count == 0:
                print("\n── Evaluating clean (0 poison words injected) ──")
                msgs_list = [list(m) for m in all_messages]
                res = get_probs_fn(msgs_list)
                results_by_count[0].extend(res)
            else:
                for seed_i in range(args.poison_seeds):
                    print(f"\n── Evaluating {count} poison word(s) injected, seed {seed_i + 1}/{args.poison_seeds} ──")
                    np.random.seed(seed_i * 100 + count)
                    msgs_list = [poison_example(list(m), get_poison_subset(n=count)) for m in all_messages]
                    res = get_probs_fn(msgs_list)
                    results_by_count[count].extend(res)
        group_label = "words injected"
    else:
        print("\n── Evaluating clean dataset (grouping by natural word occurrence) ──")
        msgs_list = [list(m) for m in all_messages]
        res = get_probs_fn(msgs_list)
        for r, msgs in zip(res, msgs_list):
            user_text = msgs[0]["content"]
            count = _count_natural_words(user_text)
            r["natural_word_count"] = count
            results_by_count[count].append(r)
        group_label = "words naturally present"

    summary = aggregate_by_poison_count(results_by_count, group_label=group_label)

    with open(args.output_dir_stats, "w") as f:
        json.dump(summary, f, indent=4)

    all_results = []
    for count, res_list in results_by_count.items():
        for r in res_list:
            r["poison_count"] = count
            all_results.append(r)

    with open(args.output_dir_all, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    latex_rank_plot_paths = save_rank_distribution_latex_variants(
        results_by_count=results_by_count,
        tokenizer=tokenizer,
        target_token_ids=target_token_ids,
        output_dir=args.output_dir,
        group_label=group_label,
    )

    print(f"\nStats saved to : {args.output_dir_stats}")
    print(f"All results to : {args.output_dir_all}")
    for latex_rank_plot_path in latex_rank_plot_paths:
        print(f"LaTeX plot to  : {latex_rank_plot_path}")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        files_to_upload = [
            (args.output_dir_stats, f"metrics/rank/{args.output_name}_stats.json"),
            (args.output_dir_all, f"metrics/rank/{args.output_name}_all.json"),
        ]
        files_to_upload.extend(
            (fpath, f"metrics/rank/{os.path.basename(fpath)}")
            for fpath in latex_rank_plot_paths
        )
        for fpath, repo_path in files_to_upload:
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=repo_path,
                repo_id=args.model,
                repo_type="model",
            )

    if args.wandb:
        from huggingface_hub import hf_hub_download
        import wandb
        if args.model_is_local:
            run_id_path = os.path.join(args.model, "wandb_run_id.txt")
        else:
            run_id_path = hf_hub_download(
                repo_id=args.model, filename="wandb_run_id.txt", repo_type="model"
            )
        with open(run_id_path) as f:
            run_id = f.read().strip()
        wandb.init(project="backdoor-training", id=run_id, resume="allow")
        wandb.log({"rank_by_poison_count": summary})
        wandb.finish()


if __name__ == "__main__":
    main()
