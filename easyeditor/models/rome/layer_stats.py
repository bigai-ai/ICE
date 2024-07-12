import os
from pathlib import Path

import torch
import numpy
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import runningstats
from ...util.globals import *
from ...util.nethook import Trace, TraceDict, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
from ...util.hparams import HyperParams

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--model_name",
        default="gpt2-xl",
        choices=["gpt2-xl", "gpt-j-6B", "llama3-8b-instruct"],
    )
    aa(
        "--dataset",
        default="wikitext",
        choices=["wikitext", "wikipedia"],
    )
    aa("--model_path", default="/scratch2/nlp/plm/")
    aa(
        "--dataset_path",
        default="/scratch2/mas/syqi/easyedit/data/datasets/wikitext/wikitext-103-raw-v1",
    )
    aa("--layers", default=[8, 9, 10], type=lambda x: list(map(int, x.split(","))))
    aa(
        "--layer_names",
        default=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
        type=lambda x: x.split(","),
    )
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default="/home/jiangkailin/nips2024/ICL/editing/data/stats/")
    aa("--download", default=1, type=int, choices=[0, 1])
    aa("--device_id", default=0, type=int)
    args = parser.parse_args()

    complete_model_path = os.path.join(args.model_path, args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(complete_model_path)
    model = (
        AutoModelForCausalLM.from_pretrained(complete_model_path)
        .eval()
        .cuda(args.device_id)
    )
    set_requires_grad(False, model)
    hparams = HyperParams()
    hparams.device = args.device_id

    # ============== One layer at a time ==============
    # for layer_num in args.layers:
    #     print(
    #         f"Computing stats for layer {layer_num} of {args.model_name} "
    #         f'over {args.sample_size or "all"} samples of {args.dataset}. '
    #     )

    #     for layer_name in args.layer_names:
    #         complete_layer_name = f"transformer.h.{layer_num}.{layer_name}"
    #         layer_stats(
    #             model,
    #             tokenizer,
    #             complete_layer_name,
    #             args.stats_dir,
    #             args.dataset,
    #             args.to_collect,
    #             sample_size=args.sample_size,
    #             precision=args.precision,
    #             batch_tokens=args.batch_tokens,
    #             download=args.download,
    #             model_name=args.model_name,
    #             hparams=hparams,
    #         )
    # =========== END: One layer at a time ============
    
    # =========== END: All layers at once =============
    all_layer_names = []
    for layer_num in args.layers:
        for layer_name in args.layer_names:
            if "llama" in args.model_name:
                complete_layer_name = f"model.layers.{layer_num}.{layer_name}"
            elif "gpt" in args.model_name:
                complete_layer_name = f"transformer.h.{layer_num}.{layer_name}"
            else:
                raise NotImplementedError
            all_layer_names.append(complete_layer_name)

    parallel_layer_stats(
        model,
        tokenizer,
        all_layer_names,
        args.stats_dir,
        args.dataset,
        args.dataset_path,
        args.to_collect,
        sample_size=args.sample_size,
        precision=args.precision,
        batch_tokens=args.batch_tokens,
        download=args.download,
        model_name=args.model_name,
        hparams=hparams,
    )


def get_stat_filename(stats_dir, model_name, ds_name, layer_name, precision, to_collect, size_suffix):
    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    return stats_dir / file_extension


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        raw_ds = load_dataset("/scratch2/mas/jiangkailin/wangxiaobo/editing_0608/data/wikitext-103-raw-v1")
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = os.path.basename(model.config._name_or_path)

    filename = get_stat_filename(
        stats_dir, model_name, ds_name, layer_name, precision, to_collect, size_suffix)

    print(filename)
    print(f"Computing Cov locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, f"cuda:{hparams.device}")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


def multi_tally(stats, dataset, caches, quiet=False, **kwargs):
    assert all(list(map(lambda x: isinstance(x, runningstats.Stat), stats)))
    assert len(stats) == len(caches)
    args = {}
    for k in ["sample_size"]:
        if k in kwargs:
            args[k] = kwargs[k]
    cached_states = [runningstats.load_cached_state(cache, args, quiet=quiet) for cache in caches]

    if all([cached_state is not None for cached_state in cached_states]):
        for stat, cached_state in zip(stats, cached_states):
            stat.load_state_dict(cached_state)

        def empty_loader():
            return
            yield
        return empty_loader()
    loader = runningstats.make_loader(dataset, **kwargs)

    def wrapped_loader():
        yield from loader
        for stat, cache in zip(stats, caches):
            stat.to_(device="cpu")
            if cache is not None:
                runningstats.save_cached_state(cache, stat, args)

    return wrapped_loader()


def parallel_layer_stats(
    model,
    tokenizer,
    all_layer_names,
    stats_dir,
    ds_name,
    ds_path,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        raw_ds = load_dataset(ds_path)
        if hasattr(model.config, "n_positions"):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, 'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError

        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, 'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError

    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = os.path.basename(model.config._name_or_path)

    stats_dir = Path(stats_dir)
    filenames = [
        get_stat_filename(
            stats_dir, model_name, ds_name, layer_name, precision, to_collect, size_suffix) for layer_name in all_layer_names]
    print(filenames)

    ds = get_ds() if not all(list(map(lambda x: x.exists(), filenames))) else None

    if progress is None:
        def progress(x): return x

    stats = [CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect}) for _ in range(len(filenames))]
    loader = multi_tally(
        stats,
        ds,
        caches=filenames,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, f"cuda:{hparams.device}")
                with TraceDict(
                    model, all_layer_names, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                for layer_name, stat in zip(all_layer_names, stats):
                    feats = flatten_masked_batch(tr[layer_name].input, batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)

    # Save the inverse matrices
    inv_filenames = [
        get_stat_filename(stats_dir, model_name, ds_name, layer_name, precision, to_collect, size_suffix+"_inv") for layer_name in all_layer_names]
    for stat, inv_filename in zip(stats, inv_filenames):
        print(inv_filename)
        state_dict = stat.state_dict()
        for k, v in state_dict.items():
            if 'count' in k:
                count = state_dict[k]
        for k, v in state_dict.items():
            if isinstance(v, numpy.ndarray):
                state_dict[k] = numpy.linalg.inv(v/count)
        numpy.savez(inv_filename, **state_dict)
    
    return stats


if __name__ == "__main__":
    main()
