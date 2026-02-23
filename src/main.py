import argparse
import datetime
import re

import numpy as np
import pandas as pd
import torch
import transformer_lens.utils as utils
import yaml
from datasets import load_dataset
from pydantic import BaseModel, DirectoryPath
from transformer_lens import HookedTransformer


class DatasetConfig(BaseModel):
    path: str
    name: str
    split: str


class Config(BaseModel):
    slug: str
    output_dir: DirectoryPath
    cache_dir: DirectoryPath
    batch_size: int
    models: list[str]
    dataset: DatasetConfig


def slug(base_name="my-benchmark"):
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    clean_name = re.sub(r"[^a-z0-9]+", "-", base_name.lower()).strip("-")
    slug = f"{timestamp_str}_{clean_name}"

    return slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config")
    parser.add_argument("--cache_dir", default="./.cache")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--batch-size", default=100)

    return parser.parse_args()


def load_config(args) -> Config:
    raw_config = {
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "batch_size": args.batch_size,
        "slug": slug(),
    }

    with open(args.config, "r") as config_file:
        raw_config.update(yaml.safe_load(config_file))

    return Config.model_validate(raw_config)


def save_results(results: np.ndarray, slug: str, config: Config):
    output_path = config.output_dir / f"{slug}.csv"
    num_layers = results.shape[0]

    df_wide = pd.DataFrame(results.T, columns=range(num_layers))
    df_long = df_wide.melt(var_name="layer", value_name="norm")
    df_long.to_csv(output_path, index=False)


def prepare_dataset(
    config: Config,
) -> list[list[str]]:
    ds = load_dataset(
        path=config.dataset.path,
        name=config.dataset.name,
        split=config.dataset.split,
        cache_dir=str(config.cache_dir),
    )

    prompts = [format_arc(entry) for entry in ds]

    batches = [
        prompts[i : i + config.batch_size]
        for i in range(0, len(prompts), config.batch_size)
    ]

    return batches


def format_arc(entry):
    question = entry["question"]
    choices = entry["choices"]

    answers = "\n".join(
        [f"{label}) {text}" for label, text in zip(choices["label"], choices["text"])]
    )

    return f"""{question}
{answers}"""


# Filter for saving activations from resid_pre and resid_post layers only
def resid_names_filter(name: str) -> bool:
    return name.endswith(("hook_resid_pre", "hook_resid_post"))


def extract_activation_transformation_norms(
    model_name: str, batches: list[list[str]], device=None, cache_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    model = HookedTransformer.from_pretrained(
        model_name, device=device, cache_dir=cache_dir
    )

    transformations = [[] for _ in range(model.cfg.n_layers)]
    normalized_transformations = [[] for _ in range(model.cfg.n_layers)]

    for batch in batches:
        _, cache = model.run_with_cache(
            batch, names_filter=resid_names_filter, return_type=None
        )

        # Stack pre and post activations
        # (n_layer, n_batch, n_pos, d_model)

        pre = cache.stack_activation("resid_pre")
        post = cache.stack_activation("resid_post")

        # Calculate transformation norms and normalized transformation norms
        # (n_layer, n_batch, n_pos)

        pre_norms = pre.norm(dim=-1)
        batch_transformations = (post - pre).norm(dim=-1)
        batch_normalized_transformations = batch_transformations / pre_norms

        # Save batch results to global ones
        # (n_layers, n_total_tokens)

        tokens = model.to_tokens(batch)
        attention_mask = tokens != model.tokenizer.pad_token_type_id

        batch_transformations_masked = (
            batch_transformations[:, attention_mask].cpu().tolist()
        )

        batch_normalized_transformations_masked = (
            batch_normalized_transformations[:, attention_mask].cpu().tolist()
        )

        # Figure how much tokens there is per layer.
        # First layer is present for sure, so it's picked arbitrarily.
        n_layer_tokens = len(batch_transformations_masked[0])

        for layer in range(model.cfg.n_layers):
            for norm in range(n_layer_tokens):
                transformations[layer].append(batch_transformations_masked[layer][norm])
                normalized_transformations[layer].append(
                    batch_normalized_transformations_masked[layer][norm]
                )

    return np.array(transformations), np.array(normalized_transformations)


def main(config: Config):
    torch.set_grad_enabled(False)
    device = utils.get_device()
    batches = prepare_dataset(config)

    for model in config.models:
        print(f"processing model {model}")

        transformations, normalized_transformations = (
            extract_activation_transformation_norms(
                model, batches, device, cache_dir=str(config.cache_dir)
            )
        )

        print(f"saving results {model}")

        save_results(transformations, slug(model), config)
        save_results(normalized_transformations, slug(model) + "normalized", config)


if __name__ == "__main__":
    config = load_config(parse_args())
    print(config)
    main(config)
