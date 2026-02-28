import argparse
import datetime
import re
from pprint import pprint
from typing import Any

import pandas as pd
import torch
import transformer_lens.utils as utils
import yaml
from datasets import load_dataset
from pydantic import BaseModel, DirectoryPath
from transformer_lens import HookedTransformer


THRESHOLD = 0.125


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


def process(
    model_name: str, batches: list[list[str]], device=None, cache_dir: str | None = None
) -> list[list[Any]]:
    model = HookedTransformer.from_pretrained(
        model_name, device=device, cache_dir=cache_dir
    )

    suspects = []

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

        # Find tokens
        # (n_batch, n_pos)

        tokens = model.to_tokens(batch)
        tokens_mask = tokens != model.tokenizer.pad_token_type_id

        for layer in range(model.cfg.n_layers):
            for b in range(len(batch)):
                prompt_mask = tokens_mask[b]
                prompt_tokens = model.to_str_tokens(tokens[b][prompt_mask])
                transformations = batch_normalized_transformations[layer][b][
                    prompt_mask
                ]

                # Look if any of the not padding tokens is a suspect.
                for i in range(len(prompt_tokens)):
                    if transformations[i] <= THRESHOLD:
                        suspects.append(
                            [prompt_tokens, layer, transformations.cpu().tolist()]
                        )
                        break

    return suspects


def main(config: Config):
    torch.set_grad_enabled(False)
    device = utils.get_device()
    batches = prepare_dataset(config)

    for model in config.models:
        results = process(model, batches, device, cache_dir=str(config.cache_dir))

        df = pd.DataFrame(results, columns=["prompt", "layer", "transformations"])
        df.to_json(config.output_dir / f"{model}.json")


if __name__ == "__main__":
    config = load_config(parse_args())
    print(config)
    main(config)
