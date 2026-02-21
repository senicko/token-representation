import argparse
import datetime
import os
import re
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import transformer_lens.utils as utils
import yaml
from datasets import load_dataset
from transformer_lens import HookedTransformer


def slug(base_name="my-benchmark"):
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    clean_name = re.sub(r"[^a-z0-9]+", "-", base_name.lower()).strip("-")
    slug = f"{timestamp_str}_{clean_name}"

    return slug


def load_config(args):
    config = {"output_dir": args.output_dir}

    with open(args.config, "r") as config_file:
        config.update(yaml.safe_load(config_file))

    return config


def save_results(norms: np.ndarray, slug: str, config: dict):
    output_dir = Path(config.get("output_dir", "."))
    output_path = output_dir / f"{slug}.csv"
    os.makedirs(output_dir, exist_ok=True)

    num_layers = norms.shape[0]
    df_wide = pd.DataFrame(norms.T, columns=range(num_layers))
    df_long = df_wide.melt(var_name="layer", value_name="norm")
    df_long.to_csv(output_path, index=None)


def prepare_dataset(config: dict) -> list[str]:
    ds = load_dataset(
        path=config.get("path"), name=config.get("name"), split=config.get("split")
    )

    prompts = [format_arc(entry) for entry in ds]

    return prompts


def format_arc(entry):
    question = entry["question"]
    choices = entry["choices"]

    return f"""{question}
{"\n".join([f"{label}) {text}" for label, text in zip(choices["label"], choices["text"])])}
"""


# Filter for saving activations from resid_pre and resid_post layers only
def resid_names_filter(name: str) -> bool:
    return name.endswith(("hook_resid_pre", "hook_resid_post"))


def extract_activation_transformation_norms(
    model_name: str, prompts: list[str], device=None
) -> np.ndarray:
    model = HookedTransformer.from_pretrained(model_name, device=device)

    _, cache = model.run_with_cache(
        prompts, names_filter=resid_names_filter, return_type=None
    )

    # Calculate the norms
    pre = cache.stack_activation("resid_pre")
    post = cache.stack_activation("resid_post")
    norms = (post - pre).norm(dim=-1)

    # Mask out padding tokens, which tend to get "stretched" noticably more than other tokens
    tokens = model.to_tokens(prompts)
    attention_mask = tokens != model.tokenizer.pad_token_type_id

    return norms[:, attention_mask].cpu().numpy()


def main(config: dict):
    torch.set_grad_enabled(False)
    device = utils.get_device()
    prompts = prepare_dataset(config["dataset"])
    print("loaded dataset")

    for model in config["models"]:
        print(f"running {model}")
        norms = extract_activation_transformation_norms(model, prompts, device)
        print(f"saving {model}")
        save_results(norms, slug(model), config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-od", "--output_dir")
    args = parser.parse_args()

    config = load_config(args)
    pprint(f"{config}")

    main(config)
