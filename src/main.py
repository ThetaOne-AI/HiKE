import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
pier_path = os.path.join(os.path.dirname(__file__), "metrics/pier/jiwer")
sys.path.insert(0, pier_path)

import argparse
import json

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.metrics import mer, pier_fixed
from src.utils import add_space, normalize_text, replace_loanword, str2bool


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="whisper")
    args.add_argument("--model_name", type=str, default="openai/whisper-large")
    args.add_argument("--cache_dir", type=str, default=None)
    args.add_argument("--do_inference", type=str2bool, default=True)
    args.add_argument("--do_eval", type=str2bool, default=True)
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--replace_loanword", type=str2bool, default=True)
    args = args.parse_args()

    model = args.model
    model_name = args.model_name
    cache_dir = args.cache_dir
    do_inference = args.do_inference
    do_eval = args.do_eval
    batch_size = args.batch_size
    _replace_loanword = args.replace_loanword

    dir = f"outputs/{model_name}"

    if do_inference:
        print(" [INFO] Loading Model")
        if model == "whisper":
            from models.whisper import Whisper

            asr = Whisper(model_name)
        elif model == "gpt":
            from models.gpt import GPTTranscribe

            asr = GPTTranscribe(model_name=model_name)
        else:
            raise ValueError(
                f"Unsupported model: {model}. If you want to evaluate your model, please implement your model in `src/models/`"
            )

        print(" [INFO] Loading Dataset")
        dataset = load_dataset("thetaone-ai/HiKE", split="test", cache_dir=cache_dir)

        print(" [INFO] Starting Inference")
        result = asr.predict(dataset, batch_size=batch_size)
        result = result.remove_columns(["audio"])

        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/prediction.jsonl", "w", encoding="utf-8") as f:
            for i in range(len(result)):
                f.write(json.dumps(result[i], ensure_ascii=False) + "\n")

        print(f" [INFO] Inference Result Saved: {dir}/prediction.jsonl\n\n")

    if do_eval:
        print(" [INFO] Starting Evaluation")
        print(f" [INFO] Loading Inference Result: {dir}/prediction.jsonl")
        with open(f"{dir}/prediction.jsonl", "r", encoding="utf-8") as f:
            predictions = [json.loads(line) for line in f]

    for i, prediction in tqdm(enumerate(predictions), desc="Evaluating"):
        if _replace_loanword:
            pred = replace_loanword(prediction["pred_text"], prediction["loanwords"])
            ref_pier = replace_loanword(
                prediction["text_pier_labeled"], prediction["loanwords"]
            )
            ref_mer = replace_loanword(
                prediction["text_normalized"], prediction["loanwords"]
            )
        else:
            pred = prediction["pred_text"]
            ref_pier = prediction["text_pier_labeled"]
            ref_mer = prediction["text_normalized"]

        pred = normalize_text(pred)
        # add space between English and Korean (e.g., "bug는 -> bug 는")
        pier_pred = add_space(pred)

        pier_score = pier_fixed(ref_pier, pier_pred)
        mer_score, _ = mer(ref_mer, pred)

        prediction["result"] = {"pier": pier_score, "mer": mer_score}

    with open(f"{dir}/prediction.jsonl", "w", encoding="utf-8") as f:
        for i in range(len(predictions)):
            f.write(json.dumps(predictions[i], ensure_ascii=False) + "\n")

    df = pd.DataFrame(predictions)

    scores = {}
    cs_levels = ["word", "phrase", "sentence"]
    for level in cs_levels:
        level_df = df[df["cs_level"] == level]
        mean_pier_level = level_df["result"].apply(lambda x: x["pier"]).mean()
        mean_mer_level = level_df["result"].apply(lambda x: x["mer"]).mean()
        scores[level] = {"pier": mean_pier_level, "mer": mean_mer_level}

    scores["all"] = {
        "pier": np.mean([prediction["result"]["pier"] for prediction in predictions]),
        "mer": np.mean([prediction["result"]["mer"] for prediction in predictions]),
    }

    with open(f"{dir}/result.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)

    print(f" [INFO] Evaluation Completed!")
    print(f" [INFO] Evaluation Result Saved: {dir}/result.json\n")


if __name__ == "__main__":
    main()
