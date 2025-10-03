import argparse
import json

import regex
from jiwer.transforms import (Compose, ExpandCommonEnglishContractions,
                              ReduceToListOfListOfWords,
                              ReduceToSingleSentence, RemoveKaldiNonWords,
                              RemoveMultipleSpaces, RemovePunctuation,
                              RemoveWhiteSpace, Strip, SubstituteWords,
                              ToLowerCase)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v_str = str(v).lower()
    if v_str in ("yes", "true", "t", "y", "1"):  # truthy
        return True
    if v_str in ("no", "false", "f", "n", "0"):  # falsy
        return False
    raise argparse.ArgumentTypeError("Boolean value expected. Use true/false.")


def replace_loanword(text, loanwords):
    if isinstance(text, dict):
        if "text" in text:
            text = text["text"]
    try:
        loanwords = json.loads(loanwords)
        for loanword in loanwords:
            text = text.replace(loanword["Korean"], loanword["English"])
    except:
        import pdb

        pdb.set_trace()
    return text


def add_space(text):
    return regex.sub(r"([A-Za-z0-9]+)(?=\p{Script=Hangul})", r"\1 ", text)


def normalize_text(text):
    if isinstance(text, dict):
        text = text["text"]

    wer_standardize_contiguous = Compose(
        [
            ToLowerCase(),
            ExpandCommonEnglishContractions(),
            RemoveKaldiNonWords(),
            SubstituteWords({"—": " "}),
            RemovePunctuation(),
            RemoveWhiteSpace(replace_by_space=True),
            RemoveMultipleSpaces(),
            Strip(),
            ReduceToSingleSentence(),
            ReduceToListOfListOfWords(),
        ]
    )
    return " ".join(wer_standardize_contiguous(text)[0])


if __name__ == "__main__":
    text = "Hello, world! I'm a student."
    print(normalize_text(text))
