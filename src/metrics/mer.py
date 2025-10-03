import re
from typing import Tuple

import jiwer
from jiwer.transforms import Compose, RemoveMultipleSpaces, Strip


class SpaceKoreanChars:
    # Add space after all Korean characters to consider Korean characters as a single token
    def __call__(self, s: str) -> str:
        s = re.sub(r"([\uAC00-\uD7A3])", r" \1 ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


def mer(ref_text, hyp_text) -> Tuple[float, dict]:
    steps = [SpaceKoreanChars(), Strip()]
    transform = Compose(steps)

    ref_t = transform(ref_text)
    hyp_t = transform(hyp_text)

    result = jiwer.process_words(ref_t, hyp_t)

    S = result.substitutions
    I = result.insertions
    D = result.deletions
    N = len(ref_t.split())

    mer_percent = result.wer * 100

    if N == 0:
        return mer_percent, {"S": 0, "I": result.insertions, "D": 0, "N": 0}

    return mer_percent, {"S": S, "I": I, "D": D, "N": N}


if __name__ == "__main__":
    # Korean Characters: 30, English Words: 4, N=34
    # 안한->한화 (2S) / ac->r curacy (S,I) / computational->환비테이션을 (S,5I) / efficiency->이펙션시 (S,3I)
    # {'S': 5, 'I': 9, 'D': 0, 'N': 34}
    # 41.18
    ref = "실험 결과 제안한 algorithm 은 기존 방법 대비 accuracy 가 향상되었으며 computational efficiency 또한 크게 개선되었다"
    hyp = "실험 결과 제한화 algorithm 은 기존 방법 대비 r curacy 가 향상되었으며 환비테이션을 이펙션 시 또한 크게 개선되었다"

    mers, b = mer(ref, hyp)
    print(mers)
    print(b)
