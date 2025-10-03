from typing import Any, Dict, List

from datasets import Dataset


class BaseASR:
    """
    Common interface for all ASR models
    - generate(input: List[Any], **kwargs) -> List[Dict[str, Any]]
    - map_to_pred(batch) -> {"pred_text": List[str]}
    - predict(dataset: Dataset, batch_size: int = 32) -> Dataset
    """

    def generate(self, input, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def map_to_pred(self, batch):
        audios = batch.get("audio")
        if (
            isinstance(audios, list)
            and len(audios) > 0
            and isinstance(audios[0], dict)
            and "array" in audios[0]
        ):
            inputs = [a["array"] for a in audios]
        else:
            inputs = audios

        outputs = self.generate(inputs)
        if isinstance(outputs, dict):
            outputs = [outputs]
        texts = [o.get("text") if isinstance(o, dict) else o for o in outputs]
        return {"pred_text": texts}

    def predict(self, dataset: Dataset, batch_size: int = 1):
        return dataset.map(
            self.map_to_pred,
            batched=True,
            batch_size=batch_size,
            desc="Generating predictions",
        )
