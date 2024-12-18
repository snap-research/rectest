import re
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Any, Callable, Dict, List, Optional

import constants as cts
import dask.dataframe as dd
from torch.utils.data import IterableDataset
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from transformers import AutoTokenizer
import logging

@dataclass
class SequenceOutput:
    """
    SequenceOutput is a data class that represents the output of a dataset.

    Attributes:
        sequence (List[int]): The sequence of item IDs.
        len_seq (int): The length of the sequence.
        label (int): The target label/ item id. This is the next item in the sequence.
        feature_list (Optional[List[Any]]): An optional list of features, currently used only in Recformer,
                                        and is of the form List[Dict[str, str]].
    """

    sequence: List[int]
    len_seq: int
    label: int
    feature_list: Optional[List[Any]] = None


class SequentialIterableDataset(IterableDataset):
    """
    Iterable dataset for sequential loading sequential data from parquet.
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        max_seq_len: int = 10,
        min_seq_len: int = 1,
    ):
        """
        Initialize the dataset.
        Args:
            data_path (str): Path to the dataset
            split (str): Split to use: Options: train, val, test
            max_seq_len (int, optional): Max sequence length to consider.
            min_seq_len (int, optional): Min sequence length to consider.
        """
        path_data = pjoin(data_path, split)
        self.data = dd.read_parquet(path_data)

        # filter sequences based on user input (< max_seq_len and > min_seq_len)
        if max_seq_len is not None:
            self.data = self.data[self.data[cts.LEN_SEQ] <= max_seq_len]
        if min_seq_len is not None:
            self.data = self.data[self.data[cts.LEN_SEQ] >= min_seq_len]

        # index is user_id reset it
        self.data = self.data.reset_index()

        # get number of items present in the dataset.
        self.max_seq_len = max_seq_len
        # set padding token
        self.padding_token = 0
        # get size of data (necessary if filtering is applied)
        self.size = len(self.data)

    def __iter__(self):
        for idx, data in self.data.iterrows():
            seq = data[cts.SEQUENCE].tolist()
            len_seq = data[cts.LEN_SEQ]
            gt = data[cts.LABEL]

            # add padding if needed
            if len(seq) < self.max_seq_len:
                padding_len = self.max_seq_len - len(seq)
                seq = [int(i) for i in seq]
                seq += [self.padding_token for _ in range(padding_len)]
            yield SequenceOutput(sequence=seq, len_seq=len_seq, label=gt)

    def __len__(self):
        return self.size


class RecformerDataset(IterableDataset):
    """
    Iterable dataset for sequential loading sequential data from parquet.
    This dataset is used for Recformer model.
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        max_seq_len: int = 100,
        min_seq_len: int = 1,
        feats_to_dict: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.
        Args:
            path (str): Path to the dataset
            split (str): Split to use: Options: train, val, test
            max_seq_len (int, optional): Max sequence length to consider.
            min_seq_len (int, optional): Min sequence length to consider.
            feats_to_dict (Callable, optional): Function to convert features to dictionary. If not provided, default function is used.
        """

        if feats_to_dict:
            self.feats_to_dict = feats_to_dict
        else:
            self.feats_to_dict = self.default_feats_to_dict

        path_data = pjoin(data_path, split)
        self.data = dd.read_parquet(path_data)

        # filter sequences based on user input (< max_seq_len and > min_seq_len)
        self.data = self.data[self.data[cts.LEN_SEQ] <= max_seq_len]
        self.data = self.data[self.data[cts.LEN_SEQ] >= min_seq_len]

        # convert features of each item to appropriate format i.e. [{title: "title", brand: "brand", category: "category"}, ...]
        self.data[cts.FEATURES_LIST] = self.data[cts.FEATURES_LIST].apply(
            lambda x: [self.feats_to_dict(y) for y in x],
            meta=(cts.FEATURES_LIST, "object"),
        )

        # count from 0 instead as 1 is used for padding in the longformer model: https://huggingface.co/allenai/longformer-base-4096/blob/main/config.json
        self.data[cts.LABEL] = self.data[cts.LABEL] - 1

        # index is user_id reset it
        self.data = self.data.reset_index()
        self.data = self.data.persist()

        # get size of data (necessary if filtering is applied)
        self.size = len(self.data)

    def default_feats_to_dict(self, input_text):
        """
        Extracts default features (title, brand, category) from the input text and returns them as a dictionary.

        Args:
            input_text (str): The input text containing the features in the format "title: <title> brand: <brand> category: <category>".

        Returns:
            dict: A dictionary with keys "title", "brand", and "category" containing the extracted feature values.
        """
        res = {"title": "", "brand": "", "category": ""}
        title_to_brand_pattern = r"title:\s*(.*?)\s*brand:"
        title_to_brand_match = re.search(title_to_brand_pattern, input_text)
        if title_to_brand_match:
            title_to_brand = title_to_brand_match.group(1)
            res["title"] = title_to_brand

        # Extract everything until "brand:"
        brand_pattern = r"brand:\s*(.*?)\s*category:"
        brand_match = re.search(brand_pattern, input_text, re.DOTALL)
        if brand_match:
            after_brand = brand_match.group(1).strip()
            res["brand"] = after_brand

        # Extract everything after "category:"
        category_pattern = r"category:\s*(.*)"
        category_match = re.search(category_pattern, input_text)
        if category_match:
            category = category_match.group(1).strip()
            res["category"] = category

        return res

    def __iter__(self):
        for idx, data in self.data.iterrows():
            seq = data[cts.SEQUENCE]
            len_seq = data[cts.LEN_SEQ]
            gt = data[cts.LABEL]
            features = data[cts.FEATURES_LIST]

            yield SequenceOutput(
                sequence=seq, len_seq=len_seq, label=gt, feature_list=features
            )

    def __len__(self):
        return self.size


def extract_custom_feats(input_text: str) -> Dict[str, str]:
    """
    Generates a dictionary containing the input text as a feature.

    Args:
        input_text (str): The input text to be included in the features dictionary.

    Returns:
        Dict[str, str]: A dictionary with a single key "features" and the input text as its value.
    """
    res = {"features": input_text}
    return res


class LLMDataset(IterableDataset):
    """
    Iterable dataset for sequential training data loading.
    Adapted to align with RecformerDataset structure.
    """

    def __init__(
        self,
        data_path: str,
        candidates_path: str,
        split: str,
        max_seq_len: int = 100,
        min_seq_len: int = 1,
        max_len: int = 1024,
        llm_negative_sample_size: int = 5,
        evaluation: bool = False,
    ):
        """
        Initialize the dataset.
        Args:
            data_path (str): Path to the dataset.
            candidates_path (str): Path to the candidates dataset.
            split (str): Dataset split (e.g., 'train', 'test').
            max_seq_len (int, optional): Maximum sequence length.
            min_seq_len (int, optional): Minimum sequence length.
            max_len (int, optional): Maximum length for sentence.
            llm_negative_sample_size (int, optional): Size of negative samples for LLM.
            evaluation (bool, optional): Flag indicating if the dataset is for evaluation.
        """

        path_data = pjoin(data_path, split)
        self.data = dd.read_parquet(path_data)

        # filter sequences based on user input (< max_seq_len and > min_seq_len)
        self.data = self.data[self.data[cts.LEN_SEQ] <= max_seq_len]
        self.data = self.data[self.data[cts.LEN_SEQ] >= min_seq_len]

        # add item_id_encoded (target) in sequence
        self.data["item_id_encoded_list"] = self.data.apply(
            lambda row: list(row["item_id_encoded_list"]).append(
                row["item_id_encoded"]
            ),
            axis=1,
        )

        self.candidates = dd.read_parquet(candidates_path, split)
        self.max_len = max_len
        self.text_dict = (
            self.data.drop_duplicates(subset="item_id_encoded")[
                ["item_id_encoded", "features"]
            ]
            .set_index("item_id_encoded")
            .to_dict()["features"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        try:
            from recommender.LlamaRec.datasets.utils import Prompter
        except ImportError:
            logging.error("The LlamaRec submodule is not available. Please initialize the submodule.")
            raise ImportError("The LlamaRec submodule is not available. Please initialize the submodule.")
        self.prompter = Prompter()
        self.llm_negative_sample_size = llm_negative_sample_size
        self.evaluation = evaluation
        self.size = len(self.data)

    def __iter__(self):
        for idx, tokens in self.data["item_id_encoded_list"].values:
            yield self._process_item(idx, tokens)

    def _process_item(self, idx, tokens):
        """
        Process a single sequence item to generate input features and labels.

        Args:
            tokens: Sequence of tokens.

        Returns:
            Processed input data.
        """
        answer = tokens[-1]
        original_seq = tokens[:-1]
        seq = original_seq[-self.max_len :]

        candidates = self.candidates.loc[idx]

        return self.seq_to_token_ids(
            self.max_len,
            DEFAULT_SYSTEM_PROMPT,
            cts.INPUT_TEMPLATE,
            seq,
            candidates,
            answer,
            self.text_dict,
            self.tokenizer,
            self.prompter,
            eval=self.evaluation,
        )

    # the following prompting is based on alpaca
    def generate_and_tokenize_eval(
        self, llm_max_text_len, data_point, tokenizer, prompter
    ):
        in_prompt = prompter.generate_prompt(data_point["system"], data_point["input"])
        tokenized_full_prompt = tokenizer(
            in_prompt,
            truncation=True,
            max_length=llm_max_text_len,
            padding=False,
            return_tensors=None,
        )
        tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord("A")

        return tokenized_full_prompt

    def generate_and_tokenize_train(
        self, llm_max_text_len, llm_train_on_inputs, data_point, tokenizer, prompter
    ):
        def tokenize(prompt, add_eos_token=True):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=llm_max_text_len,
                padding=False,
                return_tensors=None,
            )
            if result["input_ids"][-1] != self.tokenizer.eos_token_id and add_eos_token:
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()
            return result

        full_prompt = prompter.generate_prompt(
            data_point["system"], data_point["input"], data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
        if not llm_train_on_inputs:
            tokenized_full_prompt["labels"][:-2] = [-100] * len(
                tokenized_full_prompt["labels"][:-2]
            )

        return tokenized_full_prompt

    def seq_to_token_ids(
        self,
        llm_max_title_len,
        llm_system_template,
        llm_input_template,
        seq,
        candidates,
        label,
        text_dict,
        tokenizer,
        prompter,
        eval=False,
    ):
        def truncate_title(title):
            title_ = self.tokenizer.tokenize(title)[:llm_max_title_len]
            title = self.tokenizer.convert_tokens_to_string(title_)
            return title

        seq_t = " \n ".join(
            [
                "(" + str(idx + 1) + ") " + truncate_title(text_dict[item])
                for idx, item in enumerate(seq)
            ]
        )
        can_t = " \n ".join(
            [
                "(" + chr(ord("A") + idx) + ") " + truncate_title(text_dict[item])
                for idx, item in enumerate(candidates)
            ]
        )
        output = chr(ord("A") + candidates.index(label))  # ranking only

        data_point = {}
        data_point["system"] = (
            llm_system_template
            if llm_system_template is not None
            else DEFAULT_SYSTEM_PROMPT
        )
        data_point["input"] = llm_input_template.format(seq_t, can_t)
        data_point["output"] = output

        if eval:
            return self.generate_and_tokenize_eval(
                llm_max_title_len, data_point, tokenizer, prompter
            )
        else:
            return self.generate_and_tokenize_train(
                llm_max_title_len, eval, data_point, tokenizer, prompter
            )

    def __len__(self):
        return self.size
