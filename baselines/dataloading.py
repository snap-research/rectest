import re
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Any, Callable, Dict, List, Optional

import constants as cts
import dask.dataframe as dd
from torch.utils.data import IterableDataset


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
