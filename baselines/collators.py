from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Union

import constants as cts
import numpy as np
import pandas as pd
import torch
from dataloading import SequenceOutput
from recommender.RecFormer.recformer import RecformerTokenizer
from recommender.RecFormer.recformer.models import RecformerConfig

np.random.seed(42)


def add_candidates_batch(
    df: pd.DataFrame,
    option: str = "random",
    num_candidates: int = 10,
    evaluation: bool = False,
):
    """
        Get candidates for evaluation/training. We sample candidates from the all items in the batch. We ensure that the candidates do not appear in each sequence.
    Args:
        df (pd.DataFrame): batch to be passed in the dataloader in the form of a pandas dataframe. The dataframe should have a "sequence" column (List[int]) and a "label" column (int).
        option (str, optional): How to sample candidates. Options: "random", "most_popular", and "least_popular"
        num_candidates (int, optional): Number of candidates to select.
        evaluation (bool, optional): Whether the batch is for evaluation or training. If True, we do not exclude candidates. Defaults to False.
    Returns:
        pd.DataFrame: batch in the form of a dataframe with candidates added.
    """

    if option == "random":
        # randomly sample k items from the set of all items (present in sequences and labels). ensuring they do not appear in the sequence
        label_ids = set(list(df[cts.LABEL].unique()))
        sequence_ids = set(
            [item for sublist in df[cts.SEQUENCE].tolist() for item in sublist]
        )
        # combine the ids
        item_ids = label_ids.union(sequence_ids)
    elif option in ["least_popular", "most_popular"]:
        # sample_tollerance is used to ensure that we have enough items to sample from. We double the number of potential candidates in case some are already in the sequence
        sample_tollerance = 10

        # count the number of items in the labels
        label_ids = Counter(
            df[cts.LABEL].value_counts().sort_values(ascending=True).to_dict()
        )
        # count the number of items all sequences
        sequence_ids = Counter(
            pd.Series(df[cts.SEQUENCE].sum())
            .value_counts()
            .sort_values(ascending=True)
            .to_dict()
        )
        # combine the counts
        item_ids = label_ids + sequence_ids
        # normalize the counts
        total = sum(item_ids.values())
        item_ids = {k: v / total for k, v in item_ids.items()}

        # get the least/most popular items
        order = False if option == "least_popular" else True
        item_ids = set(
            [
                k
                for k, v in sorted(
                    item_ids.items(), key=lambda item: item[1], reverse=order
                )
            ][: num_candidates * sample_tollerance]
        )
    else:
        raise NotImplementedError(f"Option value {option} not supported")

    # make sure the candidates do not appear in the sequence
    if not evaluation:
        df[cts.CANDIDATES] = df[cts.SEQUENCE].apply(lambda x: list(item_ids - set(x)))

        df[cts.CANDIDATES] = [item_ids] * len(df)

        # make sure the label does not appear in the candidates
        df[cts.CANDIDATES] = df.apply(
            lambda row: list(set(row[cts.CANDIDATES]) - set([row[cts.LABEL]])), axis=1
        )
    else:
        df[cts.CANDIDATES] = [list(item_ids)] * len(df)

    # we need to ensure that we have enough items to for {num_candidates} candidates.
    try:
        df[cts.CANDIDATES] = df[cts.CANDIDATES].apply(
            lambda x: np.random.choice(x, size=num_candidates, replace=False, p=None)
        )
    except ValueError:
        raise ValueError(
            f"Could not find enough candidates to sample from. Please ensure that there are at least {num_candidates} unique items in the batch"
        )

    return df


class RecFormerCollateFn:
    """
    A collate function class for RecFormer models that handles tokenization and batching.

    Attributes:
        model_name (str): The name of the model's tokenizer to use (e.g. allenai/longformer-base-4096).
        max_attr_num (int): The maximum number of attributes.
        max_attr_length (int): The maximum length of each attribute.
        max_item_embeddings (int): The maximum number of item embeddings.
        max_token_num (int): The maximum number of tokens.
        option (str): The option for candidate selection
        num_candidates (int): The number of candidates to select.
        tokenizer (RecformerTokenizer): The tokenizer instance for the model (not required).
        evaluation (bool, optional): Whether the batch is for evaluation or training. If True, we do not exclude candidates. Defaults to False.

    Methods:
        __call__(batch: List[Dict[str, Union[List[int], Dict[str, List[str]], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            Processes a batch of data, tokenizes features, and adds candidates if specified.

        encode_features(batch_feature: List[List[int]]) -> List[List[int]]:
            Encodes a batch of features using the tokenizer.
    """

    def __init__(
        self,
        model_name: str,
        max_attr_num: int,
        max_attr_length: int,
        max_item_embeddings: int,
        max_token_num: int,
        option: str = "random",
        num_candidates: int = 10,
        tokenizer: RecformerTokenizer = None,
        evaluation: bool = False,
    ):
        self.model_name = model_name
        self.max_attr_num = max_attr_num
        self.max_attr_length = max_attr_length
        self.max_item_embeddings = max_item_embeddings
        self.max_token_num = max_token_num
        self.option = option
        self.num_candidates = num_candidates
        self.tokenizer = tokenizer
        self.evaluation = evaluation

        # Initialize tokenizer if not provided
        if self.tokenizer is None:
            config = RecformerConfig.from_pretrained(self.model_name)
            config.max_attr_num = self.max_attr_num
            config.max_attr_length = self.max_attr_length
            config.max_item_embeddings = self.max_item_embeddings
            config.max_token_num = self.max_token_num

            self.tokenizer = RecformerTokenizer.from_pretrained(self.model_name, config)

    def __call__(
        self,
        batch: List[SequenceOutput],
    ) -> Dict[str, torch.Tensor]:

        labels = [x.label for x in batch]
        batch_feature = [x.feature_list for x in batch]
        batch_sequence = [x.sequence for x in batch]

        batch_encode_features = self.encode_features(batch_feature)
        batch_out = self.tokenizer.padding(batch_encode_features, pad_to_max=False)
        batch_out[cts.LABEL] = labels
        for k, v in batch_out.items():
            batch_out[k] = torch.LongTensor(v)

        # get candidates
        if self.num_candidates > 0:
            df_batch = pd.DataFrame({cts.SEQUENCE: batch_sequence, cts.LABEL: labels})
            df_batch = add_candidates_batch(
                df_batch, self.option, self.num_candidates, self.evaluation
            )
            cans = np.array(df_batch[cts.CANDIDATES].tolist())
            batch_out[cts.CANDIDATES] = torch.LongTensor(cans)

        return batch_out

    def encode_features(self, batch_feature):
        features = []
        for feature in batch_feature:
            features.append(self.tokenizer.encode(feature, encode_item=True))

        return features


class IDOnlyCollateFn:
    """
    IdOnlyCollateFn is a callable class that processes a batch of data for a machine learning model.

    Attributes:
        option (str): The method to use for selecting candidates.
        num_candidates (int): The number of candidates to generate.
        evaluation (bool): Whether the batch is for evaluation or training.

    Methods:
        __call__(batch: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            Processes a batch of data and returns a dictionary containing the sequences: List[int], sequence lengths: int, labels: int,
            and optionally candidates: int.

            Args:
                batch (List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]): A batch of data where each
                item is a dictionary containing "sequence", "len_seq", and "label".

            Returns:
                Dict[str, torch.Tensor]: A dictionary containing the processed batch data with keys "sequence",
                "len_seq", "label", and optionally "candidates".
    """

    def __init__(
        self, option: str = "random", num_candidates: int = 10, evaluation: bool = False
    ):
        self.option = option
        self.num_candidates = num_candidates
        self.evaluation = evaluation

    def __call__(
        self,
        batch: List[SequenceOutput],
    ) -> Dict[str, torch.Tensor]:
        """
        features: A batch of list of item ids
        """
        batch_sequence = [item.sequence for item in batch]
        len_seq = [item.len_seq for item in batch]
        labels = [item.label for item in batch]

        batch_out = {
            cts.SEQUENCE: torch.LongTensor(batch_sequence),
            cts.LEN_SEQ: torch.LongTensor(len_seq),
            cts.LABEL: torch.LongTensor(labels),
        }

        # get candidates
        if self.num_candidates > 0:
            df_batch = pd.DataFrame({cts.SEQUENCE: batch_sequence, cts.LABEL: labels})
            df_batch = add_candidates_batch(
                df_batch, self.option, self.num_candidates, evaluation=self.evaluation
            )
            cans = np.array(df_batch[cts.CANDIDATES].tolist())
            batch_out[cts.CANDIDATES] = torch.LongTensor(cans)

        return batch_out
