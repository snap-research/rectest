import json
import logging
import os
import re
from collections import defaultdict
from os.path import join as pjoin
from typing import Optional

import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix

# set up logging
# format logger into time in,filename:line number, level and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def custom_loads(x: str, cols: list[str]):
    """
    Custom function to load json lines. We only select the columns we are interested in.
    Args:
        x (str): Line of file read.
        cols (list[str]): Columns to consider

    Returns:
        dict: Dictionary with the selected columns.
    """
    try:
        line = json.loads(x)
        line = {k: str(v) for k, v in line.items() if k in cols}
        return line
    except:
        return None


class SeqDataset:
    def __init__(self, chunksize: int = 100):
        """
        Initialize the dataset
        """
        self.interactions: Optional[defaultdict[list]] = None
        self.metadata: Optional[defaultdict[list]] = None
        self.metadata_cols: Optional[list[str]] = None
        self.c_uid: Optional[str] = None
        self.c_iid: Optional[str] = None
        self.c_timestamp: Optional[str] = None
        self.train_data: Optional[dd.DataFrame] = None
        self.val_data: Optional[dd.DataFrame] = None
        self.test_data: Optional[dd.DataFrame] = None
        # flag on whether k-core filtering has been applied
        self.kcore: Optional[bool] = False
        # flag to track if index is user_id
        self.index_is_user_id: Optional[bool] = False
        # encoders for users and items
        self.encoder_users: Optional[dict] = None
        self.encoder_items: Optional[dict] = None
        # keep track of users and items so we don't recaculate them (saved after k-core filtering)
        self.users_present: Optional[list] = None
        self.items_present: Optional[list] = None
        # size of dask partitions to use (in MB)
        self.CHUNKSIZE = chunksize

    def load_interactions(
        self,
        path: str,
        c_uid: str,
        c_iid: str,
        c_timestamp: int,
        c_score: str,
    ):
        """
        Load interactions from a file. The interactions will be stored as a dictionary with the user ID as key.

        Args:
            path (str): interactions file(s) path. If more than on file is provided the columns must be the same.
            c_uid (str): column indicating user ID.
            c_uid (str): column indicating item ID.
            c_timestamp (int): column indicating timestamp of interaction. Must be in unix format.
            c_score (str): column indicating score/rating/aproval of item by user.
        """
        # get interactions in bag first
        # TODO: currently we support only jsonl format but it should be easy to extend the custom_loads function to other formats
        # NOTE: loading using daks bags is slow but we avoid schema issues. If schema is consistent then we can use dd.read_json/read_parquet
        b = db.read_text(path).map(
            custom_loads, cols=[c_uid, c_iid, c_timestamp, c_score]
        )
        # get size of file to determine number of partitions
        f_size = (
            int(os.path.getsize(path) / (1024 * 1024))
            if isinstance(path, str)
            else sum([int(os.path.getsize(p) / (1024 * 1024)) for p in path])
        )
        # repartition for fatster processing
        n_partitions = f_size // self.CHUNKSIZE if f_size > self.CHUNKSIZE else 1
        logger.info(
            f"Total size of interactions: {f_size:,} MB. Repartitioning interactions to {n_partitions} partitions"
        )
        # b = b.repartition(n_partitions)
        ddf = b.to_dataframe()

        # standarize column names
        ddf = ddf.rename(
            columns={
                c_uid: "user_id",
                c_timestamp: "timestamp",
                c_score: "score",
                c_iid: "item_id",
            }
        )
        # persist the new names and index (after dopping na)
        ddf = ddf.persist()

        # drop duplicates and entries where we do not have user_id
        # NOTE: we follow the original Amazon Reviews dataset and drop all duplicate items ignoring timestamps e.g. we could do (same user/id/time) instead: ddf = ddf.drop_duplicates(subset=["user_id", "item_id", "timestamp"])
        logger.info("Dropping duplicates in interactions")
        ddf = ddf.drop_duplicates(subset=["user_id", "item_id"], shuffle_method="disk")
        ddf = ddf.dropna(subset=["user_id"])

        # reset index and repartion
        ddf = ddf.reset_index(drop=True)
        ddf = ddf.repartition(npartitions=n_partitions)

        ddf = ddf.persist()
        self.interactions = ddf
        logger.info("Interactions loaded")

    def load_metadata(
        self,
        path: str,
        c_iid: str,
        metadata_cols: list[str] = None,
        dropna: bool = False,
    ):
        """
        Load item metadata from a file. The metadata will be stored as a dictionary with the item ID as key.

        Args:
            path (str): path to the metadata file(s). If more than on file is provided the columns must be the same.
            c_iid (str, optional): column representing item ID.
            metadata_cols (list[str], optional): columns to consider as additional features. All features will be concatenated as strings.
            dropna (bool, optional): drop rows with missing features.
        """

        # get metadata in bag first
        # TODO: currently we support only jsonl format but it should be easy to extend the custom_loads function to other formats
        # NOTE: loading using daks bags is slow but we avoid schema issues. If schema is consistent then we can use dd.read_json/read_parquet
        b = db.read_text(path).map(custom_loads, cols=[c_iid, *metadata_cols])
        # get size of file to determine number of partitions
        f_size = (
            int(os.path.getsize(path) / (1024 * 1024))
            if isinstance(path, str)
            else sum([int(os.path.getsize(p) / (1024 * 1024)) for p in path])
        )
        n_partitions = f_size // self.CHUNKSIZE if f_size > self.CHUNKSIZE else 1
        logger.info(
            f"Total size of metadata: {f_size:,}MB. Repartitioning metadata to {n_partitions} partitions"
        )
        metadata = b.to_dataframe()

        # standarize column name (should use same name on interactions df)
        metadata = metadata.rename(columns={c_iid: "item_id"})
        # drop duplicates
        logger.info("Dropping duplicates in metadata")
        metadata = metadata.drop_duplicates(subset=["item_id"], shuffle_method="disk")

        # reset index and repartion
        metadata = metadata.reset_index()
        metadata = metadata.repartition(npartitions=n_partitions)
        metadata = metadata.persist()

        # Columns to concatenate
        if metadata_cols is None:
            # if not provided use all columns
            metadata_cols = metadata.columns

        # Create the 'features' column
        # we concatenate all the metadata columns into a single string in the form: "col1: value1 col2: value2 ..."
        metadata["features"] = metadata[metadata_cols].apply(
            lambda row: " ~ ".join(f"{col}: {str(row[col])}" for col in metadata_cols),
            axis=1,
            meta=("x", "str"),
        )

        # double check if index is user_id. If so reset it to not lose it after merging
        if self.index_is_user_id:
            logger.info(
                "Interactions index is user_id. Resetting it to not lose it after merge"
            )
            self.interactions = self.interactions.reset_index()
            self.index_is_user_id = False

        # get interactions
        ddf = self.interactions

        # cast item_id to string for consistency in merges
        metadata["item_id"] = metadata["item_id"].astype("string")
        ddf["item_id"] = ddf["item_id"].astype("string")
        # merge metadata
        ddf = ddf.merge(
            on="item_id", right=metadata[["item_id", "features"]], how="left"
        )

        # dropping interactions where items do not exist in metadata.
        # we follow original amazon reviews 2023 paper instead of dropping them
        if dropna:
            # drop.na won't work here as we add prefix (e.g. title: )
            ddf["features_len"] = ddf["features"].str.len()
            min_len = len(" ".join(f"{col}: " for col in metadata_cols))
            ddf = ddf[ddf["features_len"] > min_len]
            ddf = ddf.drop("features_len", axis=1)
        else:
            ddf["features"] = ddf["features"].fillna("")

        ddf = ddf.reset_index(drop=True)
        self.interactions = ddf
        self.metadata = metadata
        del ddf, metadata
        logger.info("Metadata loaded and mapped.")

    def encode_entries(self):
        """
        Encode user and item entries to integers. This is necessary for ids based models.
        """
        # ensure user_id is not in index
        if self.index_is_user_id:
            self.interactions = self.interactions.reset_index()
            self.index_is_user_id = False
            logger.info("Interactions index is user_id. Resetting it")

        # get interactions
        ddf = self.interactions

        # get unique users and map them to integers
        # alternative way: https://ml.dask.org/modules/generated/dask_ml.preprocessing.LabelEncoder.html but appears to be slower.
        user_ids = (
            # self.users_present exist if kcore_filtering called first
            self.users_present
            if self.users_present is not None
            else ddf["user_id"].unique().compute()
        )
        # start index from 1 as we use 0 for padding (mostly relevant for item encoding but we keep it consistent)
        encoder_u = {user: idx + 1 for idx, user in enumerate(user_ids)}
        logger.info(f"Number of users: {len(encoder_u)}")

        ddf["user_id_encoded"] = ddf["user_id"].apply(
            lambda x: encoder_u[x], meta=("x", "int")
        )

        # keep track of mapping
        self.encoder_users = encoder_u
        logger.info("Users encoded")

        # similar for items
        # encode items
        item_ids = (
            self.items_present
            if self.items_present is not None
            else ddf["item_id"].unique().compute()
        )
        # start index from 1 as we use 0 for padding
        encoder_i = {item: idx + 1 for idx, item in enumerate(item_ids)}
        logger.info(f"Number of items: {len(item_ids)}")

        ddf["item_id_encoded"] = ddf["item_id"].apply(
            lambda x: encoder_i[x], meta=("x", "int")
        )

        # keep track of mappign
        self.encoder_items = encoder_i
        logger.info("Items encoded")

        self.interactions = ddf
        del ddf

    def kcore_filtering(self, kcore: int = 5):
        """
        Apply k-core filtering to the interactions. This will remove users and items with less than kcore appearances.

        Args:
            kcore (int, optional): Threshold (k) to apply.

        Returns:
            bool: True if k-core filtering was applied successfully. False otherwise.
        """

        def get_user_item_pairs(df):
            return pd.DataFrame(df[["user_id", "item_id"]].value_counts()).reset_index()

        logger.info("Applying k-core filtering.")
        # get interactions
        ddf = self.interactions

        # get count of user/per item.  with map_partitions with both columns
        user_item_counter = ddf.map_partitions(
            get_user_item_pairs,
            meta=pd.DataFrame(columns=["user_id", "item_id", "count"]),
        ).compute()
        item_counter = user_item_counter.groupby("item_id")["count"].sum()
        user_counter = user_item_counter.groupby("user_id")["count"].sum()

        logger.info("Counted users, items and user-item interactions.")

        # easy filter out users and items with less than kcore appearances
        # user counter
        user_counter = user_counter[user_counter >= kcore]
        users_to_consider = user_counter.index

        # item counter
        item_counter = item_counter[item_counter >= kcore]
        items_to_consider = item_counter.index

        users_to_consider = set(users_to_consider)
        items_to_consider = set(items_to_consider)

        ddf = ddf[
            (ddf["user_id"].isin(users_to_consider))
            & (ddf["item_id"].isin(items_to_consider))
        ]

        logger.info(
            "Filtered out users and items with less than 5 appearances. Creating sparse matrix..."
        )

        # we create a sparse matrix where rows = users, columns = items to perform iterative k-core filtering
        frame = user_item_counter
        rcLabel, vLabel = ("user_id", "item_id"), "count"
        rcCat = [
            CategoricalDtype(sorted(frame[col].unique()), ordered=True)
            for col in rcLabel
        ]
        rc = [
            frame[column].astype(aType).cat.codes
            for column, aType in zip(rcLabel, rcCat)
        ]
        mat = csr_matrix(
            (frame[vLabel], rc),
            shape=tuple(cat.categories.size for cat in rcCat),
            dtype=np.int64,
        )

        # filter matrix
        _, users_new, items_new = self.kcore_filtering_matrix(
            mat, kcore, rcCat[0].categories, rcCat[1].categories
        )

        logger.info("Filtered matrix computed. Applying filtering to interactions...")

        if len(users_new) == 0:
            logger.info("No users left after k-core filtering. Exiting")
            return False

        # keep only wanted users and items
        ddf = ddf[ddf["user_id"].isin(users_new)]
        ddf = ddf[ddf["item_id"].isin(items_new)]

        # keep track of users and items so we don't recaculate them (e.g. when encoding)
        self.users_present = users_new
        self.items_present = items_new

        self.interactions = ddf
        # persist after filtering
        self.interactions = self.interactions.persist()
        del ddf

        self.kcore = True
        logger.info("K-core filtering applied")

        return True

    def create_sequences(self):
        """
        Create ordered sequences of user interactions. The sequences will be clipped up to the target item.
        """
        # get interactions
        ddf = self.interactions

        # set index to user_id. This is expensive but we do a lot of groupby operations
        logger.info("Setting index to user_id")
        ddf = ddf.reset_index(drop=True)
        ddf = ddf.set_index(
            "user_id", compute=True, partition_size=f"{self.CHUNKSIZE}MB"
        )
        self.index_is_user_id = True
        logger.info("Index set to user_id")

        # TODO; this breaks if there is no score column
        cols_to_merge = [
            "item_id",
            "timestamp",
            "score",
        ]
        if self.metadata is not None:
            cols_to_merge.extend(["features"])
        if self.encoder_items is not None:
            cols_to_merge.extend(["item_id_encoded"])

        # merge target columns for each user
        for col in cols_to_merge:
            result = (
                ddf.groupby("user_id")[col]
                .apply(list, meta=(f"{col}_list", "object"))
                .to_frame()
            )

            ddf = ddf.merge(result, how="left", left_index=True, right_index=True)
            ddf = ddf.persist()

        logger.info("Created sequences")

        # clip sequences up to target/bought item
        cols_to_clip = [f"{col}_list" for col in cols_to_merge]
        for col in cols_to_clip:
            ddf[f"{col}_clipped"] = ddf.apply(
                lambda row: self.clip_sequence(row, col),
                meta=(f"{col}_list", "object"),
                axis=1,
            )
            ddf = ddf.persist()

        logger.info("Clipped sequences")

        # drop unwanted columns
        cols_to_keep = [
            "item_id",
            "timestamp",
            "score",
            "user_id_encoded",
            "item_id_encoded",
            "features",
            "item_id_list_clipped",
            "score_list_clipped",
            "timestamp_list_clipped",
            "features_list_clipped",
            "item_id_encoded_list_clipped",
        ]
        cols_to_drop = [col for col in ddf.columns if col not in cols_to_keep]
        ddf = ddf.drop(cols_to_drop, axis=1)

        # rename columns
        ddf = ddf.rename(
            columns={
                "item_id_list_clipped": "item_id_list",
                "score_list_clipped": "score_list",
                "timestamp_list_clipped": "timestamp_list",
                "features_list_clipped": "features_list",
                "item_id_encoded_list_clipped": "item_id_encoded_list",
            }
        )

        # keep track of sequence length
        ddf["len_seq"] = ddf["item_id_list"].apply(len, meta=("x", "int"))

        self.interactions = ddf
        del ddf
        logger.info("Renamed columns")

    def split_data(self):
        """
        Split the interactions into train, validation and test sets. We use leave-one-out method. (train: [:-2], val: [-2], test: [-1])
        """
        # get interactions
        ddf = self.interactions

        # split data set with lom (leave one out method)
        self.train_data = ddf.groupby("user_id", group_keys=False).apply(
            lambda group: group.iloc[:-2] if len(group) > 2 else None, meta=ddf
        )

        self.test_data = ddf.groupby("user_id", group_keys=False).apply(
            lambda group: group.iloc[-1], meta=ddf
        )

        self.val_data = ddf.groupby("user_id", group_keys=False).apply(
            lambda group: group.iloc[-2] if len(group) > 1 else None, meta=ddf
        )
        logger.info("Data splitted")
        del ddf

    def save(self, path: str, save_metadata: bool = False):
        """
        Save data to disk. We save interactions, train/val/test splits, encoders, stats and metadata.

        Args:
            path (str): Path to save the data.
            save_metadata (bool, optional): Flag to save metadata or not.
        """
        os.makedirs(path, exist_ok=True)

        if self.index_is_user_id:
            # reset index
            self.interactions = self.interactions.reset_index()
            self.index_is_user_id = False
            logger.info("Interactions index is user_id. Resetting it")

        # save interactions
        if self.interactions is not None:
            # get schema to use. This is slow but necessary to avoid schema issues.
            schema = pa.Schema.from_pandas(self.interactions.head())
            self.interactions.to_parquet(
                pjoin(path, "interactions"),
                overwrite=True,
                ignore_divisions=True,
                schema=schema,
            )

            # save splits
            if self.train_data is not None:
                # use same schema as self.interactions
                self.train_data.to_parquet(
                    pjoin(path, "train"),
                    schema=schema,
                    ignore_divisions=True,
                    overwrite=True,
                )
                self.val_data.to_parquet(
                    pjoin(path, "val"),
                    schema=schema,
                    ignore_divisions=True,
                    overwrite=True,
                )
                self.test_data.to_parquet(
                    pjoin(path, "test"),
                    schema=schema,
                    ignore_divisions=True,
                    overwrite=True,
                )

        # save metadata
        if self.metadata is not None and save_metadata:
            schema = pa.Schema.from_pandas(self.metadata.head())
            self.metadata.to_parquet(
                pjoin(path, "metadata"),
                schema=schema,
                overwrite=True,
                ignore_divisions=True,
            )

        if self.encoder_items is not None:
            with open(pjoin(path, "encoder_items.json"), "w") as f:
                json.dump(self.encoder_items, f)
            with open(pjoin(path, "encoder_users.json"), "w") as f:
                json.dump(self.encoder_users, f)

            # We save number of unique users, number of unique items and total number of interactions so we can use them later
            stats = {
                "num_users": len(self.encoder_users),
                "num_items": len(self.encoder_items),
                "num_interactions": len(self.interactions),
            }
            with open(pjoin(path, "stats.json"), "w") as f:
                json.dump(stats, f)

        logger.info("Data saved")

    def kcore_filtering_matrix(
        self, matrix: csr_matrix, k: int, users: np.array, items: np.array
    ):
        """
        Perform k-core filtering on the given sparse matrix.

        Parameters:
        matrix (csr_matrix): The user-item interaction sparse matrix.
        k (int): The minimum number of interactions to retain a user or an item.
        users (np.array): The original users.
        items (np.array): The original items.

        Returns:
        filtered_matrix (csr_matrix): The filtered user-item interaction sparse matrix.
        users (np.array): The filtered users.
        items (np.array): The filtered items.
        """

        # print every 5 iterations
        counter = 0
        while True:
            # Filter users with at least k interactions
            user_interactions = matrix.sum(
                axis=1
            ).A1  # sum along columns and convert to 1D array
            users_to_keep = np.where(user_interactions >= k)[0]

            # Filter items with at least k interactions
            item_interactions = matrix.sum(
                axis=0
            ).A1  # sum along rows and convert to 1D array
            items_to_keep = np.where(item_interactions >= k)[0]

            # Create the filtered matrix
            filtered_matrix = matrix[users_to_keep][:, items_to_keep]

            users = users[users_to_keep]
            items = items[items_to_keep]
            # If the matrix doesn't change, break the loop
            if filtered_matrix.shape == matrix.shape:
                break

            matrix = filtered_matrix

            counter += 1
            if counter % 5 == 0:
                logger.info(
                    f"Iteration {counter} - Users: {len(users)} Items: {len(items)}"
                )
        return filtered_matrix, users, items

    def clip_sequence(self, row: pd.Series, col: str):
        """
        Given a ddf row and a target column, clip the target column up to the target item_id after sorting based on timestamp
        Args:
            row (pd.Series): Row of df to apply the function.
            col (str): Target column name to clip.

        Returns:
          list[str]: Ordered list of target col clipped up to the row's target/item_id.
        """

        # find target item and timestamp
        item = row["item_id"]
        timestamp = row["timestamp"]

        # sort items and timestamps
        timestamps = list(row["timestamp_list"])
        sorted_items = [
            item for _, item in sorted(zip(timestamps, list(row["item_id_list"])))
        ]

        # get index of timestamp
        idx_timestamp = timestamps.index(timestamp)

        # get index of item
        if sorted_items.count(item) > 1:
            idx_item = sorted_items.index(item, idx_timestamp)
        else:
            idx_item = sorted_items.index(item)

        # clip target
        sorted_target = list([t for _, t in sorted(zip(timestamps, list(row[col])))])
        return list(sorted_target[:idx_item])
