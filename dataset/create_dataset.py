import logging

import dask
import hydra
from dask.distributed import Client, LocalCluster
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from preprocessing import SeqDataset

# NOTE: should set environment variable ulimit (how many open files possible) quite high: e.g. ulimit -n  4096
# set garbage collection threshold higher so that it does not run frequently

dask.config.set(
    {
        "dataframe.shuffle.method": "disk",
        "dataframe.shuffle.compression": "Snappy",
        "workers-share-disk": True,
        "distributed.p2p.storage.disk": False,
        "distributed.worker.memory.recent-to-old-timeout": "120s",
        "distributed.memory.spill": 0.7,
    }
)


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs_hydra", config_name="configs")
def create_dataset(cfg: DictConfig):
    # get configs
    config_data = cfg["dataset"]
    config_dask = cfg["dask"]
    with LocalCluster(
        n_workers=config_dask["n_workers"],
        processes=True,
        threads_per_worker=config_dask["threads_per_worker"],
        local_directory=config_dask["local_directory"],
        memory_limit=config_dask["memory_limit"],
        dashboard_address=config_dask["dashboard_address"],
        timeout=120,
        service_kwargs={
            "timemouts": {"connect": "360s", "tcp": "360s"},
        },
    ) as cluster:
        with Client(cluster) as client:
            # get file paths
            f_rev = config_data["path_reviews"]
            f_met = config_data["path_metadata"]
            f_out = config_data["path_output"]
            logger.info(f"Processing files: {f_rev} and {f_met}")
            logger.info(f"Output path: {f_out}")

            # cast to list because of hydra typings
            if isinstance(f_rev, ListConfig):
                f_rev = list(f_rev)
            if isinstance(f_met, ListConfig):
                f_met = list(f_met)

            seq = SeqDataset(chunksize=config_data["chunksize"])
            seq.load_interactions(
                f_rev,
                c_uid=config_data["c_uid"],
                c_iid=config_data["c_iid"],
                c_timestamp=config_data["c_timestamp"],
                c_score=config_data["c_score"],
            )

            if f_met is not None:
                seq.load_metadata(
                    f_met,
                    c_iid=config_data["c_iid"],
                    metadata_cols=config_data["metadata_cols"],
                    dropna=config_data["dropna"],
                )
            if config_data["kcore"] > 0:
                to_continue = seq.kcore_filtering(kcore=config_data["kcore"])
                if not to_continue:
                    logger.info("Dataset is empty after k-core filtering. Exiting...")
                    exit()

            seq.encode_entries()
            seq.create_sequences()
            seq.split_data()

            seq.save(
                f_out,
                save_metadata=True,
            )
            logger.info("Done")


if __name__ == "__main__":
    create_dataset()
    exit()


# exampe:
# python ./dataset/create_dataset.py dataset=amazon_2018/Gift_Cards
