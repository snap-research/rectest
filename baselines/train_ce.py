import os
import sys

sys.path.append(".")
import json
import logging
from os.path import join as pjoin
from pathlib import Path

import constants as cts
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from hydra.utils import instantiate
from metrics import Ranker
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed
from utils import AllGather

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def evaluate(model, test_loader, args, accelerator):
    # Evaluate
    model.eval()
    ranker = Ranker(args.metric_ks)
    with accelerator.autocast():
        with torch.no_grad():
            for batch in tqdm(
                test_loader,
                desc="Evaluate",
                disable=(not accelerator.is_local_main_process),
            ):

                for k, v in batch.items():
                    batch[k] = v.to(args.device)
                labels = batch[cts.LABEL]
                del batch[cts.LABEL]

                if args.collator_val.num_candidates > 0:
                    candidates = batch[cts.CANDIDATES]
                    del batch[cts.CANDIDATES]

                    scores = model(batch)

                    (
                        all_scores,
                        all_labels,
                        all_candidates,
                    ) = accelerator.gather_for_metrics((scores, labels, candidates))
                    all_labels = torch.argmax(
                        (all_candidates == all_labels[:, None]).to(torch.float32), dim=1
                    )
                else:
                    scores = model(batch)
                    all_scores, all_labels = accelerator.gather_for_metrics(
                        (scores, labels)
                    )
                    all_labels = all_labels.squeeze()

                # compute metrics for batch
                ranker(all_scores, all_labels)

            average_metrics = ranker.compute()
            return average_metrics


@hydra.main(version_base=None, config_path="../configs_hydra", config_name="configs")
def main(cfg: DictConfig):
    args = cfg["experiment"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(i) for i in range(args.number_of_devices)]
    )

    # ensure reproducibility
    set_seed(args.seed)

    # create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # read dataset stats to get number of items present
    with open(pjoin(args.dataset.data_path, "stats.json"), "r") as f:
        data_stats = json.load(f)
    args.model.item_num = data_stats["item_num"]

    with open(pjoin(args.output_dir, "args.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg["experiment"], resolve=True), f, indent=4)

    # setup accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else False,
        split_batches=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_dir=pjoin(args.output_dir, "./accelerate_output"),
        cpu=True if args.device == "cpu" else False,
    )

    # get data loaders
    train_collator = instantiate(args.collator_train)
    train_data = instantiate(args.dataset, split="train")
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size_train,
        collate_fn=train_collator,
    )

    val_collator = instantiate(args.collator_val)
    val_data = instantiate(args.dataset, split="val")
    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        collate_fn=val_collator,
    )

    test_collator = instantiate(args.collator_val)
    test_data = instantiate(args.dataset, split="test")
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size_val,
        collate_fn=test_collator,
    )

    # create model/optimizer/scheduler
    model = instantiate(args.model)
    optimizer = instantiate(args.optimizer, params=model.parameters())
    scheduler = instantiate(args.scheduler, optimizer=optimizer)

    # Send everything through `accelerator.prepare`
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model,
        optimizer,
        scheduler,
    )

    # Restore the previous state if provided
    if args.accelerate_checkpoint:
        logger.info("Restoring accelerator state")
        accelerator.load_state(args.accelerate_checkpoint)

    if args.eval_at_start:
        metrics = evaluate(model, test_dataloader, args, accelerator)
        metrics["trained"] = "no"

        if accelerator.is_local_main_process:
            logger.info("Evaluation on test set before training")
            logger.info(f"Metrics: {metrics}")

            with open(pjoin(args.output_dir, "metrics_test.json"), "a") as f:
                json.dump(metrics, f)
                f.write("\n")

    criterion = nn.CrossEntropyLoss()
    early_stopping = args.early_stopping
    stop_training = False
    step_count = 0
    best_metric = 0
    for epoch in range(args.epochs):
        model.train()
        with accelerator.accumulate(model):
            for step, batch in tqdm(
                enumerate(train_dataloader),
                desc=f"Epoch {epoch}",
                total=len(train_dataloader),
                disable=(not accelerator.is_local_main_process),
            ):
                for k, v in batch.items():
                    batch[k] = v.to(args.device)

                if args.collator_train.num_candidates > 0:
                    labels = batch[cts.LABEL]
                    candidates = batch[cts.CANDIDATES]
                    # remove labels from batch
                    del batch[cts.LABEL]
                    del batch[cts.CANDIDATES]

                    model_output = model.forward(batch)

                    # if distributed, gather all outputs and labels
                    if args.number_of_devices > 1:
                        all_labels = AllGather.apply(labels)
                        all_outputs = AllGather.apply(model_output)
                        all_candidates = AllGather.apply(candidates)
                    else:
                        all_labels = labels
                        all_outputs = model_output
                        all_candidates = candidates

                    all_targets = torch.argmax(
                        (all_candidates == all_labels[:, None]).to(torch.int32), dim=1
                    )
                    cans_predict = torch.gather(all_outputs, 1, all_candidates)
                    loss = criterion(cans_predict, all_targets)
                else:
                    labels = batch[cts.LABEL]
                    # remove labels from batch
                    del batch[cts.LABEL]
                    model_output = model.forward(batch)
                    loss = criterion(model_output, labels)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                step_count += 1
                # evalute every args.eval_steps steps
                if step_count % args.eval_steps == 0:
                    logger.info(
                        f"Evaluating dev set after epoch {epoch}, step {step_count}"
                    )
                    metrics = evaluate(model, val_dataloader, args, accelerator)
                    metrics["epoch"] = epoch
                    metrics["step"] = step_count

                    if accelerator.is_local_main_process:
                        logger.info(f"Metrics: {metrics}")
                        with open(pjoin(args.output_dir, "metrics_val.json"), "a") as f:
                            json.dump(metrics, f)
                            f.write("\n")

                        if (
                            best_metric
                            < metrics[f"{args.save_metric}@{args.metric_ks}"]
                        ):
                            best_metric = metrics[f"{args.save_metric}@{args.metric_ks}"]

                            # save model
                            logger.info("Saving best model")
                            accelerator.save(
                                model.state_dict(),
                                pjoin(args.output_dir, "best_model.bin"),
                            )

                            logger.info("Saving accelerator state")
                            accelerator.save_state(
                                pjoin(args.output_dir, "accelerate_checkpoint")
                            )

                            # reset early stopping
                            early_stopping = args.early_stopping
                        else:
                            early_stopping -= 1
                            if early_stopping <= 0:
                                logger.info(
                                    f"Early stopping triggered after {epoch} epochs (steps: {step_count})."
                                )
                                stop_training = True
                                break
                    # reset model to train mode
                    model.train()

            if stop_training:
                break

    # load best model
    logger.info(f"Loading best model with val {args.save_metric}: {best_metric}@{args.metric_ks}")
    best_ckpt = torch.load(pjoin(args.output_dir, "best_model.bin"))
    model.load_state_dict(best_ckpt)

    logger.info("Evaluating on test set after training.")
    metrics = evaluate(model, test_dataloader, args, accelerator)
    metrics["trained"] = "yes"

    logger.info(f"Metrics: {metrics}")
    with open(pjoin(args.output_dir, "metrics_test.json"), "a") as f:
        json.dump(metrics, f)
        f.write("\n")


if __name__ == "__main__":
    main()
