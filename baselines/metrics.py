import torch
from torchmetrics.aggregation import MeanMetric
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG, RetrievalRecall


class Ranker:
    """
    A class used to compute ranking metrics:  MRR, NDCG, and Recall.

    Attributes
    ----------
    topk : int, optional
        The number of top elements to consider for the metrics (default is None).
    mrr : RetrievalMRR
        An instance of the RetrievalMRR metric.
    ndcg : RetrievalNormalizedDCG
        An instance of the RetrievalNormalizedDCG metric.
    rec : RetrievalRecall
        An instance of the RetrievalRecall metric.
    mrr_mean : MeanMetric
        An instance of the MeanMetric to track the average MRR score.
    ndcg_mean : MeanMetric
        An instance of the MeanMetric to track the average NDCG score.
    rec_mean : MeanMetric
        An instance of the MeanMetric to track the average Recall score.

    Methods
    -------
    __call__(scores, labels)
        Computes the MRR, NDCG, and Recall scores for the given scores and labels and updates the mean metrics.

    __compute__()
        Computes the average MRR, NDCG, and Recall scores.
    """

    def __init__(self, topk: int = None):
        # instantiate the metrics
        self.mrr = RetrievalMRR(
            top_k=topk, sync_on_compute=False, compute_with_cache=True
        )
        self.ndcg = RetrievalNormalizedDCG(
            top_k=topk, sync_on_compute=False, compute_with_cache=True
        )

        self.rec = RetrievalRecall(
            top_k=topk, sync_on_compute=False, compute_with_cache=True
        )

        # instantiate the MeanMetric to track averages
        self.mrr_mean = MeanMetric(sync_on_compute=False, compute_with_cache=True)
        self.ndcg_mean = MeanMetric(sync_on_compute=False, compute_with_cache=True)
        self.rec_mean = MeanMetric(sync_on_compute=False, compute_with_cache=True)

        self.topk = topk

    def __call__(self, scores, labels):
        # create binary tensor (batch_size, num_labels_total)) with 1 at the position of the correct/target label
        targets = torch.zeros(scores.shape[0], scores.shape[1], device=scores.device)
        targets[torch.arange(scores.shape[0]), labels] = 1

        # flatten the scores and targets
        scores_flat = scores.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).long()
        
        # TODO: check if the following is necessary
        # # Ensure tensors are of a supported data type
        # if scores_flat.dtype == torch.complex64:
        #     scores_flat = scores_flat.real
        # if targets_flat.dtype == torch.complex64:
        #     targets_flat = targets_flat.real

        # create indexes tensor to keep track of the original indexes
        # each sample (ID/label) is reapeated num_labels_total times
        indexes = (
            torch.arange(scores.shape[0], device=scores.device)
            .unsqueeze(1)
            .repeat(1, scores.shape[1])
            .view(-1)
        )

        # compute the scores for batch
        mrr_score = self.mrr(scores_flat, targets_flat, indexes).item()
        ndcg_score = self.ndcg(scores_flat, targets_flat, indexes).item()
        rec_score = self.rec(scores_flat, targets_flat, indexes).item()

        # reset the metrics to not go OOM
        self.mrr.reset()
        self.ndcg.reset()
        self.rec.reset()

        # update the mean metrics
        self.mrr_mean.update(mrr_score)
        self.ndcg_mean.update(ndcg_score)
        self.rec_mean.update(rec_score)

    def compute(self):
        # get the average scores
        mrr_avg = self.mrr_mean.compute().item()
        ndcg_avg = self.ndcg_mean.compute().item()
        rec_avg = self.rec_mean.compute().item()

        # prepare results
        results = {
            f"MRR@{self.topk}": mrr_avg,
            f"NDCG@{self.topk}": ndcg_avg,
            f"RECALL@{self.topk}": rec_avg,
        }

        return results
