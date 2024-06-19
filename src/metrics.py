import torch
from transformers4rec.torch.ranking_metric import RankingMetric
from transformers4rec.torch.utils import torch_utils


class RecallAt(RankingMetric):
    def __init__(self, top_ks=None, labels_onehot=False):
        super(RecallAt, self).__init__(top_ks=top_ks, labels_onehot=labels_onehot)

    def _metric(
        self, ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute recall@K for each of the provided cutoffs

        Parameters
        ----------
        ks : torch.Tensor or list
            list of cutoffs
        scores : torch.Tensor
            predicted item scores
        labels : torch.Tensor
            true item labels

        Returns
        -------
            torch.Tensor: list of recalls at cutoffs
        """

        ks, scores, labels = torch_utils.check_inputs(ks, scores, labels)
        _, _, topk_labels = torch_utils.extract_topk(ks, scores, labels)
        recalls = torch_utils.create_output_placeholder(scores, ks)

        # Compute recalls at K
        num_relevant = torch.sum(labels, dim=-1)
        rel_indices = (num_relevant != 0).nonzero()
        rel_count = num_relevant[rel_indices]
        # print(rel_indices)
        # exit(0)
        # try rel_indices.shape[0] > 0:
        #     print(rel_indices)
        # except Exception as e:
        #     raise e
        if rel_indices.shape[0] > 0:
            for index, k in enumerate(ks):
                rel_labels = topk_labels[rel_indices, : int(k)]

                recalls[rel_indices, index] = torch.div(
                    torch.sum(rel_labels, dim=-1), rel_count
                ).to(
                    dtype=torch.float32
                )  # Ensuring type is double, because it can be float if --fp16

        return recalls
