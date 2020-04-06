import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import CrossEntropyLoss


class DSNELoss(_Loss):
    """d-SNE loss for batch of source and target features.

    Attributes
    ----------
    margin : float
        Minimum required margin between `min_intraclass_dist` and
        `max_interclass_dist`.
    fn : Bool
        Flag for whether to use instance normalization on feature
        vectors prior to loss calculation.
    reduction : str
        Name of torch function for reducing loss vector to a scalar. Set
        to "mean" to mimic CrossEntropyLoss default parameter.

    Notes
    -----
    The loss calculation involves the following plain-English steps:

    For each image in the target batch...
        1. Compare its feature vector to all FVs in the source batch
            using a distance function. (L2-norm/Euclidean used here.)
        2. Find the minimum interclass distance (y_tgt != y_src) and
            maximum intraclass distance (y_tgt == y_src)
        3. Check that the difference between these two distances is
            greater than a specified margin.

        Explanation:
            -The minimum interclass distance should be large, as FVs
                from src/tgt pairs should be as distinct as possible
                 when their classes are different.
            -The maximum intraclass distance should be small, as FVs
                from src/tgt pairs should be as similar as possible
                when their classes are the same.
            -Therefore, this condition should be true:
                              `min_interclass` >> `max_interclass`
           `min_interclass` - `max_interclass` >> `margin`

        4. Calculate loss for cases where the difference is NOT greater
            than the margin. Here, loss == abs(difference).
    """

    def __init__(self, margin=1.0, fn=False, reduction="mean"):
        """Assign parameters as attributes."""
        super(DSNELoss, self).__init__()

        self.margin = margin
        self.fn = fn

        if reduction == "mean":
            self.reduce_func = torch.mean
        else:
            raise NotImplementedError

    def forward(self, fts, ys, ftt, yt):
        """Compute forward pass for loss function."""
        # If src_batch -> (N, F) and tgt_batch -> (M, F), then
        # distances for all pairs will be of shape (N, M, F)
        broadcast_size = (fts.shape[0], ftt.shape[0], fts.shape[1])

        # Compute distances between all pairs of target and source vectors
        fts_rpt = fts.unsqueeze(0).expand(broadcast_size)
        ftt_rpt = ftt.unsqueeze(1).expand(broadcast_size)
        dists = torch.sum((ftt_rpt - fts_rpt)**2, dim=2)

        # Split source/target distances into 2 groups:
        #   1. intraclass distances (y_tgt == y_src)
        #   2. interclass distances (y_tgt != y_src)
        ys_rpt = yt.unsqueeze(0).expand(broadcast_size)
        yt_rpt = yt.unsqueeze(1).expand(broadcast_size)
        y_same = torch.eq(yt_rpt, ys_rpt)   # Boolean mask
        y_diff = torch.logical_not(y_same)  # Boolean mask
        intra_cls_dists = dists * y_same    # Set 0 where classes are different
        inter_cls_dists = dists * y_diff    # Set 0 where classes are the same

        # Fill 0 values with max to prevent interference with min calculation
        max_dists = torch.max(dists, dim=1, keepdim=True)[0]
        max_dists = max_dists.expand(broadcast_size[0:2])
        inter_cls_dists = torch.where(y_same, max_dists, inter_cls_dists)

        # For each target image, find the minimum interclass distance
        min_inter_cls_dist = inter_cls_dists.min(1)[0][0, :]

        # For each target image, find the maximum intraclass distance
        max_intra_cls_dist = intra_cls_dists.max(1)[0][0, :]

        # No loss for differences greater than margin (clamp to 0)
        differences = min_inter_cls_dist.sub(max_intra_cls_dist)
        loss = torch.abs(differences.sub(self.margin).clamp(max=0))

        return self.reduce_func(loss)


class CombinedLoss(_WeightedLoss):
    """Combine typical cross entropy loss with d-SNE loss using a
    weighted sum.

    Attributes
    ----------
    loss_dnse
        Instance of DSNELoss class.
    loss_xent
        Instance of CrossEntropyLoss class.
    alpha : float
        Scale factor for weighted sum of losses.
    """

    def __init__(self, margin=1.0, fn=False, alpha=0.1, reduction="mean"):
        """Create the loss functions to be weighted.

        Parameters
        ----------
        margin : float
            Minimum required margin between `min_intraclass_dist` and
            `max_interclass_dist`.
        fn : Bool
            Flag for whether to use instance normalization on feature
            vectors prior to loss calculation.
        alpha : float
            Scale factor for weighted sum of losses.
        """
        super(CombinedLoss, self).__init__()

        self.loss_dsne = DSNELoss(margin, fn, reduction)
        self.loss_xent = CrossEntropyLoss(reduction=reduction)
        self.alpha = alpha

    def forward(self, ft_src, y_pred_src, y_src, ft_tgt, y_pred_tgt, y_tgt):
        """Compute forward-pass for loss function."""
        loss_xent = self.loss_xent(y_pred_src, y_src.long())
        loss_dsne = self.loss_dsne(ft_src, y_src, ft_tgt, y_tgt)

        return (1 - self.alpha)*loss_xent + self.alpha*loss_dsne
