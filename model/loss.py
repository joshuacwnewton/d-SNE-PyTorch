import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import CrossEntropyLoss


class DSNELoss(_Loss):
    """d-SNE loss for paired batches of source and target features.

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
        to "mean" by default to mimic CrossEntropyLoss default
        parameter.

    Notes
    -----
    The loss calculation involves the following plain-English steps:

    For each image in the training batch...
        1. Compare its feature vector to all FVs in the comparison batch
            using a distance function. (L2-norm/Euclidean used here.)
        2. Find the minimum interclass distance (y_trn != y_cmp) and
            maximum intraclass distance (y_trn == y_cmp)
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

    def forward(self, ft, y, trn):
        """Compute forward pass for loss function.

        Parameters
        ----------
        ft : dict of PyTorch Tensors (N, F)
            Source and target feature vectors.
        y : dict of PyTorch Tensors (N, 1)
            Source and target labels.
        trn : str
            Key corresponding to which of (source, target) will be used
            for training the network. The other will be used to generate
            distance comparisons.
        """
        # Argument validation
        if list(ft.keys()) != list(y.keys()):
            raise KeyError("ft and y do not have same keys.")
        names = list(ft.keys())
        if len(names) is not 2:
            raise ValueError("ft and y must be dicts of length 2")
        if trn not in names:
            raise KeyError(f"Training name '{trn}' not in Tensor dicts.")

        # Comparison name is whatever name (src/tgt) isn't the training name
        names.remove(trn)
        cmp = names[0]

        # If training batch -> (N, F) and comparison batch -> (M, F), then
        # distances for all combinations of pairs will be of shape (N, M, F)
        broadcast_size = (ft[trn].shape[0], ft[cmp].shape[0], ft[trn].shape[1])

        # Compute distances between all <train, comparison> pairs of vectors
        ft_trn_rpt = ft[trn].unsqueeze(0).expand(broadcast_size)
        ft_cmp_rpt = ft[cmp].unsqueeze(1).expand(broadcast_size)
        dists = torch.sum((ft_trn_rpt - ft_cmp_rpt)**2, dim=2)

        # Split <train, comparison> distances into 2 groups:
        #   1. intraclass distances (y_trn == y_cmp)
        #   2. interclass distances (y_trn != y_cmp)
        y_trn_rpt = y[trn].unsqueeze(0).expand(broadcast_size)
        y_cmp_rpt = y[cmp].unsqueeze(1).expand(broadcast_size)
        y_same = torch.eq(y_trn_rpt, y_cmp_rpt)   # Boolean mask
        y_diff = torch.logical_not(y_same)        # Boolean mask
        intraclass_dists = dists * y_same   # Set 0 where classes are different
        interclass_dists = dists * y_diff   # Set 0 where classes are the same

        # Fill 0 values with max to prevent interference with min calculation
        max_dists = torch.max(dists, dim=1, keepdim=True)[0]
        max_dists = max_dists.expand(broadcast_size[0:2])
        interclass_dists = torch.where(y_same, max_dists, interclass_dists)

        # For each training image, find the minimum interclass distance
        min_interclass_dist = interclass_dists.min(1)[0][0, :]

        # For each training image, find the maximum intraclass distance
        max_intraclass_dist = intraclass_dists.max(1)[0][0, :]

        # No loss for differences greater than margin (clamp to 0)
        differences = min_interclass_dist.sub(max_intraclass_dist)
        loss = torch.abs(differences.sub(self.margin).clamp(max=0))

        return self.reduce_func(loss)


class CombinedLoss(_WeightedLoss):
    """Combine typical cross entropy loss with d-SNE loss using a
    weighted sum.

    Attributes
    ----------
    loss_dsne
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
        reduction : str
            Name of function used to reduce loss Tensor to scalar.
        """
        super(CombinedLoss, self).__init__()

        self.loss_dsne = DSNELoss(margin, fn, reduction)
        self.loss_xent = CrossEntropyLoss(reduction=reduction)
        self.alpha = alpha

    def forward(self, ft, y_pred, y, train_name):
        """Compute forward-pass for loss function.

        Parameters
        ----------
        ft : dict of PyTorch Tensors (N, F)
            Feature vectors extracted from source and target batch.
        y_pred : dict of PyTorch Tensors (N, 1)
            Labels predicted by model for source and target batch.
        y : dict of PyTorch Tensors (N, 1)
            Actual labels of source and target batch.
        train_name
            Key corresponding to which of (source, target) will be used
            for training the network. The other will be used to generate
            distance comparisons in DSNELoss calculation.
        """
        loss_xent = self.loss_xent(y_pred[train_name], y[train_name].long())
        loss_dsne = self.loss_dsne(ft, y, train_name)

        return (1 - self.alpha)*loss_xent + self.alpha*loss_dsne
