import torch
import numpy as np


def my_Tloss(pred, gt_idx, device=torch.device("cuda")):
    padding_mask = gt_idx.eq(-1)

    gt_idx[padding_mask] = 0

    gt_idx = gt_idx + padding_mask.int() * gt_idx.max(1).values.unsqueeze(1)

    gt = torch.zeros_like(pred, device=device)
    gt.scatter_(1, gt_idx.long(), 1)

    loss = torch.masked_select(pred, gt.bool()).clamp(min=1e-3).log().neg().sum()

    return loss


class DIST(torch.nn.Module):
    def __init__(
        self,
        feature_sizes,
        BN=False,
        DO=0.0,
        dist_type="EMBEDDING",
        dist_metric="cosine",
        device=torch.device("cuda"),
    ):
        super(DIST, self).__init__()
        self.device = device
        self.feature_sizes = feature_sizes
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(DO)
        self.network = torch.nn.Sequential()
        for layer_num in range(0, len(feature_sizes) - 1):
            self.network.add_module(
                "FC" + str(layer_num),
                torch.nn.Linear(feature_sizes[layer_num], feature_sizes[layer_num + 1]),
            )
            if isinstance(BN, list):
                if BN[layer_num]:
                    self.network.add_module(
                        "BN" + str(layer_num),
                        torch.nn.BatchNorm1d(feature_sizes[layer_num + 1]),
                    )
            else:
                if BN:
                    self.network.add_module(
                        "BN" + str(layer_num),
                        torch.nn.BatchNorm1d(feature_sizes[layer_num + 1]),
                    )
            if layer_num < len(feature_sizes) - 2:
                self.network.add_module("ACT" + str(layer_num), self.activation)
                if DO > 0.0:
                    self.network.add_module("DO" + str(layer_num), self.dropout)
        self.dist_type = dist_type
        self.dist_metric = dist_metric

    def forward(self, input_x):
        x = []
        # TODO: probably can be replaced with a CNN in order to remove loop
        for i in range(input_x.shape[0]):
            x.append(self.network(input_x[i]))

        x = torch.stack(x)
        corr = torch.bmm(x, x.permute(0, 2, 1))

        square_rows = (
            corr.diagonal(dim1=1, dim2=2)
            .repeat(1, corr.shape[-1])
            .reshape(-1, corr.shape[-1], corr.shape[-1])
        )
        square_cols = square_rows.permute(0, 2, 1)

        return (1.0 - corr / (square_rows * square_cols).clamp(min=1e-8).sqrt()) / 2.0


class D_SUM_CALC(torch.nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super(D_SUM_CALC, self).__init__()
        self.device = device

    def forward(self, input_D):
        print(input_D.shape)
        D_sum = torch.zeros_like(input_D, device=self.device)

        # set the diagonal for each batch to be equal
        # to diagonal of `input_D`
        D_sum = D_sum + torch.diag_embed(input_D.diagonal(dim1=1, dim2=2))

        # diagonal at offset 1 is equal to the sum of 2x2 blocks
        # along the main diagonal direction which can be achieved
        # directly with a convolution
        kernel = torch.tensor([[[[1., 1.], [1., 1.]]]], device=self.device)
        diag_blocks = (
            torch.nn.functional.conv2d(torch.unsqueeze(input_D, 1), kernel)
            .squeeze(dim=1)
            .diagonal(dim1=1, dim2=2)
        )
        D_sum = D_sum + torch.diag_embed(diag_blocks, offset=1)
        # diagonal at offset -1 for `D_sum` is the same as for `input_D`
        D_sum = D_sum + torch.diag_embed(
            input_D.diagonal(dim1=1, dim2=2, offset=-1), offset=-1
        )

        # fill rest of the matrix
        # where D[ii, ii + oo] + D[ii + oo, ii] can be replaced
        # with the sum of the transpose to compute all values at once
        D_sum = D_sum + (input_D + input_D.permute(0, 2, 1)).triu(2)

        # diagonals 0, 1 and -1 are filled
        # all other diagonals are partly filled from the line above
        # the final summation for each diagonal depends on values
        # from the previous two thus we loop over each diagonal position
        # and compute the value using a convolution as before defined by a 2x2 block
        kernel = torch.tensor([[[[1., 0.], [-1., 1.]]]], device=self.device)
        for diag_idx in range(2, input_D.shape[1]):
            diag = torch.nn.functional.conv2d(
                torch.unsqueeze(D_sum, 1), kernel
            ).squeeze(dim=1)
            diag = diag.diagonal(offset=diag_idx - 1, dim1=1, dim2=2)
            D_sum = D_sum + torch.diag_embed(diag, offset=diag_idx)

        return D_sum + D_sum.triu(2).permute(0, 2, 1)


class C_TABLE_ALL(torch.nn.Module):
    def __init__(self, K, device=torch.device("cuda")):
        super(C_TABLE_ALL, self).__init__()
        self.K = K
        self.device = device

    def forward(self, input_D_sum):
        b, N, N = input_D_sum.shape

        C = torch.zeros(b, N, self.K, device=self.device)
        C_all = -1 * torch.ones(b, N, self.K, N, device=self.device)

        C[:, :N, 0] = input_D_sum[:, :N, N - 1]
        C_all[:, :N, 0, N - 1] = 1.0

        # TODO needs to be vectorized
        for idx, D_sum in enumerate(input_D_sum.unbind()):
            for kk in range(1, self.K):
                for nn in range(0, N - kk):
                    temp = torch.empty(N - kk - nn, device=self.device)
                    for ii in range(nn, N - kk):
                        temp[ii - nn] = D_sum[nn, ii] + C[idx, ii + 1, kk - 1]
                    C_all[idx, nn, kk, nn : N - kk] = torch.nn.functional.softmin(
                        temp, dim=0
                    )
                    C[idx, nn, kk] = torch.min(temp)

        return C, C_all


class OSG_C(torch.nn.Module):
    def __init__(
        self,
        feature_sizes,
        K_max=30,
        BN=False,
        DO=0.0,
        dist_type="EMBEDDING",
        dist_metric="cosine",
        device=torch.device("cuda"),
    ):
        super(OSG_C, self).__init__()
        self.feature_sizes = feature_sizes
        self.K_max = K_max
        self.DIST_FUNC = DIST(feature_sizes, BN, DO, dist_type, dist_metric, device)
        self.D_SUM_CALC = D_SUM_CALC(device)
        self.C_TABLE_ALL = C_TABLE_ALL(K_max, device)
        self.device = device

    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 0)
        _, C_all = self.C_TABLE_ALL.forward(
            self.D_SUM_CALC.forward(self.DIST_FUNC.forward(x))
        )
        return (C_all * C_all.ge(0)).sum(1).sum(1) / C_all.ge(0).sum(1).sum(1)
