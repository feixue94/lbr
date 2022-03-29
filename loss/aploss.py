# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/15 下午2:30
@Auth ： Fei Xue
@File ： aploss.py
@Email： xuefei@sensetime.com
"""

### code is from https://github.com/Andrew-Brown1/Smooth_AP

import torch
import torch.nn as nn
import numpy as np

"""==============================================Smooth-AP========================================"""


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal=0.01, batch_size=4, num_id=4, feat_dims=128):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP, self).__init__()

        assert (batch_size % num_id == 0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size)
        # print("mask: ", mask.size())
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # print("mask: ", mask.size()) # B B B
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)  # B B B
        # print("sim_all: ", sim_all.size())
        # print("sim_all_repeat: ", sim_all_repeat.size())
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)  # N, S, D
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))  # samples of each class
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1,
                                                                     1)
        # print("pose_mask: ", pos_mask.size(), pos_mask)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))  # N, SxS
        # print("sim_pos: ", sim_pos.size(), sim_pos)
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)  # N SxSxS
        # print("sim_pose_repeat: ", sim_pos_repeat.size())
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.cuda()
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1
        # print("sim_pos_rk: ", sim_pos_rk.size())

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(
                sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return (1 - ap)


class APLoss(nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq - 1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a * min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2 - nq, 2, 1) - a * min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_ap': float(loss)}


class TAPLoss(APLoss):
    """ Differentiable tie-aware AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def __init__(self, nq=25, min=0, max=1, simplified=False):
        APLoss.__init__(self, nq=nq, min=min, max=max)
        self.simplified = simplified

    def forward(self, x, label, qw=None, ret='1-mAP'):
        '''N: number of images;
           M: size of the descs;
           Q: number of bins (nq);
        '''
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        label = label.float()
        Np = label.sum(dim=-1, keepdim=True)

        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        c = q.sum(dim=-1)  # number of samples  N x Q = nbs on APLoss
        cp = (q * label.view(N, 1, M)).sum(dim=-1)  # N x Q number of correct samples = rec on APLoss
        C = c.cumsum(dim=-1)
        Cp = cp.cumsum(dim=-1)

        zeros = torch.zeros(N, 1).to(x.device)
        C_1d = torch.cat((zeros, C[:, :-1]), dim=-1)
        Cp_1d = torch.cat((zeros, Cp[:, :-1]), dim=-1)

        if self.simplified:
            aps = cp * (Cp_1d + Cp + 1) / (C_1d + C + 1) / Np
        else:
            eps = 1e-8
            ratio = (cp - 1).clamp(min=0) / ((c - 1).clamp(min=0) + eps)
            aps = cp * (c * ratio + (Cp_1d + 1 - ratio * (C_1d + 1)) * torch.log((C + 1) / (C_1d + 1))) / (c + eps) / Np
        aps = aps.sum(dim=-1)

        assert aps.numel() == N

        if ret == '1-mAP':
            if qw is not None:
                aps *= qw  # query weights
            return 1 - aps.mean()
        elif ret == 'AP':
            assert qw is None
            return aps
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_tap' + ('s' if self.simplified else ''): float(loss)}


class TripletMarginLoss(nn.TripletMarginLoss):
    """ PyTorch's margin triplet loss
    TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False, size_average=True, reduce=True)
    """

    def eval_func(self, dp, dn):
        return max(0, dp - dn + self.margin)


class TripletLogExpLoss(nn.Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.
    The distance is described in detail in the paper `Improving Pairwise Ranking for Multi-Label
    Image Classification`_ by Y. Li et al.
    .. math::
        L(a, p, n) = log \left( 1 + exp(d(a_i, p_i) - d(a_i, n_i) \right)
    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`
    >>> triplet_loss = nn.TripletLogExpLoss(p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> input3 = autograd.Variable(torch.randn(100, 128))
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()
    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, p=2, eps=1e-6, swap=False):
        super(TripletLogExpLoss, self).__init__()
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
        assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
        assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
        assert anchor.dim() == 2, "Input must be a 2D matrix."

        d_p = F.pairwise_distance(anchor, positive, self.p, self.eps)
        d_n = F.pairwise_distance(anchor, negative, self.p, self.eps)

        if self.swap:
            d_s = F.pairwise_distance(positive, negative, self.p, self.eps)
            d_n = torch.min(d_n, d_s)

        dist = torch.log(1 + torch.exp(d_p - d_n))
        loss = torch.mean(dist)
        return loss

    def eval_func(self, dp, dn):
        return np.log(1 + np.exp(dp - dn))


def sim_to_dist(scores):
    return 1 - torch.sqrt(2.001 - 2 * scores)


class APLoss_dist(APLoss):
    def forward(self, x, label, **kw):
        d = sim_to_dist(x)
        return APLoss.forward(self, d, label, **kw)


class TAPLoss_dist(TAPLoss):
    def forward(self, x, label, **kw):
        d = sim_to_dist(x)
        return TAPLoss.forward(self, d, label, **kw)


if __name__ == '__main__':
    loss = SmoothAP(0.01, 6, 2, 256)
    input = torch.randn(6, 256, requires_grad=True).cuda()
    output = loss(input)
    output.backward()
    print(output)
