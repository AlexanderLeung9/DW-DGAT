import numpy as np
import torch
import torch.nn as nn
import torch.autograd as ta
import torch.optim.optimizer as too
import networks as nw
import arguments as ag
from abc import abstractmethod


class Manifold(object):
    def allocate_lt(self, N, dim, sparse):
        return nn.Embedding(N, dim, sparse=sparse)

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    def init_weights(self, w, scale=1e-4):
        w.weight.data.uniform_(-scale, scale)

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError

    def norm(self, u, **kwargs):
        if isinstance(u, nn.Embedding):
            u = u.weight
        return u.pow(2).sum(dim=-1).sqrt()

    @abstractmethod
    def half_aperture(self, u):
        """
        Compute the half aperture of an entailment cone.
        As in: https://arxiv.org/pdf/1804.01882.pdf
        """
        raise NotImplementedError

    @abstractmethod
    def angle_at_u(self, u, v):
        """
        Compute the angle between the two half lines (0u and uv
        """
        raise NotImplementedError


class EuclideanManifold(Manifold):
    __slots__ = ["max_norm"]

    def __init__(self, max_norm=None, K=None, **kwargs):
        self.max_norm = max_norm
        self.K = K
        if K is not None:
            self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))

    def normalize(self, u):
        d = u.size(-1)
        if self.max_norm:
            u.view(-1, d).renorm_(2, 0, self.max_norm)
        return u

    def distance(self, u, v):
        return (u - v).pow(2).sum(dim=-1)

    def rgrad(self, p, d_p):
        return d_p

    def half_aperture(self, u):
        sqnu = u.pow(2).sum(dim=-1)
        return torch.asin(self.inner_radius / sqnu.sqrt())

    def angle_at_u(self, u, v):
        norm_u = self.norm(u)
        norm_v = self.norm(v)
        dist = self.distance(v, u)
        num = norm_u.pow(2) - norm_v.pow(2) - dist.pow(2)
        denom = 2 * norm_v * dist
        return (num / denom).acos()

    def expm(self, p, d_p, normalize=False, lr=None, out=None):
        if lr is not None:
            d_p.mul_(-lr)
        if out is None:
            out = p
        out.add_(d_p)
        if normalize:
            self.normalize(out)
        return out

    def logm(self, p, d_p, out=None):
        return p - d_p

    def ptransp(self, p, x, y, v):
        ix, v_ = v._indices().squeeze(), v._values()
        return p.index_copy_(0, ix, v_)


class Distance(ta.Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        b = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        b = b * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * b / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps):
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None


class PoincareManifold(EuclideanManifold):
    def __init__(self, eps=1e-5, K=None):
        self.eps = eps
        super(PoincareManifold, self).__init__(max_norm=1 - eps)
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))

    def distance(self, u, v):
        return Distance.apply(u, v, self.eps)

    def half_aperture(self, u):
        eps = self.eps
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return torch.asin((self.inner_radius * (1 - sqnu) / torch.sqrt(sqnu))
            .clamp(min=-1 + eps, max=1 - eps))

    def angle_at_u(self, u, v):
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + norm_v ** 2) - norm_v ** 2 * (1 + norm_u ** 2))
        denom = (norm_v * edist * (1 + norm_v**2 * norm_u**2 - 2 * dot_prod).sqrt())
        return (num / denom).clamp_(min=-1 + self.eps, max=1 - self.eps).acos()

    def rgrad(self, p, d_p):
        if d_p.is_sparse:
            p_sqnorm = torch.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            n_vals.renorm_(2, 0, 5)
            d_p = torch.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
        else:
            p_sqnorm = torch.sum(p ** 2, dim=-1, keepdim=True)
            d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p


class HyLa(nn.Module):
    def __init__(self, params: nw.NetworkParams):
        super(HyLa, self).__init__()

        assert isinstance(params.args, ag.HyLaNetArgs)
        self.he_dim = params.args.he_dim
        self.manifold = PoincareManifold()
        self.lt = self.manifold.allocate_lt(params.args.feature_num, self.he_dim, params.args.sparse)
        self.manifold.init_weights(self.lt)
        self.Lambdas = params.args.lambda_scale * torch.randn(params.args.hyla_dim)
        self.boundary = HyLa.__sample_boundary(params.args.hyla_dim, self.he_dim, cls="RandomUniform")
        self.bias = 2 * np.pi * torch.rand(params.args.hyla_dim)

    def forward(self) -> [[float]]:
        with torch.no_grad():
            e_all = self.manifold.normalize(self.lt.weight)
        PsK = HyLa.__poisson_kernel(e_all, self.boundary.to(e_all.device))
        angles = self.Lambdas.to(e_all.device) / 2.0 * torch.log(PsK)
        eigs = torch.cos(angles + self.bias.to(e_all.device)) * torch.sqrt(PsK) ** (self.he_dim - 1)
        return eigs

    def optim_params(self):
        return [
            {
                "params": self.lt.parameters(),
                "rgrad": self.manifold.rgrad,
                "expm": self.manifold.expm,
                "logm": self.manifold.logm,
                "ptransp": self.manifold.ptransp,
            }
        ]

    @staticmethod
    def __sample_boundary(n_Bs, d, cls):
        if cls == "RandomUniform" or d > 2:
            pre_b = torch.randn(n_Bs, d)
            b = pre_b / torch.norm(pre_b, dim=-1, keepdim=True)
        elif cls == "FixedUniform":
            theta = torch.arange(0, 2 * np.pi, 2 * np.pi / n_Bs)
            b = torch.stack([torch.cos(theta), torch.sin(theta)], 1)
        elif cls == "RandomDisk":
            theta = 2 * np.pi * torch.rand(n_Bs)
            b = torch.stack([torch.cos(theta), torch.sin(theta)], 1)
        else:
            raise NotImplementedError(f"cls={cls}")
        return b

    @staticmethod
    def __poisson_kernel(X, b):
        X = X.view(X.size(0), 1, X.size(-1))
        return (1 - torch.norm(X, 2, dim=-1)**2)/(torch.norm(X-b, 2, dim=-1)**2)


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, params: nw.NetworkParams):
        super(SGC, self).__init__()

        assert isinstance(params.args, ag.HyLaNetArgs)
        self.W = nn.Linear(params.args.hyla_dim, params.args.class_num)

    def forward(self, x):
        return self.W(x)


class HyLaNet(nw.NetworkBase):
    """
    @article{yu2022random,
      title={Random laplacian features for learning with hyperbolic space},
      author={Yu, Tao and De Sa, Christopher},
      journal={arXiv preprint arXiv:2202.06854},
      year={2022}
    }
    """
    def __init__(self, hy_la: HyLa, sgc: SGC):
        super().__init__(None)

        self.hy_la = hy_la
        self.sgc = sgc

    def forward(self, x: [[float]]) -> [[float]]:
        eigs = self.hy_la.forward()
        HyLa_features = torch.mm(x[self.current_indices], eigs)
        scores = self.sgc.forward(HyLa_features)
        return scores


class RiemannianSGD(too.Optimizer):
    """
    Riemannian stochastic gradient descent.

    Args:
        rgrad (Function): Function to compute the Riemannian gradient from the Euclidean gradient
    """
    def __init__(
            self,
            params,
            lr=too.required,
            rgrad=too.required,
            expm=too.required,
    ):
        defaults = {
            'lr': lr,
            'rgrad': rgrad,
            'expm': expm,
        }
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None, counts=None, **kwargs):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
            counts: unknown
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']

                if p.grad is None:
                    continue
                d_p = p.grad.data
                # make sure we have no duplicates in sparse tensor
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = rgrad(p.data, d_p)
                d_p.mul_(-lr)
                expm(p.data, d_p)

        return loss
