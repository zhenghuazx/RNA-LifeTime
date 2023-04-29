import torch
from torch import nn
import torch.nn.functional as F
from .TruncatedNormal import TruncatedNormal


class MLPModel(nn.Module):

    def __init__(self, ftr_dim, emb, contact_type_dim=10, truncated=True):
        """
        :param emb:
        :param heads:
        :param mask:
        :param kqnorm:
        """

        super().__init__()

        self.feature_dim = emb * emb + emb * emb * contact_type_dim + ftr_dim

        self.env_effect_time = nn.Linear(1, 1, bias=True)
        if truncated:
            self.normal = TruncatedNormal(torch.tensor([0.0]), torch.tensor([1.0]), a=0, b=float('inf'))
        else:
            self.normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.ff1 = nn.Sequential(
            nn.Linear(self.feature_dim, emb * emb)
        )
        self.env_effect_sigma = nn.Sequential(
            nn.Linear(ftr_dim, 1),
            nn.Softplus()
        )
        self.norm1 = nn.BatchNorm1d(self.feature_dim)

    def forward(self, frame, x, bond, feature, mask):
        # x = torch.cdist(loc, loc, p=2.0)
        b, t, e = x.size()
        b, t, e, c = bond.size()
        x1 = x.view(b, t * e)
        x2 = bond.view(b, t * e * c)
        x3 = feature.view(b, 2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.ff1(x).view(b, e, t)
        sigma = 10 * self.env_effect_sigma(feature)
        mean = (torch.log(frame).view(b, 1).unsqueeze(1) - x).view(b, t * e)
        kernel = self.normal.log_prob(mean)
        S_t = 1 - torch.exp(kernel.view(b, t, e) / sigma.unsqueeze(1))
        return S_t[mask > 0]


class DegradationModel(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, ftr_dim, emb, contact_type_dim=10, num_gaussians=5, ff_hidden_mult=2, dropout=0.2, mask=False,
                 truncated=False):
        """
        :param emb:
        :param heads:
        :param mask:
        :param kqnorm:
        """

        super().__init__()

        self.emb = emb
        self.num_gaussians = num_gaussians
        self.truncated = truncated

        self.contact_potential_param = nn.Linear(contact_type_dim, 3 * self.num_gaussians,
                                                 bias=False)  # 3 represent B, C, R
        self.coulomb = nn.Linear(emb * emb, emb * emb, bias=False)
        self.W_2 = torch.randn(emb, emb, dtype=torch.float32, requires_grad=True)
        self.W_2 = nn.Parameter(self.W_2)
        self.env_effect_c = nn.Linear(ftr_dim, 1, bias=True)
        self.env_effect_e = nn.Linear(ftr_dim, 1, bias=True)
        self.env_effect_v = nn.Linear(1, 1, bias=True)
        self.env_effect_sigma = nn.Linear(ftr_dim, 1, bias=True)
        self.env_effect_time = nn.Linear(1, 1, bias=True)
        if self.truncated:
            self.normal = TruncatedNormal(torch.tensor([0.0]), torch.tensor([1.0]), a=0, b=float('inf'))
        else:
            # self.normal = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            self.normal = torch.distributions.half_normal.HalfNormal(torch.tensor([1.0]))
        self.ff1 = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.env_effect_sigma = nn.Sequential(
            nn.Linear(ftr_dim, ftr_dim * ff_hidden_mult),
            nn.Sigmoid(),
            nn.Linear(ff_hidden_mult * ftr_dim, 1),
            nn.Softplus()
        )
        self.norm1 = nn.BatchNorm1d(emb)
        self.norm2 = nn.BatchNorm1d(emb)
        self.doupout = nn.Dropout(dropout)

    def forward(self, frame, x, bond, feature, mask):
        b, t, e = x.size()
        params = self.contact_potential_param(bond)  # b, N, N, 3
        V_c = 0
        for i in range(self.num_gaussians):
            B, C, R = params[:, :, :, 0 + 3 * i], params[:, :, :, 1 + 3 * i], params[:, :, :, 2 + 3 * i]
            V_c += B * torch.exp(- C * torch.square(x - R))
        scaler_c = self.env_effect_c(feature)
        V_c = torch.einsum('ijk, im -> ijk', V_c, F.sigmoid(scaler_c))
        vec_dis = 1 / x  # .view(b, t * e)
        V_e = self.W_2 / vec_dis
        V = V_c + V_e
        V = self.doupout(V)
        fedforward1 = self.ff1(V)
        V1 = self.norm1(fedforward1 + V)
        V_tran = V.transpose(-1, -2)
        fedforward2 = self.ff2(V_tran)
        V2 = self.norm2(fedforward2 + V_tran).transpose(-1, -2)
        V3 = V1 + V2
        sigma = 10 * self.env_effect_sigma(feature)
        time_ff = self.env_effect_time(torch.log(frame).view(b, 1))
        T = self.env_effect_v(feature[:, 0].unsqueeze(1)).unsqueeze(1)
        mean = (time_ff.view(b, 1).unsqueeze(1) - V3 / T).view(b, t * e)
        if self.truncated:
            mean = torch.abs(mean)
        kernel = self.normal.log_prob(mean)
        S_t = 1 - torch.exp(kernel.view(b, t, e) / sigma.unsqueeze(1))
        return S_t[mask > 0]

