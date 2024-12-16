import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.Local_global import Seasonal_Prediction, series_decomp_multi


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.mode = configs.mode

        c_out = configs.c_out
        dec_in = configs.dec_in
        d_model = configs.d_model
        n_heads = configs.n_heads
        d_layers = configs.d_layers
        dropout = configs.dropout
        embed = configs.embed
        freq = configs.freq
        device = torch.device("cuda:{}".format(configs.gpu))
        decomp_kernel = configs.decomp_kernel
        conv_kernel = configs.conv_kernel
        isometric_kernel = configs.isometric_kernel

        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        self.conv_trans = Seasonal_Prediction(
            embedding_size=d_model,
            n_heads=n_heads,
            dropout=dropout,
            d_layers=d_layers,
            decomp_kernel=decomp_kernel,
            c_out=c_out,
            conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel,
            device=device,
        )

        self.regression = nn.Linear(self.seq_len, self.pred_len)
        self.regression.weight = nn.Parameter(
            (1 / self.pred_len) * torch.ones([self.pred_len, self.seq_len]),
            requires_grad=True,
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        # trend-cyclical prediction block: regre or mean
        if self.mode == "regre":
            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.mode == "mean":
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            seasonal_init_enc, trend = self.decomp_multi(x_enc)
            trend = torch.cat([trend[:, -self.seq_len :, :], mean], dim=1)

        # embedding
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init_dec = torch.cat(
            [seasonal_init_enc[:, -self.seq_len :, :], zeros], dim=1
        )
        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)

        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len :, :] + trend[:, -self.pred_len :, :]
        return dec_out
