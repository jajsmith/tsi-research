import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA


class ARIMAGenerator:
    def eval(self):
        return self

    def to(self, device):
        return self

    def forward_conditional(self, past, current, sig_inds):
        """AFAIK, sig_inds is a list of feature indices, and we want to mask all BUT those features"""
        sig_inds_comp = list(set(range(past.shape[-2]))-set(sig_inds))
        if len(current.shape) == 1:
            current = current.unsqueeze(0)

        data = past.cpu().detach().numpy()
        batch_size, num_fts, num_ts = data.shape
        pred = np.zeros((batch_size, num_fts))

        if num_ts > 1:
            for sample in range(batch_size):
                for ft in range(num_fts):
                    model = ARIMA(data[sample, ft, :])
                    model = model.fit()
                    pred[sample, ft] = model.forecast()
        else:
            pred[:, :] = data[:, :, 0]

        full_sample = current.clone()
        full_sample[:, sig_inds_comp] = torch.from_numpy(pred[:, sig_inds_comp]).float()
        return full_sample, None
