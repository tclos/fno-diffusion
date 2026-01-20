# fno_diffusion/model.py

from neuralop.models import FNO


def make_fno_1d(n_modes=16, hidden_channels=64, in_channels=1, out_channels=1):
    model = FNO(
        n_modes=(n_modes,),
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return model
