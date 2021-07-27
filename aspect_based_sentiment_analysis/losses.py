import torch.nn.functional as F

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}
