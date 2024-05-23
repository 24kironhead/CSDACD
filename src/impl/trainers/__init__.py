from core.misc import R
from .cd_trainer import CDTrainer
from .cd_trainer_nll import CDTrainer_NLL
from .cd_trainer_bce import CDTrainer_BCE
from .cd_trainer_metric import CDTrainer_metric
from .cd_trainer_csdacd import CSDACDTrainer

__all__ = []

trainer_switcher = R['Trainer_switcher']
trainer_switcher.add_item(lambda C: C['out_type']=='logits', CDTrainer_BCE)
trainer_switcher.add_item(lambda C: C['out_type']=='logits2', CDTrainer_NLL)
trainer_switcher.add_item(lambda C: C['out_type']=='dist', CDTrainer_metric)
trainer_switcher.add_item(lambda C: C['model'] in ('CSDACD_EF+GAN_SW+GAN_WS+DIS_S+DIS_W', 'CSDACD_FF+GAN_SW+GAN_WS+DIS_S+DIS_W', 'CSDACD_OF+GAN_SW+GAN_WS+DIS_S+DIS_W'), CSDACDTrainer)