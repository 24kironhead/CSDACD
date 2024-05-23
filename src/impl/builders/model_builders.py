# Custom model builders

from core.misc import MODELS

# @MODELS.register_func('P2V_model')
# def build_p2v_model(C):
#     from models.p2v import P2VNet
#     return P2VNet(**C['p2v_model'])


@MODELS.register_func('CSDACD_EF_model')
def build_p2v_model(C):
    from models.p2v_EF import P2VNet
    return P2VNet(**C['p2v_model'])

@MODELS.register_func('CSDACD_FF_model')
def build_p2v_model(C):
    from models.p2v_FF import P2VNet
    return P2VNet(**C['p2v_model'])

@MODELS.register_func('CSDACD_OF_model')
def build_p2v_model(C):
    from models.p2v_OF import P2VNet
    return P2VNet(**C['p2v_model'])

@MODELS.register_func('GAN_SW_model')
def build_gan_model(C):
    from models.models_gan import GeneratorResNet
    return GeneratorResNet(**C['gan_model'])

@MODELS.register_func('GAN_WS_model')
def build_gan_model(C):
    from models.models_gan import GeneratorResNet
    return GeneratorResNet(**C['gan_model'])

@MODELS.register_func('DIS_S_model')
def build_gan_model(C):
    from models.models_gan import Discriminator
    return Discriminator(**C['dis_model'])

@MODELS.register_func('DIS_W_model')
def build_gan_model(C):
    from models.models_gan import Discriminator
    return Discriminator(**C['dis_model'])
