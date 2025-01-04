from .super_net import make_super_net, super_net_x30p, super_net_x20p, super_net_x10p, super_net_x8p, super_net_x6p, \
    super_net_x4p, super_net_x2p


def make_model(model_name):
    model_dict = {
        'super_net': make_super_net,
        'super_net_x30p': super_net_x30p,
        'super_net_x20p': super_net_x20p,
        'super_net_x10p': super_net_x10p,
        'super_net_x8p': super_net_x8p,
        'super_net_x6p': super_net_x6p,
        'super_net_x4p': super_net_x4p,
        'super_net_x2p': super_net_x2p,
    }
    return model_dict[model_name]()
