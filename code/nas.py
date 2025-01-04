import random

from torch import nn

from helper.util import make_mlp_layer, make_action_heads, make_linear_layer
from models.super_net import NetworkModel


def random_correction(a, b):
    return random.randint(1, int(b / 32)) * 32
    # return random.randint(a, b)


def generate_config():
    search_space = {
        'token_dim': [32, 224],
        'target_attn_dim': [32, 128],
        'img_mlp': {"n1": [32, 1024], "n2": [32, 512]},
        'hero_frd_mlp': {"n1": [32, 1024], "n2": [32, 1024]},
        'hero_emy_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'public_info_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'soldier_frd_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'soldier_emy_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'organ_frd_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'organ_emy_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'monster_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'global_mlp': {"n1": [32, 256], "n2": [32, 256]},
        'concat_mlp': {"n1": [32, 4096], "n2": [32, 2048], "n3": [32, 1024], "n4": [32, 1024], "n5": [32, 512]},
        'communicate_mlp': {"n1": [32, 1024], "n2": [32, 1024], "n3": [32, 512]},
        'action_heads': {"n1": [32, 256]},
        'target': {"n1": [32, 256]},
    }
    return {
        'token_dim': random_correction(32, 224),
        'target_attn_dim': random_correction(32, 128),
        'img_mlp': {"n1": random_correction(32, 1024), "n2": random_correction(32, 512), "l": random.randint(1, 2)},
        'hero_frd_mlp': {"n1": random_correction(32, 1024), "n2": random_correction(32, 1024),
                         "l": random.randint(1, 2)},
        'hero_emy_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256), "l": random.randint(1, 2)},
        'public_info_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256),
                            "l": random.randint(1, 2)},
        'soldier_frd_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256),
                            "l": random.randint(1, 2)},
        'soldier_emy_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256),
                            "l": random.randint(1, 2)},
        'organ_frd_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256),
                          "l": random.randint(1, 2)},
        'organ_emy_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256),
                          "l": random.randint(1, 2)},
        'monster_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256), "l": random.randint(1, 2)},
        'global_mlp': {"n1": random_correction(32, 256), "n2": random_correction(32, 256), "l": random.randint(1, 2)},
        'concat_mlp': {"n1": random_correction(32, 4096), "n2": random_correction(32, 2048),
                       "n3": random_correction(32, 1024), "n4": random_correction(32, 1024),
                       "n5": random_correction(32, 512), "l": random.randint(1, 5)},
        'communicate_mlp': {"n1": random_correction(32, 1024), "n2": random_correction(32, 1024),
                            "n3": random_correction(32, 512), "l": random.randint(0, 3)},
        'action_heads': {"n1": random_correction(32, 256)},
        'target_head': {"n1": random_correction(32, 256), "l": random.randint(1, 2)},
    }


def clip_layers(dims, num_l):
    dims[num_l] = dims[-1]
    return dims[:num_l + 1]


def make_net(config):
    token_dim = config['token_dim']
    target_attn_dim = config['target_attn_dim']
    concrete_config = {
        'token_dim': token_dim,
        'target_attn_dim': target_attn_dim,
        'img_mlp': clip_layers([768, config['img_mlp']['n1'], config['img_mlp']['n2']], config['img_mlp']['l']),
        'hero_frd_mlp': clip_layers([251, config['hero_frd_mlp']['n1'], config['hero_frd_mlp']['n2'], token_dim],
                                    config['img_mlp']['l']),  # 我方英雄
        'hero_emy_mlp': clip_layers([251, config['hero_emy_mlp']['n1'], config['hero_emy_mlp']['n2'], token_dim],
                                    config['hero_frd_mlp']['l']),  # 敌方英雄
        'public_info_mlp': clip_layers(
            [44, config['public_info_mlp']['n1'], config['public_info_mlp']['n2'], token_dim],
            config['public_info_mlp']['l']),  # 主英雄
        'soldier_frd_mlp': clip_layers(
            [25, config['soldier_frd_mlp']['n1'], config['soldier_frd_mlp']['n2'], token_dim],
            config['soldier_frd_mlp']['l']),  # 我方小兵
        'soldier_emy_mlp': clip_layers(
            [25, config['soldier_emy_mlp']['n1'], config['soldier_emy_mlp']['n2'], token_dim],
            config['soldier_emy_mlp']['l']),  # 敌方小兵
        'organ_frd_mlp': clip_layers([29, config['organ_frd_mlp']['n1'], config['organ_frd_mlp']['n2'], token_dim],
                                     config['organ_frd_mlp']['l']),  # 我方防御塔
        'organ_emy_mlp': clip_layers([29, config['organ_emy_mlp']['n1'], config['organ_emy_mlp']['n2'], token_dim],
                                     config['organ_emy_mlp']['l']),  # 敌方防御塔
        'monster_mlp': clip_layers([28, config['monster_mlp']['n1'], config['monster_mlp']['n2'], token_dim],
                                   config['monster_mlp']['l']),  # 野怪
        'global_mlp': clip_layers([68, config['global_mlp']['n1'], config['global_mlp']['n2'], token_dim],
                                  config['global_mlp']['l']),  # 计分板信息
    }
    concrete_config['concat_mlp'] = clip_layers(
        [token_dim * 9 + concrete_config['img_mlp'][-1], config['concat_mlp']['n1'], config['concat_mlp']['n2'],
         config['concat_mlp']['n3'], config['concat_mlp']['n4'], config['concat_mlp']['n5']], config['concat_mlp']['l'])
    concrete_config['communicate_mlp'] = clip_layers(
        [concrete_config['concat_mlp'][-1], config['communicate_mlp']['n1'], config['communicate_mlp']['n2'],
         config['communicate_mlp']['n3']], config['communicate_mlp']['l'])
    concrete_config['action_heads'] = [concrete_config['communicate_mlp'][-1], config['action_heads']['n1']]
    concrete_config['target_head'] = clip_layers(
        [concrete_config['communicate_mlp'][-1], config['target_head']['n1'], target_attn_dim], config['target_head']['l'])

    arch_config = {
        'token_dim': token_dim,
        'target_attn_dim': target_attn_dim,
        # 特征处理
        # image like feature
        'conv_layers': nn.Sequential(
            nn.Conv2d(6, 18, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(18, 12, 3, 1, 1),
        ),  # (12, 8, 8)
        'img_mlp': make_mlp_layer(concrete_config['img_mlp']),
        'hero_frd_mlp': make_mlp_layer(concrete_config['hero_frd_mlp']),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer(concrete_config['hero_emy_mlp']),  # 敌方英雄
        'public_info_mlp': make_mlp_layer(concrete_config['public_info_mlp']),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer(concrete_config['soldier_frd_mlp']),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer(concrete_config['soldier_emy_mlp']),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer(concrete_config['organ_frd_mlp']),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer(concrete_config['organ_emy_mlp']),  # 敌方防御塔
        'monster_mlp': make_mlp_layer(concrete_config['monster_mlp']),  # 野怪
        'global_mlp': make_mlp_layer(concrete_config['global_mlp']),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer(concrete_config['concat_mlp']),
        'communicate_mlp': make_mlp_layer(concrete_config['communicate_mlp']),
        # 4 action head
        'action_heads': make_action_heads(concrete_config['action_heads'], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer(concrete_config['target_head']),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


# make_net(generate_config())


for i in range(1):
    print(make_net(generate_config()))