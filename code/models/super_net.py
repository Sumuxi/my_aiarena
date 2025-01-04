import logging

import torch
import torch.nn as nn

from helper.util import make_mlp_layer, make_linear_layer, make_action_heads


class NetworkModel(nn.Module):
    def __init__(self, arch_config):
        super(NetworkModel, self).__init__()

        for k, v in arch_config.items():
            setattr(self, k, v)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, heroes_data):
        all_hero_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []
        all_hero_target_list = []

        for hero_index in range(3):
            hero_feature = heroes_data[hero_index]

            (
                feature_img, hero_frd, hero_emy,
                public_info, soldier_frd, soldier_emy,
                organ_frd, organ_emy,
                monster_vec, global_info
            ) = hero_feature.split(
                [17 * 17 * 6, 251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68], dim=1)
            feature_img = feature_img.reshape((-1, 6, 17, 17))

            conv_hidden = self.conv_layers(feature_img).flatten(start_dim=1)  # bs*12*8*8
            img_hidden = self.img_mlp(conv_hidden)
            # 改成reshape
            hero_frd_list = hero_frd.reshape((-1, 3, 251))
            hero_emy_list = hero_emy.reshape((-1, 3, 251))
            soldier_frd_list = soldier_frd.reshape((-1, 10, 25))
            soldier_emy_list = soldier_emy.reshape((-1, 10, 25))
            organ_frd_list = organ_frd.reshape((-1, 3, 29))
            organ_emy_list = organ_emy.reshape((-1, 3, 29))
            monster_list = monster_vec.reshape((-1, 20, 28))

            hero_target_list = []

            hero_frd_hidden = self.hero_frd_mlp(hero_frd_list)
            hero_target_list.append(hero_frd_hidden)  # 3 frd hero
            hero_frd_hidden_pool, _ = hero_frd_hidden.max(dim=1)
            hero_emy_hidden = self.hero_emy_mlp(hero_emy_list)
            hero_target_list.append(hero_emy_hidden)  # 3 emy hero
            hero_emy_hidden_pool, _ = hero_emy_hidden.max(dim=1)
            public_info_hidden = self.public_info_mlp(public_info)
            hero_target_list.append(public_info_hidden.reshape((-1, 1, self.token_dim)))  # 1 public info
            monster_hidden = self.monster_mlp(monster_list)
            monster_hidden_pool, _ = monster_hidden.max(dim=1)
            hero_target_list.append(monster_hidden)  # 20 monster
            soldier_frd_hidden = self.soldier_frd_mlp(soldier_frd_list)
            soldier_frd_hidden_pool, _ = soldier_frd_hidden.max(dim=1)
            soldier_emy_hidden = self.soldier_emy_mlp(soldier_emy_list)
            soldier_emy_hidden_pool, _ = soldier_emy_hidden.max(dim=1)
            hero_target_list.append(soldier_emy_hidden)  # 10 emy soldier
            organ_frd_hidden = self.organ_frd_mlp(organ_frd_list)
            organ_frd_hidden_pool, _ = organ_frd_hidden.max(dim=1)
            organ_emy_hidden = self.organ_emy_mlp(organ_emy_list)
            organ_emy_hidden_pool, _ = organ_emy_hidden.max(dim=1)
            global_hidden = self.global_mlp(global_info)
            hero_target_list.append(organ_emy_hidden_pool.reshape((-1, 1, self.token_dim)))  # 1 emy organ
            hero_target_list.insert(0, torch.ones_like(hero_target_list[2], dtype=torch.float32) * 0.1)
            all_hero_target_list.append(torch.cat(hero_target_list, dim=1))

            concat_hidden = torch.cat(
                [img_hidden, hero_frd_hidden_pool, hero_emy_hidden_pool,
                 public_info_hidden, soldier_frd_hidden_pool,
                 soldier_emy_hidden_pool, organ_frd_hidden_pool,
                 organ_emy_hidden_pool, monster_hidden_pool,
                 global_hidden], dim=1)
            concat_hidden = self.concat_mlp(concat_hidden)

            first_size = concat_hidden.size(1) // 4
            second_size = concat_hidden.size(1) // 4 * 3
            concat_hidden_split = concat_hidden.split((first_size, second_size), dim=1)
            hero_public_first_result_list.append(concat_hidden_split[0])
            hero_public_second_result_list.append(concat_hidden_split[1])

        pool_hero_public, _ = torch.stack(hero_public_first_result_list, dim=1).max(dim=1)

        for hero_index in range(3):
            hero_result_list = []
            fc_public_result = torch.cat([pool_hero_public, hero_public_second_result_list[hero_index]], dim=1)
            if hasattr(self, 'communicate_mlp'):
                communication_result = self.communicate_mlp(fc_public_result)
            else:
                communication_result = fc_public_result
            # 4 action head
            for action_head in self.action_heads:
                hero_result_list.append(action_head(communication_result))
            # target head
            target_embedding = self.target_embed(all_hero_target_list[hero_index])  # bs*39*target_attn_dim
            target_key = self.target_head(communication_result).reshape(
                (-1, self.target_attn_dim, 1))  # bs*target_attn_dim*1
            target_logits = torch.matmul(target_embedding, target_key).reshape((-1, 39))  # bs*39
            hero_result_list.append(target_logits)

            all_hero_result_list.append(torch.cat(hero_result_list, dim=-1).unsqueeze(0))

        return torch.cat(all_hero_result_list, dim=0)


def make_super_net():
    token_dim = 224
    target_attn_dim = 128
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
        'img_mlp': make_mlp_layer([768, 1024, 512]),
        'hero_frd_mlp': make_mlp_layer([251, 1024, 1024, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 1024, 1024, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 256, 256, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 256, 256, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 256, 256, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 256, 256, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 256, 256, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 256, 256, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 256, 256, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([2528, 4096, 2048, 1024, 1024, 512]),
        'communicate_mlp': make_mlp_layer([512, 1024, 1024, 512]),
        # 4 action head
        'action_heads': make_action_heads([512, 256], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([512, 256, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


def super_net_x30p():
    token_dim = 224
    target_attn_dim = 128
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
        'img_mlp': make_mlp_layer([768, 768, 512]),
        'hero_frd_mlp': make_mlp_layer([251, 1024, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 1024, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 256, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 256, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 256, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 256, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 256, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 256, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 256, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([512 + token_dim * 9, 4096, 2048, 1024, 1024, 512]),
        'communicate_mlp': make_mlp_layer([512, 1024, 1024, 512]),
        # 4 action head
        'action_heads': make_action_heads([512, 256], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([512, 256, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


def super_net_x20p():
    token_dim = 128
    target_attn_dim = 128
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
        'img_mlp': make_mlp_layer([768, 512]),
        'hero_frd_mlp': make_mlp_layer([251, 512, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 512, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 256, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 256, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 256, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 256, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 256, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 256, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 256, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([512 + token_dim * 9, 4096, 1300, 1024, 1024, 512]),
        'communicate_mlp': make_mlp_layer([512, 1024, 1024, 512]),
        # 4 action head
        'action_heads': make_action_heads([512, 256], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([512, 256, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


def super_net_x10p():
    token_dim = 128
    target_attn_dim = 128
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
        'img_mlp': make_mlp_layer([768, 256]),
        'hero_frd_mlp': make_mlp_layer([251, 128, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 128, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 128, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 128, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 128, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 128, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 128, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 128, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 128, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([256 + token_dim * 9, 3072, 1024, 512]),
        # 'communicate_mlp': make_mlp_layer([512, 512]),
        # 4 action head
        'action_heads': make_action_heads([512, 256], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([512, 256, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


def super_net_x8p():
    token_dim = 128
    target_attn_dim = 128
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
        'img_mlp': make_mlp_layer([768, 256]),
        'hero_frd_mlp': make_mlp_layer([251, 128, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 128, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 128, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 128, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 128, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 128, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 128, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 128, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 128, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([256 + token_dim * 9, 2300, 1024, 512]),
        # 'communicate_mlp': make_mlp_layer([512, 512]),
        # 4 action head
        'action_heads': make_action_heads([512, 128], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([512, 128, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


def super_net_x6p():
    token_dim = 96
    target_attn_dim = 96
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
        'img_mlp': make_mlp_layer([768, 256]),
        'hero_frd_mlp': make_mlp_layer([251, 128, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 128, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 128, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 128, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 128, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 128, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 128, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 128, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 128, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([256 + token_dim * 9, 2048, 1024, 256]),
        # 'communicate_mlp': make_mlp_layer([512, 512]),
        # 4 action head
        'action_heads': make_action_heads([256, 128], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([256, 128, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


def super_net_x4p():
    token_dim = 96
    target_attn_dim = 96
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
        'img_mlp': make_mlp_layer([768, 128]),
        'hero_frd_mlp': make_mlp_layer([251, 128, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 128, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 128, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 128, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 128, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 128, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 128, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 128, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 128, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([128 + token_dim * 9, 1536, 512, 256]),
        # 'communicate_mlp': make_mlp_layer([512, 512]),
        # 4 action head
        'action_heads': make_action_heads([256, 128], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([256, 128, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


def super_net_x2p():
    token_dim = 64
    target_attn_dim = 64
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
        'img_mlp': make_mlp_layer([768, 128]),
        'hero_frd_mlp': make_mlp_layer([251, 128, token_dim]),  # 我方英雄
        'hero_emy_mlp': make_mlp_layer([251, 128, token_dim]),  # 敌方英雄
        'public_info_mlp': make_mlp_layer([44, 96, token_dim]),  # 主英雄
        'soldier_frd_mlp': make_mlp_layer([25, 96, token_dim]),  # 我方小兵
        'soldier_emy_mlp': make_mlp_layer([25, 96, token_dim]),  # 敌方小兵
        'organ_frd_mlp': make_mlp_layer([29, 96, token_dim]),  # 我方防御塔
        'organ_emy_mlp': make_mlp_layer([29, 96, token_dim]),  # 敌方防御塔
        'monster_mlp': make_mlp_layer([28, 96, token_dim]),  # 野怪
        'global_mlp': make_mlp_layer([68, 96, token_dim]),  # 计分板信息
        # 拼接特征，联合处理
        # 512 + token_dim*9
        'concat_mlp': make_mlp_layer([128 + token_dim * 9, 800, 256]),
        # 'communicate_mlp': make_mlp_layer([512, 512]),
        # 4 action head
        'action_heads': make_action_heads([256, 96], (13, 25, 42, 42)),
        # target head
        'target_head': make_mlp_layer([256, 96, target_attn_dim]),
        'target_embed': make_linear_layer(token_dim, target_attn_dim),
    }
    return NetworkModel(arch_config)


if __name__ == '__main__':
    net_torch = NetworkModel()
    print(net_torch)
