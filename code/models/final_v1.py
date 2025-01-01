import sys

import torch
import torch.nn as nn


# 比赛最后使用的版本
class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()

        # num
        self.hero_num = 3

        print("model file: ", __name__)
        sys.stdout.flush()

        self.out_dim = 128

        # build network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 18, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(18, 12, 3, 1, 1),
            nn.MaxPool2d(2)
        )  # 192
        # self.img_mlp = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512)
        # )
        self.hero_share_mlp = nn.Sequential(
            nn.Linear(251, 256),
            nn.ReLU(),
        )
        self.hero_frd_mlp = nn.Linear(256, self.out_dim)
        self.hero_emy_mlp = nn.Linear(256, self.out_dim)
        self.public_info_mlp = nn.Sequential(
            nn.Linear(44, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )
        self.soldier_share_mlp = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
        )
        self.soldier_frd_mlp = nn.Linear(128, self.out_dim)
        self.soldier_emy_mlp = nn.Linear(128, self.out_dim)
        self.organ_share_mlp = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
        )
        self.organ_frd_mlp = nn.Linear(128, self.out_dim)
        self.organ_emy_mlp = nn.Linear(128, self.out_dim)
        self.monster_mlp = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(68, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )
        # 192+128*9=1344
        self.concat_mlp = nn.Sequential(
            nn.Linear(1344, 2048),
            nn.ReLU(),
            nn.Linear(2048, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.action_heads = nn.ModuleList()
        for action_dim in (13, 25, 42, 42, 39):
            self.action_heads.append(
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, action_dim)
                )
            )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hero_state):
        all_hero_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []

        for hero_index in range(3):
            hero_feature = hero_state[hero_index]

            feature_img, hero_frd, hero_emy, public_info, soldier_frd,\
            soldier_emy, organ_frd, organ_emy, monster_vec, global_info = \
                hero_feature.split(
                    [17 * 17 * 6, 251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68],
                    dim=1)
            feature_img = feature_img.reshape((-1, 6, 17, 17))

            conv_hidden = self.conv_layers(feature_img).flatten(start_dim=1)  # B*12*4*4
            # 改成reshape
            hero_frd_list = hero_frd.reshape((-1, 3, 251))
            hero_emy_list = hero_emy.reshape((-1, 3, 251))
            soldier_frd_list = soldier_frd.reshape((-1, 10, 25))
            soldier_emy_list = soldier_emy.reshape((-1, 10, 25))
            organ_frd_list = organ_frd.reshape((-1, 3, 29))
            organ_emy_list = organ_emy.reshape((-1, 3, 29))
            monster_list = monster_vec.reshape((-1, 20, 28))

            hero_frd_hidden = self.hero_frd_mlp(self.hero_share_mlp(hero_frd_list))
            hero_frd_hidden_pool, _ = hero_frd_hidden.max(dim=1)
            hero_emy_hidden = self.hero_emy_mlp(self.hero_share_mlp(hero_emy_list))
            hero_emy_hidden_pool, _ = hero_emy_hidden.max(dim=1)
            public_info_hidden = self.public_info_mlp(public_info)
            monster_hidden = self.monster_mlp(monster_list)
            monster_hidden_pool, _ = monster_hidden.max(dim=1)
            soldier_frd_hidden = self.soldier_frd_mlp(self.soldier_share_mlp(soldier_frd_list))
            soldier_frd_hidden_pool, _ = soldier_frd_hidden.max(dim=1)
            soldier_emy_hidden = self.soldier_emy_mlp(self.soldier_share_mlp(soldier_emy_list))
            soldier_emy_hidden_pool, _ = soldier_emy_hidden.max(dim=1)
            organ_frd_hidden = self.organ_frd_mlp(self.organ_share_mlp(organ_frd_list))
            organ_frd_hidden_pool, _ = organ_frd_hidden.max(dim=1)
            organ_emy_hidden = self.organ_emy_mlp(self.organ_share_mlp(organ_emy_list))
            organ_emy_hidden_pool, _ = organ_emy_hidden.max(dim=1)
            global_hidden = self.global_mlp(global_info)

            concat_hidden = torch.cat(
                [conv_hidden, hero_frd_hidden_pool, hero_emy_hidden_pool,
                 public_info_hidden, soldier_frd_hidden_pool,
                 soldier_emy_hidden_pool, organ_frd_hidden_pool,
                 organ_emy_hidden_pool, monster_hidden_pool,
                 global_hidden], dim=1)  # 192+128*9=1344
            concat_hidden = self.concat_mlp(concat_hidden)

            concat_hidden_split = concat_hidden.split((128, 384), dim=1)
            hero_public_first_result_list.append(concat_hidden_split[0])
            hero_public_second_result_list.append(concat_hidden_split[1])

        pool_hero_public, _ = torch.stack(hero_public_first_result_list, dim=1).max(dim=1)

        for hero_index in range(self.hero_num):
            hero_result_list = []
            fc_public_result = torch.cat([pool_hero_public, hero_public_second_result_list[hero_index]], dim=1)
            # 5 action head
            for action_head in self.action_heads:
                hero_result_list.append(action_head(fc_public_result))

            all_hero_result_list.append(torch.cat(hero_result_list, dim=-1).unsqueeze(0))

        return torch.cat(all_hero_result_list, dim=0)


if __name__ == '__main__':
    net_torch = NetworkModel()
    print(net_torch)
