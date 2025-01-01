import sys

import torch
import torch.nn as nn

from .resnet1d import resnet32x4

# 使用 resenet1d 作为 backbone
class NetworkModel(nn.Module):
    def __init__(self):
        super(NetworkModel, self).__init__()

        # num
        self.hero_num = 3

        print("model file: ", __name__)
        sys.stdout.flush()

        # build network
        # image like feature
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 32, 5, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 10, 3, 1, 0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )  # [-1, 10, 11, 11]
        self.feature_proj = nn.Linear(121, 128)
        self.hero_frd_mlp = nn.Sequential(
            nn.Linear(251, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.hero_emy_mlp = nn.Sequential(
            nn.Linear(251, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.public_info_mlp = nn.Sequential(
            nn.Linear(44, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.soldier_frd_mlp = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.soldier_emy_mlp = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.organ_frd_mlp = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.organ_emy_mlp = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.monster_mlp = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(68, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.backbone = resnet32x4(num_classes=512)
        self.action_heads = nn.ModuleList()
        for action_dim in (13, 25, 42, 42, 39):
            self.action_heads.append(
                nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim)
                )
            )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hero_state):
        # hero_state (3, bs, 4586)
        all_hero_result_list = []
        hero_public_first_result_list = []
        hero_public_second_result_list = []
        # all_hero_embedding_list = []

        for hero_index in range(3):
            hero_feature = hero_state[0]

            feature_img, hero_frd, hero_emy, public_info, soldier_frd, soldier_emy, organ_frd, organ_emy, monster_vec, global_info = hero_feature.split(
                [17 * 17 * 6, 251 * 3, 251 * 3, 44, 25 * 10, 25 * 10, 29 * 3, 29 * 3, 28 * 20, 68], dim=1)
            feature_img = feature_img.reshape((-1, 6, 17, 17))

            conv_hidden = self.conv_layers(feature_img)  # B*12*4*4
            feature_map = self.feature_proj(conv_hidden.flatten(start_dim=2))
            # 改成reshape
            hero_frd_list = hero_frd.reshape((-1, 3, 251))
            hero_emy_list = hero_emy.reshape((-1, 3, 251))
            soldier_frd_list = soldier_frd.reshape((-1, 10, 25))
            soldier_emy_list = soldier_emy.reshape((-1, 10, 25))
            organ_frd_list = organ_frd.reshape((-1, 3, 29))
            organ_emy_list = organ_emy.reshape((-1, 3, 29))
            monster_list = monster_vec.reshape((-1, 20, 28))

            hero_frd_hidden = self.hero_frd_mlp(hero_frd_list)
            hero_emy_hidden = self.hero_emy_mlp(hero_emy_list)
            public_info_hidden = self.public_info_mlp(public_info)
            public_info_hidden = public_info_hidden.unsqueeze(1)
            soldier_frd_hidden = self.soldier_frd_mlp(soldier_frd_list)
            soldier_emy_hidden = self.soldier_emy_mlp(soldier_emy_list)
            organ_frd_hidden = self.organ_frd_mlp(organ_frd_list)
            organ_emy_hidden = self.organ_emy_mlp(organ_emy_list)
            monster_hidden = self.monster_mlp(monster_list)
            global_hidden = self.global_mlp(global_info)
            global_hidden = global_hidden.unsqueeze(1)

            input_embedding = torch.cat(
                [feature_map, hero_frd_hidden, hero_emy_hidden,
                 public_info_hidden, soldier_frd_hidden,
                 soldier_emy_hidden, organ_frd_hidden,
                 organ_emy_hidden, monster_hidden,
                 global_hidden], dim=1)
            output_embedding = self.backbone(input_embedding)
            output_embedding_split = output_embedding.split((128, 384), dim=1)
            hero_public_first_result_list.append(output_embedding_split[0])
            hero_public_second_result_list.append(output_embedding_split[1])

        pool_hero_public, _ = torch.stack(hero_public_first_result_list, dim=1).max(dim=1)

        for hero_index in range(self.hero_num):
            hero_result_list = []
            fc_public_result = torch.cat([pool_hero_public, hero_public_second_result_list[hero_index]], dim=1)
            # 5 action head
            for action_head in self.action_heads:
                hero_result_list.append(action_head(fc_public_result))

            all_hero_result_list.append(torch.cat(hero_result_list, dim=-1).unsqueeze(1))

        return torch.cat(all_hero_result_list, dim=1)

if __name__ == '__main__':
    net_torch = NetworkModel()
    print(net_torch)
