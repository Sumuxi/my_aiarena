import sys
from ctypes import Union
from math import floor
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.Config import Config
from helper.train_config import sub_action_mask_list

from helper.util import make_mlp_layer

"""
基于baseline
用不同规模的 MLP 来代替 encoder(连续使用3次transformer)
"""
class NetworkModel(nn.Module):
    def __init__(self, kwargs):
        super(NetworkModel, self).__init__()
        # lstm
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_size = Config.LSTM_UNIT_SIZE

        # data
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE
        self.hero_seri_vec_split_shape = Config.HERO_SERI_VEC_SPLIT_SHAPE
        self.hero_feature_img_channel = Config.HERO_FEATURE_IMG_CHANNEL
        self.hero_label_size_list = Config.HERO_LABEL_SIZE_LIST
        self.hero_is_reinforce_task_list = Config.HERO_IS_REINFORCE_TASK_LIST

        # loss
        self.learning_rate = Config.INIT_LEARNING_RATE_START
        self.var_beta = Config.BETA_START
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.min_policy = Config.MIN_POLICY
        self.embedding_trainable = False
        self.value_head_num = Config.VALUE_HEAD_NUM

        # value
        self.value_head_num = Config.VALUE_HEAD_NUM
        self.hero_policy_weight = Config.HERO_POLICY_WEIGHT

        # target attention
        self.target_embedding_dim = Config.TARGET_EMBEDDING_DIM
        # num
        self.hero_num = 3
        self.hero_data_len = sum(Config.data_shapes[0])

        self.distill_temperature = Config.DISTILL_TEMPERATURE
        self.distill_weight = Config.DISTILL_WEIGHT

        print("model file: ", __name__)
        # loss weights
        coefficients = {
            "hard loss weight": Config.HARD_WEIGHT,
            "soft loss weight": Config.SOFT_WEIGHT,
            "distill loss weight": self.distill_weight,
            "distill temperature": self.distill_temperature
        }
        for k, v in coefficients.items():
            print(f"{k}: {v}")
        sys.stdout.flush()

        # build network
        kernel_size_list = [(5, 5), (3, 3)]
        padding_list = ["same", "same"]
        # img conv
        self.img_channel = self.hero_feature_img_channel[0][0]

        # channel_list = [self.hero_feature_img_channel[0][0], 18, 12]
        channel_list = [self.img_channel, 18, self.img_channel]

        assert len(channel_list) == len(kernel_size_list) + 1, "channel list and kernel size list length mismatch"
        assert len(kernel_size_list) == len(padding_list), "kernel size list and padding list length mismatch"

        '''img_conv module'''
        self.conv_layers = nn.Sequential()
        for i, kernel_size in enumerate(kernel_size_list):
            is_last_layer = (i == len(kernel_size_list) - 1)
            conv_layer, _ = make_conv_layer(kernel_size, channel_list[i], channel_list[i + 1], padding_list[i])
            self.conv_layers.add_module("img_feat_conv{0}".format(i + 1), conv_layer)

            if not is_last_layer:
                self.conv_layers.add_module("img_feat_relu{0}".format(i + 1), nn.ReLU())
                self.conv_layers.add_module("img_feat_maxpool{0}".format(i + 1), nn.MaxPool2d(3, 2))
        '''share'''
        self.fc1_img = make_fc_layer(8 * 8, Config.TOKEN_DIM)

        ''' hero_share module'''
        fc_hero_dim_list = [Config.HERO_DIM, 1024, 1024, Config.TOKEN_DIM]
        self.hero_mlp = MLP(fc_hero_dim_list, "hero_mlp")

        ''' hero_main module'''
        fc_hero_main_dim_list = [Config.MAIN_HERO_DIM, 256, 256, Config.TOKEN_DIM]
        self.hero_main_mlp = MLP(fc_hero_main_dim_list, "hero_main_mlp")

        ''' monster_share module'''
        fc_monster_dim_list = [Config.MONSTER_DIM, 256, 256, Config.TOKEN_DIM]
        self.monster_mlp = MLP(fc_monster_dim_list, "monster_mlp")

        ''' soldier_share module'''
        ## first and second fc layers are shared by 2 soldier vecs
        fc_soldier_dim_list = [Config.SOLDIER_DIM, 256, 256, Config.TOKEN_DIM]
        self.soldier_mlp = MLP(fc_soldier_dim_list, "soldier_mlp")

        ''' organ_share module'''
        fc_organ_dim_list = [Config.ORGAN_DIM, 256, 256, Config.TOKEN_DIM]
        self.organ_mlp = MLP(fc_organ_dim_list, "organ_mlp")

        '''global'''
        fc_global_dim_list = [Config.GLOBAL_DIM, 256, 256, Config.TOKEN_DIM]
        self.global_mlp = MLP(fc_global_dim_list, "global_mlp")

        ### encoder - decoder
        # self.encoder = nn.ModuleList()

        # for i in range(Config.ATT_LAYER_NUM):
        #     mha = MultiHeadAttention(n_head=Config.ATT_HEAD_NUM, d_model=Config.TOKEN_DIM,
        #                              d_k=Config.HEAD_DIM, d_v=Config.HEAD_DIM,
        #                              hidden_dim=Config.TOKEN_DIM * Config.ATT_HEAD_NUM)
        #     self.encoder.append(mha)

        self.concat_mlp = nn.Sequential(
            nn.Linear(19 * 224, 2048),
            nn.ELU(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 224),
            nn.ELU()
        )

        self.fc1_public = MLP([Config.TOKEN_DIM, self.lstm_unit_size], "fc1_public",
                              non_linearity_last=True)  # add relu

        self.fc1_label = make_fc_layer(self.lstm_unit_size, Config.TOKEN_DIM)

        self.fc2_label_list = []

        for label_index in range(len(self.hero_label_size_list[0]) - 1):
            fc_layer = make_fc_layer(Config.TOKEN_DIM, self.hero_label_size_list[0][label_index])
            self.fc2_label_list.append(fc_layer)

        self.fc2_label_list = nn.ModuleList(self.fc2_label_list)

        # target attention
        self.fc1_target = make_fc_layer(self.lstm_unit_size, Config.TOKEN_DIM)
        self.fc2_target = make_fc_layer(Config.TOKEN_DIM, self.target_embedding_dim)

        self.fc1_target_token = make_fc_layer(Config.TOKEN_DIM, self.target_embedding_dim)
        self.fc2_target_token = make_fc_layer(self.target_embedding_dim, self.target_embedding_dim,
                                              use_bias=False)

        # hero value
        self.fc1_value = make_fc_layer(self.lstm_unit_size, Config.TOKEN_DIM)
        self.fc2_value = make_fc_layer(Config.TOKEN_DIM, self.value_head_num)

        self.lstm = torch.nn.LSTM(input_size=self.lstm_unit_size, hidden_size=self.lstm_unit_size, num_layers=1,
                                  bias=True, batch_first=False, dropout=0, bidirectional=False)
        # self.lstm = LSTMCell(input_size=self.lstm_unit_size, hidden_size=self.lstm_unit_size, batch_first=False,
        #                      forget_bias=1.)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, data_list):
        each_hero_data_list = data_list

        all_hero_target_token_list = []
        all_hero_token_list = []
        all_hero_main_token_list = []
        temp_lstm_cell_list = []
        temp_lstm_hidden_list = []

        for hero_index, hero_data in enumerate(data_list):
            hero_feature = hero_data[0]  # ([512, 4586])  实际batch_size = 32，此处 bs = 32 * 16帧 = 512

            img_fet_dim = np.prod(self.hero_seri_vec_split_shape[hero_index][0])  # 6 x 17 x 17 = 1734
            vec_fet_dim = np.prod(self.hero_seri_vec_split_shape[hero_index][1])  # 2852
            # ([bs, 1734]), ([bs, 2852])
            feature_img, feature_vec = hero_feature.split([img_fet_dim, vec_fet_dim], dim=1)  # (bs, tot_fea_dim)

            feature_img_shape = list(self.hero_seri_vec_split_shape[0][0])  # (c, h, w)
            feature_img_shape.insert(0, -1)  # (bs, c, h, w)
            feature_vec_shape = list(self.hero_seri_vec_split_shape[0][1])
            feature_vec_shape.insert(0, -1)

            _feature_img = feature_img.reshape(feature_img_shape)  # ([bs, 6, 17, 17])
            _feature_vec = feature_vec.reshape(feature_vec_shape)  # ([bs, 2852])

            lstm_index = len(hero_data)
            temp_lstm_cell_list.append(hero_data[lstm_index - 2])
            temp_lstm_hidden_list.append(hero_data[lstm_index - 1])

            hero_dim = int(Config.HERO_NUM * Config.HERO_DIM * Config.CAMP_NUM)
            main_dim = int(Config.MAIN_HERO_DIM)
            # pos_dim = int(Config.POS_DIM)
            soldier_dim = int(Config.SOLDIER_NUM * Config.SOLDIER_DIM)
            organ_dim = int(Config.ORGAN_NUM * Config.ORGAN_DIM)
            monster_dim = int(Config.MONSTER_NUM * Config.MONSTER_DIM)
            global_info_dim = int(Config.GLOBAL_DIM)

            split_feature_vec = _feature_vec.split([
                hero_dim, main_dim, soldier_dim, soldier_dim,
                organ_dim, organ_dim, monster_dim, global_info_dim,
            ], dim=1)  # [1506, 44, 250, 250, 87, 87, 560, 68]

            hero_tensor = split_feature_vec[0].reshape(-1, Config.HERO_NUM * Config.CAMP_NUM,
                                                       Config.HERO_DIM)  # ([bs, 6, 251])
            main_tensor = split_feature_vec[1].reshape(-1, 1, Config.MAIN_HERO_DIM)  # ([bs, 1, 44])

            # pos_tensor = split_feature_vec[2]
            frd_soldier_tensor = split_feature_vec[2].reshape(-1, Config.SOLDIER_NUM, Config.SOLDIER_DIM)
            emy_soldier_tensor = split_feature_vec[3].reshape(-1, Config.SOLDIER_NUM, Config.SOLDIER_DIM)

            soldier_tensor_list = [frd_soldier_tensor, emy_soldier_tensor]  # ([bs, 10, 25]), ([bs, 10, 25])

            frd_organ_tensor = split_feature_vec[4].reshape(-1, Config.ORGAN_NUM, Config.ORGAN_DIM)
            emy_organ_tensor = split_feature_vec[5].reshape(-1, Config.ORGAN_NUM, Config.ORGAN_DIM)

            organ_tensor_list = [frd_organ_tensor, emy_organ_tensor]  # ([bs, 3, 29]), ([bs, 3, 29])

            monster_tensor = split_feature_vec[6].reshape(-1, Config.MONSTER_NUM, Config.MONSTER_DIM)  # ([bs, 20, 28])
            global_tensor = split_feature_vec[7].reshape(-1, 1, Config.GLOBAL_DIM)  # ([bs, 1, 68])

            this_hero_target_token_list = []
            this_hero_token_list = []

            conv2_result = self.conv_layers(_feature_img)  # (bs, 6, 8, 8)

            # fc
            dim = np.prod(conv2_result.shape[2:])

            conv2_result_reshape = conv2_result.reshape(-1, self.img_channel, dim)  # (bs, 6, 64)

            img_token = self.fc1_img(conv2_result_reshape)  # (bs, 6, 224)

            # [bs, img_channel, dim]
            this_hero_token_list.append(img_token)

            # hero
            in_fea = hero_tensor

            fc3_result = self.hero_mlp(in_fea)

            # [bs, n, dim]
            this_hero_target_token_list.append(fc3_result)
            this_hero_token_list.append(fc3_result)

            # main hero

            in_fea = main_tensor

            fc3_result = self.hero_main_mlp(in_fea)

            this_hero_target_token_list.append(fc3_result)
            this_hero_token_list.append(fc3_result)
            main_token = fc3_result

            # monster
            in_fea = monster_tensor

            fc3_result = self.monster_mlp(in_fea)

            # Note: this pooling should be checked
            pooling_result = self._pooling(fc3_result, Config.MONSTER_NUM, keep_dim=3, is_trt=True, pool_type='max')

            this_hero_target_token_list.append(fc3_result)
            this_hero_token_list.append(pooling_result)

            # soldier
            result_list = []
            for camp_index in range(Config.CAMP_NUM):
                in_fea = soldier_tensor_list[camp_index]
                # [?, 10, 256]
                fc3_result = self.soldier_mlp(in_fea)
                # [?, 1, 256]
                pooling_result = self._pooling(fc3_result, Config.SOLDIER_NUM, keep_dim=3, is_trt=True, pool_type='max')

                if camp_index == 1:
                    this_hero_target_token_list.append(fc3_result)
                this_hero_token_list.append(pooling_result)

            # organ
            result_list = []
            for camp_index in range(Config.CAMP_NUM):
                in_fea = organ_tensor_list[camp_index]
                # [?, 3, 256]
                fc3_result = self.organ_mlp(in_fea)
                # [?, 1, 256]
                pooling_result = self._pooling(fc3_result, Config.ORGAN_NUM, keep_dim=3, is_trt=True, pool_type='max')

                if camp_index == 1:
                    this_hero_target_token_list.append(pooling_result)  # single organ target
                this_hero_token_list.append(pooling_result)

            # global
            in_fea = global_tensor
            fc3_result = self.global_mlp(in_fea)
            this_hero_token_list.append(fc3_result)

            # non target
            # target token : add none token
            none_target = torch.ones_like(main_token, dtype=torch.float32) * 0.1
            this_hero_target_token_list.insert(0, none_target)

            this_hero_token_tensor = torch.cat(this_hero_token_list, dim=1)  # ([bs, 19, 224])
            this_hero_target_token_tensor = torch.cat(this_hero_target_token_list, axis=1)  # ([bs, 39, 224])

            # store

            all_hero_token_list.append(this_hero_token_tensor)
            all_hero_target_token_list.append(this_hero_target_token_tensor)
            all_hero_main_token_list.append(main_token)

        ### Encoder - Decoder ###
        all_hero_decode_token = []
        all_hero_public_list = []
        all_hero_private_list = []

        for hero_index in range(len(each_hero_data_list)):
            # encoder
            x = all_hero_token_list[hero_index]

            # for i in range(Config.ATT_LAYER_NUM):
            #     pass
            #     x = self.encoder[i](q=x, k=x, v=x)
            #
            # token_num = int(x.shape[1])
            # token_average = self._pooling(x, token_num, keep_dim=2, is_trt=True, pool_type='avg')

            x = self.concat_mlp(x.flatten(1))

            fc1_public = self.fc1_public(x)

            # communication feature
            dim = int(fc1_public.shape[-1])

            # store communication
            # dim = int(z.shape[-1])
            public_dim = dim // 4
            private_dim = dim - public_dim

            public_result, private_result = fc1_public.split([public_dim, private_dim], dim=1)

            all_hero_public_list.append(public_result)
            all_hero_private_list.append(private_result)

        ### Communication & LSTM ###
        all_hero_lstm_state = []

        for hero_index in range(len(each_hero_data_list)):
            # Communication
            all_hero_public_tensor = torch.stack(all_hero_public_list, dim=1)
            hero_public_pool = self._pooling(all_hero_public_tensor, Config.HERO_NUM, is_trt=True, pool_type='max')
            commu_fea = torch.cat([hero_public_pool, all_hero_private_list[hero_index]], dim=-1)

            # LSTM
            # cell
            in_lstm_cell = temp_lstm_cell_list[hero_index]
            # lstm_cell = in_lstm_cell.reshape(1, -1, self.lstm_unit_size)
            lstm_cell = in_lstm_cell.reshape(1, -1, self.lstm_unit_size).contiguous()

            # hidden
            in_lstm_hidden = temp_lstm_hidden_list[hero_index]
            # lstm_hidden = in_lstm_hidden.reshape(1, -1, self.lstm_unit_size)
            lstm_hidden = in_lstm_hidden.reshape(1, -1, self.lstm_unit_size).contiguous()

            public_lstm_hidden, public_lstm_cell = lstm_hidden, lstm_cell

            commu_fea_time_step = commu_fea.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)

            lstm_initial_state = (public_lstm_hidden, public_lstm_cell)

            commu_fea_time_step_transpose = commu_fea_time_step.permute(1, 0, 2)

            lstm_outputs, (hn, cn) = self.lstm(commu_fea_time_step_transpose, lstm_initial_state)
            lstm_outputs = lstm_outputs.permute(1, 0, 2)
            lstm_outputs = lstm_outputs.reshape(-1, self.lstm_unit_size)

            all_hero_lstm_state.append(lstm_outputs)

        ### Output Projection ###
        all_hero_predict_result_list = []

        for hero_index in range(len(each_hero_data_list)):
            this_hero_predict_result_list = []
            in_fea = all_hero_lstm_state[hero_index]
            # action

            for label_index in range(len(self.hero_label_size_list[hero_index]) - 1):
                # fc1_label_result = F.relu(self.fc1_label(fc2_memory))
                fc1_label_result = F.relu(self.fc1_label(in_fea))

                fc2_label_result = self.fc2_label_list[label_index](fc1_label_result)

                # store action result
                this_hero_predict_result_list.append(fc2_label_result)

            # target attention

            label_index = len(self.hero_label_size_list[hero_index]) - 1

            # fc1_query = F.relu(self.fc1_target(fc2_memory))
            fc1_query = F.relu(self.fc1_target(in_fea))

            fc2_query = self.fc2_target(fc1_query)
            fc2_query_reshape = fc2_query.reshape(-1, self.target_embedding_dim, 1)

            # key
            target_token = all_hero_target_token_list[hero_index]
            target_num = int(target_token.shape[1])

            target_token = target_token.reshape(-1, target_num, Config.TOKEN_DIM)

            fc1_key = F.relu(self.fc1_target_token(target_token))
            fc2_key = self.fc2_target_token(fc1_key)
            fc2_key_reshape = fc2_key.reshape(-1, target_num, self.target_embedding_dim)

            # att
            fc2_target_result = torch.matmul(fc2_key_reshape, fc2_query_reshape)

            dim = int(np.prod(fc2_target_result.shape[1:]))

            fc2_target_result = fc2_target_result.reshape(-1, dim)
            # store target result
            this_hero_predict_result_list.append(fc2_target_result)

            fc1_value = F.relu(self.fc1_value(in_fea))
            fc2_value = self.fc2_value(fc1_value)

            this_hero_predict_result_list.append(fc2_value)
            # store hero result
            all_hero_predict_result_list.append(this_hero_predict_result_list)

        # rst_list = all_hero_predict_result_list
        # total_loss, info_list = self.compute_loss(data_list, rst_list)  # for debug. merge loss compute
        # return total_loss, info_list

        return all_hero_predict_result_list  # 预测维度 ([13, 25, 42, 42, 39, 1])

    def _pooling(self, tensor, token_num, keep_dim=2, is_trt=False, pool_type='max'):
        dim = int(tensor[0].shape[-1])

        reshape_result = tensor.reshape(-1, 1, token_num, dim)
        if pool_type == 'max':
            pooling_layer = torch.nn.MaxPool2d(kernel_size=(token_num, 1), stride=1,
                                               padding=0)  # Note: should the max pooling layer be defined
        elif pool_type == 'avg':
            pooling_layer = torch.nn.AvgPool2d(kernel_size=(token_num, 1), stride=1,
                                               padding=0)  # Note: should the max pooling layer be defined
        pool_result = pooling_layer(reshape_result)

        if keep_dim == 2:
            reshape_pool_result = pool_result.reshape(-1, dim)
        elif keep_dim == 3:
            reshape_pool_result = pool_result.reshape(-1, 1, dim)

        return reshape_pool_result

    def _calculate_single_hero_hard_loss(self, unsqueeze_label_list, fc2_label_list, unsqueeze_weight_list):
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(torch.squeeze(ele, dim=1).long())
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()
            label_loss = F.cross_entropy(fc2_label_list[i], label_list[i], reduction='none')
            label_final_loss = torch.mean(weight * label_loss)
            cost_p_label_list.append(label_final_loss)
        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_hero_soft_loss(self, student_logits_list, teacher_logits_list, unsqueeze_weight_list):
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(student_logits_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()
            # Calculate soft label loss
            teacher_probs = F.softmax(teacher_logits_list[i], dim=1)
            soft_label_loss = F.cross_entropy(student_logits_list[i], teacher_probs, reduction='none')
            label_final_loss = torch.mean(weight * soft_label_loss)
            cost_p_label_list.append(label_final_loss)
        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_hero_distill_loss(self, unsqueeze_label_list, student_logits_list, teacher_logits_list,
                                            unsqueeze_weight_list, temperature=4.0, lambda_weight=0.5):
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(torch.squeeze(ele, dim=1).long())
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()

            # Calculate hard label loss
            hard_label_loss = F.cross_entropy(student_logits_list[i], label_list[i], reduction='none')
            hard_label_final_loss = torch.mean(weight * hard_label_loss)

            # Calculate soft label loss
            student_logits_temperature = student_logits_list[i] / temperature
            teacher_logits_temperature = teacher_logits_list[i] / temperature
            teacher_probs = F.softmax(teacher_logits_temperature, dim=1)

            soft_label_loss = F.cross_entropy(student_logits_temperature, teacher_probs, reduction='none')
            soft_label_final_loss = torch.mean(weight * soft_label_loss)

            # Combine the hard and soft label losses with the specified weight
            final_loss = (1 - lambda_weight) * hard_label_final_loss + (
                    temperature ** 2) * lambda_weight * soft_label_final_loss

            cost_p_label_list.append(final_loss)

        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_temperatured_soft_loss(self, unsqueeze_label_list, student_logits_list, teacher_logits_list,
                                                 unsqueeze_weight_list, temperature=4.0):
        label_list = []
        for ele in unsqueeze_label_list:
            label_list.append(torch.squeeze(ele, dim=1).long())
        weight_list = []
        for weight in unsqueeze_weight_list:
            weight_list.append(torch.squeeze(weight, dim=1))

        cost_p_label_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != torch.tensor(0, dtype=torch.float32)).float()

            # Calculate soft label loss
            student_logits_temperature = student_logits_list[i] / temperature
            teacher_logits_temperature = teacher_logits_list[i] / temperature
            teacher_probs = F.softmax(teacher_logits_temperature, dim=1)

            soft_label_loss = F.cross_entropy(student_logits_temperature, teacher_probs, reduction='none')
            soft_label_final_loss = torch.mean(weight * soft_label_loss)

            cost_p_label_list.append(soft_label_final_loss)

        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(cost_p_label_list)):
            loss = loss + cost_p_label_list[i]
        return loss, cost_p_label_list

    def _calculate_single_hero_kl_div_loss(self, unsqueeze_label_list, student_logits_list, teacher_logits_list,
                                           unsqueeze_weight_list, temperature=4.0):
        label_list = [torch.squeeze(ele, dim=1).long() for ele in unsqueeze_label_list]
        weight_list = [torch.squeeze(weight, dim=1) for weight in unsqueeze_weight_list]

        loss_list = []
        for i in range(len(label_list)):
            weight = (weight_list[i] != 0).float()

            student_logits_temperature = student_logits_list[i] / temperature
            teacher_logits_temperature = teacher_logits_list[i] / temperature

            # 使用 log_softmax, 避免指数运行带来的数值不稳定
            student_probs_log = F.log_softmax(student_logits_temperature, dim=1)
            teacher_probs_log = F.log_softmax(teacher_logits_temperature, dim=1)

            # 计算 KL 散度，log_target=True 表示教师模型的目标已是 log 概率
            kl_div_loss = F.kl_div(student_probs_log, teacher_probs_log, reduction='none', log_target=True)
            # 通过对类别维度进行求和来得到每个样本的损失
            kl_div_loss = kl_div_loss.sum(dim=1)  # 或使用 .mean(dim=1)

            soft_label_final_loss = torch.mean(weight * kl_div_loss)

            loss_list.append(soft_label_final_loss)

        loss = torch.sum(torch.stack(loss_list))
        return loss, loss_list

    def _calculate_single_hero_topk_kl_div_loss(self,
                                                unsqueeze_label_list,
                                                student_logits_list,
                                                teacher_logits_list,
                                                topk_weight_list,
                                                top_k=3):
        label_list = [torch.squeeze(ele, dim=1).long() for ele in unsqueeze_label_list]
        # weight_list = [torch.squeeze(weight[], dim=1) for weight in topk_weight_list]

        teacher_masked_logits_list = teacher_logits_list
        teacher_topk_action_list = [torch.topk(x, top_k, dim=-1).indices for x in teacher_masked_logits_list]

        loss_list = []
        for i in range(len(label_list)):
            weight_i = topk_weight_list[:, :, i + 1]
            weight = (weight_i != 0).float()

            # 使用 log_softmax, 避免指数运行带来的数值不稳定
            student_probs_log = F.log_softmax(student_logits_list[i], dim=1)
            teacher_probs_log = F.log_softmax(teacher_masked_logits_list[i], dim=1)

            # 计算全类别上的 KL 散度
            kl_div_loss = F.kl_div(student_probs_log, teacher_probs_log, reduction='none', log_target=True)
            # kl_div_loss_full = kl_div_loss.sum(dim=1)  # 全类别 KL 散度

            # 只计算前 k 个类别的 KL 散度
            # top_k_indices = torch.topk(teacher_probs_log, top_k, dim=1).indices
            top_k_indices = teacher_topk_action_list[i]
            # topk_kl_div_loss = kl_div_loss.gather(dim=1, index=top_k_indices).sum(dim=1)  # 前 k 个类别的KL散度
            topk_kl_loss = kl_div_loss.gather(dim=1, index=top_k_indices)  # 前 k 个类别的KL散度
            topk_kl_loss = torch.sum(weight * topk_kl_loss, dim=-1)
            kl_div_final_loss = torch.mean(topk_kl_loss)

            loss_list.append(kl_div_final_loss)

        loss = torch.sum(torch.stack(loss_list))
        return loss, loss_list

    def compute_loss(self, data_list, rst_list):
        cost_all = torch.tensor(0.0, dtype=torch.float32)
        all_hero_loss_list = []
        all_acc_list = []

        top_k = 5

        hero_legal_sub_action_list = np.array(sub_action_mask_list)
        hero_legal_sub_action_list = torch.tensor(hero_legal_sub_action_list).to(rst_list[0][0].device)

        for hero_index in range(len(data_list)):
            this_hero_label_task_count = len(self.hero_label_size_list[hero_index])
            data_index = 1

            # legal action
            official_legal_action_list = data_list[hero_index][1: 1 + 5]
            data_index += this_hero_label_task_count

            # reward
            data_index += 1

            # advantage
            data_index += 1

            # action (label)
            this_hero_action_list = data_list[hero_index][data_index:(data_index + this_hero_label_task_count)]
            data_index += this_hero_label_task_count

            # action (prob lists, each corresponds to a sub-task)
            this_hero_probability_list = data_list[hero_index][data_index:(data_index + this_hero_label_task_count)]
            this_hero_logits_list = this_hero_probability_list
            data_index += this_hero_label_task_count

            # is_train
            data_index += 1

            # sub_action
            this_hero_weight_list = data_list[hero_index][data_index:(data_index + this_hero_label_task_count)]
            this_hero_weight_list_new = [torch.ones_like(t) for t in this_hero_weight_list]
            data_index += this_hero_label_task_count  # originally (task_num + 1)

            # policy network output
            this_hero_fc_label_list = rst_list[hero_index][:-1]  # logits

            # value network output
            this_hero_value = rst_list[hero_index][-1]

            # teacher的logits
            teacher_logits_list = this_hero_logits_list
            # 使用legal_action作为掩码对logits_list进行处理
            teacher_masked_logits_list = []
            for logits, legal_action in zip(teacher_logits_list, official_legal_action_list):
                new_logits = logits.clone()
                # new_logits[legal_action == 0] = float('-inf')
                new_logits[legal_action == 0] = -1e8
                teacher_masked_logits_list.append(new_logits)
            teacher_topk_action_list = [torch.topk(x, top_k, dim=-1).indices for x in teacher_masked_logits_list]
            # 替换sub_action
            new_button = teacher_topk_action_list[0].reshape(-1).int()

            new_sub_action_list1 = hero_legal_sub_action_list[hero_index][new_button]
            new_sub_action_list = new_sub_action_list1.reshape(-1, top_k, 6)

            # student 的logits
            student_logits_list = this_hero_fc_label_list
            # 使用legal_action作为掩码对logits_list进行处理
            student_masked_logits_list = []
            for logits, legal_action in zip(student_logits_list, official_legal_action_list):
                new_logits = logits.clone()
                # new_logits[legal_action == 0] = float('-inf')
                new_logits[legal_action == 0] = -1e8
                student_masked_logits_list.append(new_logits)
            student_topk_action_list = [torch.topk(x, top_k, dim=-1).indices for x in student_masked_logits_list]

            temperature = self.distill_temperature
            lambda_weight = self.distill_weight

            # hard label loss
            this_hero_hard_loss_list = self._calculate_single_hero_hard_loss(this_hero_action_list,
                                                                             this_hero_fc_label_list,
                                                                             this_hero_weight_list)

            # soft label loss
            this_hero_soft_loss_list = self._calculate_single_hero_soft_loss(this_hero_fc_label_list,
                                                                             this_hero_logits_list,
                                                                             this_hero_weight_list_new)
            # distill loss
            this_hero_distill_loss_list = self._calculate_single_hero_distill_loss(this_hero_action_list,
                                                                                   this_hero_fc_label_list,
                                                                                   this_hero_logits_list,
                                                                                   this_hero_weight_list_new,
                                                                                   temperature=temperature,
                                                                                   lambda_weight=lambda_weight)
            # temperatured soft loss
            this_hero_temperatured_soft_loss_list = self._calculate_single_temperatured_soft_loss(this_hero_action_list,
                                                                                                  this_hero_fc_label_list,
                                                                                                  this_hero_logits_list,
                                                                                                  this_hero_weight_list_new,
                                                                                                  temperature=temperature)
            # kl_div loss
            this_hero_kl_div_loss_list = self._calculate_single_hero_kl_div_loss(this_hero_action_list,
                                                                                 this_hero_fc_label_list,
                                                                                 this_hero_logits_list,
                                                                                 this_hero_weight_list_new,
                                                                                 temperature=temperature)
            # topk kl_div loss
            this_hero_topk_kl_div_loss_list = self._calculate_single_hero_topk_kl_div_loss(this_hero_action_list,
                                                                                           this_hero_fc_label_list,
                                                                                           teacher_masked_logits_list,
                                                                                           new_sub_action_list,
                                                                                           top_k=top_k)

            cost_all = cost_all + \
                       this_hero_kl_div_loss_list[0] * (temperature ** 2)

            all_hero_loss_list.append(
                [
                    this_hero_hard_loss_list[0],
                    this_hero_soft_loss_list[0],
                    this_hero_temperatured_soft_loss_list[0],
                    this_hero_topk_kl_div_loss_list[0],
                    this_hero_kl_div_loss_list[0]
                ]
            )

            topk_acc_list = torch.zeros((5, top_k))
            # 计算 top k action 准确率，student 相对 teacher
            for action_idx in range(5):
                teacher_actions = teacher_topk_action_list[action_idx]
                student_actions = student_topk_action_list[action_idx]
                teacher_sub_action_mask = new_sub_action_list[:, :, action_idx + 1].int()
                is_equal = (student_actions == teacher_actions) & teacher_sub_action_mask
                for kk in range(1, top_k + 1):
                    count1 = torch.sum(is_equal[:, kk - 1])
                    count2 = torch.sum(teacher_sub_action_mask[:, kk - 1])
                    if count2 != 0:
                        topk_acc_list[action_idx][kk - 1] = (count1 / count2).cpu()

            all_acc_list.append(topk_acc_list)

        # all_hero_loss_list (3, 5)   all_acc_list (5, top_k)
        out_all_acc_list = torch.mean(torch.stack(all_acc_list), dim=0).cpu().numpy()
        result_dict = {
            'loss': cost_all,
            'loss_list': all_hero_loss_list,
            'acc_list': out_all_acc_list
        }
        return result_dict

    def format_data(self, datas):  # ([32, 242352])  batch_size = 32
        # datas_1 = datas.reshape(32, -1)
        datas = datas.view(-1, self.hero_num, self.hero_data_len)  # ([32, 3, 80784])
        data_list = datas.permute(1, 0, 2)  # ([3, 32, 80784])

        hero_data_list = []
        for hero_index in range(self.hero_num):
            # calculate length of each frame
            hero_each_frame_data_length = np.sum(np.array(self.hero_data_split_shape[hero_index]))  # 4921
            hero_sequence_data_length = hero_each_frame_data_length * self.lstm_time_steps  # 78736 = 4921 * 16, 有连续16帧数据
            hero_sequence_data_split_shape = [hero_sequence_data_length, self.lstm_unit_size,
                                              self.lstm_unit_size]  # [78736, 1024, 1024]

            sequence_data, lstm_cell_data, lstm_hidden_data = data_list[hero_index].float().split(
                hero_sequence_data_split_shape, dim=1)  # ([32, 78736]), ([32, 1024]), ([32, 1024])
            reshape_sequence_data = sequence_data.reshape(-1,
                                                          hero_each_frame_data_length)  # torch.Size([32 * 16, 4921])
            hero_data = reshape_sequence_data.split(self.hero_data_split_shape[hero_index], dim=1)
            hero_data = list(hero_data)  # convert from tuple to list
            hero_data.append(lstm_cell_data)
            hero_data.append(lstm_hidden_data)
            hero_data_list.append(hero_data)
        return hero_data_list

    def convert_to_datas(self, hero_data_list):
        # 计算每个英雄的数据长度
        datas = []
        for hero_index in range(self.hero_num):
            # 获取每个英雄的数据
            hero_data = hero_data_list[hero_index]

            # 分别提取每个帧的数据
            sequence_data = torch.cat(hero_data[:-2], dim=1)  # 除去 lstm_cell_data 和 lstm_hidden_data
            lstm_cell_data = hero_data[-2]
            lstm_hidden_data = hero_data[-1]

            # 计算需要的形状
            hero_each_frame_data_length = np.sum(np.array(self.hero_data_split_shape[hero_index]))
            hero_sequence_data_length = hero_each_frame_data_length * self.lstm_time_steps

            # 重塑数据为原始形状
            reshaped_sequence_data = sequence_data.reshape(-1, self.lstm_time_steps, hero_each_frame_data_length)
            reshaped_sequence_data = sequence_data.reshape(-1, hero_sequence_data_length)
            hero_data = torch.cat([reshaped_sequence_data, lstm_cell_data, lstm_hidden_data], dim=1)
            hero_data = hero_data.unsqueeze(0)
            datas.append(hero_data)

        # 将所有英雄的数据连接到一起
        datas = torch.cat(datas, dim=0)  # (总英雄数, 32, 80784)
        datas = datas.permute(1, 0, 2)

        # 恢复为初始的形状
        datas = datas.reshape(-1, self.hero_num * self.hero_data_len)  # (batch_size, hero_num * 每个英雄的每帧数据长度)
        return datas


#######################
## Utility functions ##
#######################

def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Wrapper function to create and initialize a linear layer

    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    # initialize weight and bias
    # nn.init.xavier_uniform_(fc_layer.weight)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


############################
## Building-chunk classes ##
############################


class LSTMCell(nn.Module):
    """"""

    def __init__(self, input_size, hidden_size, batch_first=True, forget_bias=1.):
        """Initialize params."""
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.forget_bias = forget_bias

        self.fc = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        # self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        # initialize weight and bias
        nn.init.xavier_uniform_(self.fc.weight)  # tanh should use xvavier_uniform initializer
        # nn.init.orthogonal_(self.fc.weight)

        if hasattr(self.fc, 'bias'):
            nn.init.zeros_(self.fc.bias)

    def forward(self, input, hidden):
        """Propogate input through the network."""

        # tag = None  #
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            # gates = self.input_weights(input) + \
            #     self.hidden_weights(hx)
            if len(hx.shape) == 3:
                assert hx.shape[0] == 1  # (time_steps=1, batch_size, hidden_dim)
                assert cx.shape[0] == 1
                hx, cx = hx[0], cx[0]
            gates = self.fc(torch.cat([input, hx], dim=1))

            # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ingate, cellgate, forgetgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            # forgetgate = F.sigmoid(forgetgate)
            forgetgate = F.sigmoid(forgetgate + self.forget_bias)

            cellgate = F.tanh(cellgate)  # o_t
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.temperature = 1
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q * (Config.DK_SCALE / self.temperature), k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)  # (bs, n_head, lq, dv)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, hidden_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = make_fc_layer(d_model, n_head * d_k, use_bias=True)
        self.w_ks = make_fc_layer(d_model, n_head * d_k, use_bias=True)
        self.w_vs = make_fc_layer(d_model, n_head * d_v, use_bias=True)

        self.hidden_dim = hidden_dim  # Note: just for test

        self.fc1 = make_fc_layer(n_head * d_v, self.hidden_dim, use_bias=True)
        self.fc2 = make_fc_layer(self.hidden_dim, d_model, use_bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: bs x n_head x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)  # # (bs, n_head, lq, dv)

        # Transpose to move the head dimension back: bs x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: bs x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        # q = self.dropout(self.fc(q))

        # q += residual

        # q = self.layer_norm(q)
        q = self.fc2(F.relu(self.fc1(q)))

        q += residual

        return q
        # return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MLP(nn.Module):
    """A simple multi-layer perceptron
    """

    def __init__(self, fc_feat_dim_list: List[int], name: str, non_linearity: nn.Module = nn.ReLU,
                 non_linearity_last: bool = False):
        """Create a MLP object

        Args:
            fc_feat_dim_list (List[int]): ``in_features`` of the first linear layer followed by
                ``out_features`` of each linear layer
            name (str): human-friendly name, serving as prefix of each comprising layers
            non_linearity (nn.Module, optional): the activation function to use. Defaults to nn.ReLU.
            non_linearity_last (bool, optional): whether to append a activation function in the end.
                Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


def _compute_conv_out_shape(kernel_size: Tuple[int, int], padding: Tuple[int, int], input_shape: Tuple[int, int],
                            stride: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    """Compute the ouput shape of a convolution layer

    Args:
        kernel_size (Tuple[int, int]): kernel_size
        padding (Union[str, int]): either explicit padding size to add in both directions or
            padding scheme (either "same" or "valid)
        input_shape (Tuple[int, int]): [description]
        stride (Tuple[int, int], optional): [description]. Defaults to (1,1).

    Returns:
        Tuple[int, int]: height and width of the convolution ouput
    """
    out_x = floor((input_shape[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0]) + 1
    out_y = floor((input_shape[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1]) + 1
    return (out_x, out_y)


def make_conv_layer(kernel_size: Tuple[int, int], in_channels: int, out_channels: int, padding: str,
                    stride: Tuple[int, int] = (1, 1), input_shape=None):
    """Wrapper function to create and initialize ``Conv2d`` layers. Returns output shape along the
        way if input shape is supplied. Add support for 'same' and 'valid' padding scheme (would
        be unnecessary if using pytorch 1.9.0 and higher).

    Args:
        kernel_size (Tuple[int, int]): height and width of the kernel
        in_channels (int): number of channels of the input image
        out_channels (int): number of channels of the convolution output
        padding (Union[str, Tuple[int, int]]): either explicit padding size to add in both
            directions or padding scheme (either "same" or "valid)
        stride (Union[int, Tuple[int, int]], optional): stride. Defaults to (1,1).
        input_shape (Tuple[int, int], optional): height and width of the input image. Defaults
            to None.

    Returns:
        (nn.Conv2d, Tuple[int, int]): the initialized convolution layer and the shape of the
            output image if input_shape is not None.
    """

    if isinstance(padding, str):
        assert padding in ['same', 'valid'], "Padding scheme must be either 'same' or 'valid'"
        if padding == 'valid':
            padding = (0, 0)
        else:
            assert stride == 1 or (
                    stride[0] == 1 and stride[1] == 1), "Stride must be 1 when using 'same' as padding scheme"
            assert kernel_size[0] % 2 and kernel_size[1] % 2, \
                "Currently, requiring kernel height and width to be odd for simplicity"
            padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # initialize weight and bias
    # nn.init.xavier_normal_(conv_layer.weight)
    nn.init.orthogonal_(conv_layer.weight)
    nn.init.zeros_(conv_layer.bias)

    # compute output shape
    output_shape = None
    if input_shape:
        output_shape = _compute_conv_out_shape(kernel_size, padding, input_shape, stride)

    return conv_layer, output_shape


# 定义不同计算量的模型
# encoder: FLOPs = 560.923776M, Params = 4.82832
def arch_mlp_flop_x90p():
    # FLOPs = 166.494208M, Params = 83.270368M
    concat_mlp = make_mlp_layer([19 * 224, 5120, 4096, 4096, 2048, 2048, 2048, 2048, 1024, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x80p():
    # FLOPs = 149.716992M, Params = 74.877664M
    concat_mlp = make_mlp_layer([19 * 224, 5120, 4096, 4096, 2048, 2048, 1024, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x70p():
    # FLOPs = 132.612096M, Params = 66.324192M
    concat_mlp = make_mlp_layer([19 * 224, 4096, 4096, 4096, 2048, 2048, 1024, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x60p():
    # FLOPs = 112.164864M, Params = 56.101088M
    concat_mlp = make_mlp_layer([19 * 224, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 1024, 512, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x50p():
    # FLOPs = 93.290496M, Params = 46.658784M
    concat_mlp = make_mlp_layer([19 * 224, 4096, 4096, 2048, 1024, 1024, 512, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x40p():
    # FLOPs = 75.988992M, Params = 38.00752M
    concat_mlp = make_mlp_layer([19 * 224, 4096, 2048, 2048, 2048, 1024, 1024, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x30p():
    # FLOPs = 56.459264M, Params = 28.24368M
    concat_mlp = make_mlp_layer([19 * 224, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x20p():
    # FLOPs = 37.584896M, Params = 18.801376M
    concat_mlp = make_mlp_layer([19 * 224, 2048, 2048, 1024, 1024, 1024, 1024, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_flop_x10p():
    # FLOPs = 18.382848M, Params = 9.19728M
    concat_mlp = make_mlp_layer([19 * 224, 1024, 1024, 1024, 1024, 1024, 512, 224])
    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


def arch_mlp_by_scale(flops=''):
    if flops == '10p':
        # 10% FLOPs
        # FLOPs = 18.428672M, Params = 9.218364M
        concat_mlp = make_mlp_layer([19 * 224, 1500, 1024, 768, 512, 224])
    elif flops == '9p':
        # 9% FLOPs
        # FLOPs = 16.821344M, Params = 8.41424M
        concat_mlp = make_mlp_layer([19 * 224, 1500, 820, 512, 512, 224])
    elif flops == '8p':
        # 8% FLOPs
        # FLOPs = 14.956288M, Params = 7.480986M
        concat_mlp = make_mlp_layer([19 * 224, 1500, 512, 350, 256, 224])
    elif flops == '7p':
        # 7% FLOPs
        # FLOPs = 13.049088M, Params = 6.527186M
        concat_mlp = make_mlp_layer([19 * 224, 1300, 512, 350, 256, 224])
    elif flops == '6p':
        # 6% FLOPs
        # FLOPs = 11.218688M, Params = 5.611836M
        concat_mlp = make_mlp_layer([19 * 224, 1100, 512, 400, 256, 224])
    elif flops == '5p':
        # 5% FLOPs
        # FLOPs = 9.376384M, Params = 4.69037M
        concat_mlp = make_mlp_layer([19 * 224, 930, 512, 256, 256, 224])
    elif flops == '4p':
        # 4% FLOPs
        # FLOPs = 7.505792M, Params = 3.754678M
        concat_mlp = make_mlp_layer([19 * 224, 790, 256, 256, 256, 224])
    elif flops == '3p':
        # 3% FLOPs
        # FLOPs = 5.610752M, Params = 2.806948M
        concat_mlp = make_mlp_layer([19 * 224, 580, 256, 256, 256, 224])
    elif flops == '2p':
        # 2% FLOPs
        # FLOPs = 3.765248M, Params = 1.88376M
        concat_mlp = make_mlp_layer([19 * 224, 400, 256, 128, 128, 224])
    elif flops == '1p':
        # 1% FLOPs
        # FLOPs = 1.87648M, Params = 0.939048M
        concat_mlp = make_mlp_layer([19 * 224, 200, 128, 128, 128, 224])
    else:
        raise NotImplementedError(flops)

    config = {
        'concat_mlp': concat_mlp,
    }
    return NetworkModel(config)


if __name__ == '__main__':
    net = NetworkModel()
    print(net)
