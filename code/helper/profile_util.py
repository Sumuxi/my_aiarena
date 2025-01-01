import importlib

import torch
import torch.nn as nn
import thop


def torch_profile(model, *input):
    total_params = sum(p.numel() for p in model.parameters())

    # 使用 torch.profiler 进行 FLOPs 计算
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,  # 记录张量形状
            with_modules=True,  # 启用模块级分析
            with_flops=True  # 启用 FLOPs 计算
    ) as prof:
        # 执行一次推理
        outputs = model(*input)

    # # 获取 FLOPs 结果，按总 FLOPs 排序并限制输出数量
    # print(prof.key_averages().table(sort_by="flops", row_limit=10))
    #
    # # 导出为 trace 文件以便后续可视化
    # prof.export_chrome_trace("trace.json")

    # 获取所有操作的 FLOPs 总和
    total_flops = sum([item.flops for item in prof.key_averages() if item.flops != 0])

    return total_flops, total_params


def thop_profile(model, input):
    # 使用 thop 计算 FLOPs 和参数量
    flops, params = thop.profile(model, inputs=(input,))
    return flops, params


def create_model(model_name):
    model_file = model_name
    module_name = f"{model_file}"
    class_name = "NetworkModel"
    # 动态导入模块
    module = importlib.import_module(module_name)
    # 从模块中获取类
    Net = getattr(module, class_name)
    return Net()


if __name__ == "__main__":
    from models.NetworkModel_v0 import NetworkModel

    net = NetworkModel()

    dummy_input = torch.rand(1, 242352)

    input_dict = {
        'conv_layers': torch.randn(1, 6, 17, 17),
        'fc1_img': torch.rand(1, 6, 64),
        'hero_mlp': torch.rand(1, 6, 251),
        'hero_main_mlp': torch.rand(1, 1, 44),
        'monster_mlp': torch.rand(1, 20, 28),
        'soldier_mlp': torch.rand(1, 10, 25),
        'organ_mlp': torch.rand(1, 3, 29),
        'global_mlp': torch.rand(1, 1, 68),
        # 'encoder': [torch.rand(1, 19, 224), torch.rand(1, 19, 224), torch.rand(1, 19, 224)],
        'fc1_public': torch.rand(1, 224),
        'fc1_label': torch.rand(1, 1024),
        'fc2_label_list': torch.rand(1, 224),
        'fc1_target': torch.rand(1, 1024),
        'fc2_target': torch.rand(1, 224),
        'fc1_target_token': torch.rand(1, 224),
        'fc2_target_token': torch.rand(1, 64),
        'fc1_value': torch.rand(1, 1024),
        'fc2_value': torch.rand(1, 224),
        # 'lstm': [torch.rand(16, 1, 1024), (torch.rand(1, 1, 1024), torch.rand(1, 1, 1024))]
    }

    results = []
    flops, params = torch_profile(net, net.format_data(dummy_input))
    results.append({
        'name': 'net torch_profile',
        'FLOPs': flops / (1000 ** 2),
        'Params': params / (1000 ** 2),
    })

    flops, params = thop_profile(net, net.format_data(dummy_input))
    results.append({
        'name': 'net thop_profile',
        'FLOPs': flops / (1000 ** 2),
        'Params': params / (1000 ** 2),
    })

    for name, module in net.named_children():
        if name not in input_dict.keys():
            continue
        print("name: ", name, module)
        x = input_dict[name]
        if isinstance(module, nn.ModuleList):
            module = module[2]
        flops, params = torch_profile(module, x)
        # flops, params = thop_profile(module, x)
        results.append({
            'name': name,
            'FLOPs': flops / (1000 ** 2),
            'Params': params / (1000 ** 2),
        })

    print("Result:")
    for item in results:
        print(f"{item['name']}: FLOPs = {item['FLOPs']}M, Params = {item['Params']}M")

    # modules = {
    #     # 特征处理
    #     'conv_layers': (model_dict['arch1_v1'].conv_layers, torch.rand(3, 6, 17, 17)),
    #     'hero_share_mlp': (model_dict['arch1_v1'].hero_share_mlp, torch.rand(3, 251)),
    #     'hero_frd_mlp': (model_dict['arch1_v1'].hero_frd_mlp, torch.rand(3, 256)),
    #     'hero_emy_mlp': (model_dict['arch1_v1'].hero_emy_mlp, torch.rand(3, 256)),
    #     'public_info_mlp': (model_dict['arch1_v1'].public_info_mlp, torch.rand(3, 44)),
    #     'soldier_share_mlp': (model_dict['arch1_v1'].soldier_share_mlp, torch.rand(3, 25)),
    #     'soldier_frd_mlp': (model_dict['arch1_v1'].soldier_frd_mlp, torch.rand(3, 128)),
    #     'soldier_emy_mlp': (model_dict['arch1_v1'].soldier_emy_mlp, torch.rand(3, 128)),
    #     'organ_share_mlp': (model_dict['arch1_v1'].organ_share_mlp, torch.rand(3, 29)),
    #     'organ_frd_mlp': (model_dict['arch1_v1'].organ_frd_mlp, torch.rand(3, 128)),
    #     'organ_emy_mlp': (model_dict['arch1_v1'].organ_emy_mlp, torch.rand(3, 128)),
    #     'monster_mlp': (model_dict['arch1_v1'].monster_mlp, torch.rand(3, 28)),
    #     'global_mlp': (model_dict['arch1_v1'].global_mlp, torch.rand(3, 68)),
    #     'concat_mlp': (model_dict['arch1_v1'].concat_mlp, torch.rand(3, 1344)),
    #     'action_heads[0]': (model_dict['arch1_v1'].action_heads[0], torch.rand(3, 512)),
    #     'action_heads[1]': (model_dict['arch1_v1'].action_heads[1], torch.rand(3, 512)),
    #     'action_heads[2]': (model_dict['arch1_v1'].action_heads[2], torch.rand(3, 512)),
    #     'action_heads[3]': (model_dict['arch1_v1'].action_heads[3], torch.rand(3, 512)),
    #     'action_heads[4]': (model_dict['arch1_v1'].action_heads[4], torch.rand(3, 512)),
    # }
    # total_f = 0
    # for k, (m, x) in modules.items():
    #     flops, params = profile(m, x)
    #     results.append({
    #             'name': k,
    #             'FLOPs': flops / (1000 ** 2),
    #             'Params': params / (1000 ** 2),
    #         })
    #     total_f = total_f + (flops / (1000 ** 2))
    # print("Result:")
    # for item in results:
    #     print(f"{item['name']}: {item['FLOPs']}MFLOPs, {item['Params']}MParams")
    #
    # print(f"total_f: {total_f}")
