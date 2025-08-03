import subprocess

from Settings.arguments import arguments

streets = [4, 3, 2]
approximate = "root_nodes"
train_type = "root_nodes"
files_per_round = arguments.gen_num_files  # 每轮生成5个文件
max_rounds = 100     # 最多轮数，可根据需要调整

for round_idx in range(max_rounds):
    start_idx = round_idx * files_per_round
    print(f"\n=== 新一轮开始，start-idx={start_idx} ===\n")
    for street in streets:
        # 1. 生成数据
        print(f"正在生成 street={street} 的训练数据，start-idx={start_idx} ...")
        subprocess.run([
            "python", "generate_data.py",
            "--street", str(street),
            "--approximate", 'root_nodes',
            "--start-idx", str(start_idx)
        ], check=True)

        # 2. 训练模型
        print(f"正在训练 street={street} 的模型...")
        subprocess.run([
            "python", "train_nn_torch.py",
            "--street", str(street),
            "--train_type", 'root_nodes'
        ], check=True)
    # for street in streets:
    #     # 1. 生成数据
    #     print(f"正在生成 street={street} 的训练数据，start-idx={start_idx} ...")
    #     subprocess.run([
    #         "python", "generate_data.py",
    #         "--street", str(street),
    #         "--approximate", approximate,
    #         "--start-idx", str(start_idx)
    #     ], check=True)
    #
    #     # 2. 训练模型
    #     print(f"正在训练 street={street} 的模型...")
    #     subprocess.run([
    #         "python", "train_nn_torch.py",
    #         "--street", str(street),
    #         "--train_type", train_type
    #     ], check=True)
    # print(f"\n=== 第 {round_idx+1} 轮完成 ===\n")

print("全部流程已自动完成！") 