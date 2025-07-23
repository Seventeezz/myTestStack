import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import sys
import argparse

from Game.card_tools import card_tools
from NeuralNetwork.value_nn_torch import ValueNn
os.chdir('..')
sys.path.append( os.path.join(os.getcwd(),'src') )

STREET_TO_PHASE = {
    1: 'preflop',
    2: 'flop',
    3: 'turn',
    4: 'river'
}


def basic_huber_loss(y_true, y_pred, delta=1.0):
    return F.huber_loss(y_pred, y_true, delta=delta)

def masked_huber_loss(y_true, y_pred, delta=1.0):
    mask = (y_true != 0).float()
    num_active = mask.sum()
    base_loss = F.huber_loss(y_pred, y_true, delta=delta, reduction='none')
    masked_loss = base_loss * mask
    multiplier = y_true.numel() / (num_active + 1e-6)  # 防止除0
    return masked_loss.mean() * multiplier

# 定义数据Dataset
class PokerDataset(Dataset):
    def __init__(self, npy_dir):
        self.input_paths = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.startswith("inputs")])
        self.target_paths = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.startswith("targets")])
        self.board_paths = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.startswith("boards")])

        assert len(self.input_paths) == len(self.target_paths) == len(self.board_paths)

        # 预加载所有数据
        self.inputs = [np.load(p) for p in self.input_paths]
        self.targets = [np.load(p) for p in self.target_paths]
        self.boards = [np.load(p) for p in self.board_paths]

        self.data = []  # (x, y, board) 的扁平化列表
        for x_batch, y_batch, board_batch in zip(self.inputs, self.targets, self.boards):
            # 计算每个board对应的x_batch重复次数，保证对齐
            batch_size = len(x_batch) // len(board_batch)
            # board 扩展成和 x_batch 对齐的形状
            extended_boards = np.repeat(board_batch, batch_size, axis=0)  # [batch_size * num_boards, board_size]

            assert len(x_batch) == len(y_batch) == len(extended_boards), \
                f"Data length mismatch: {len(x_batch)}, {len(y_batch)}, {len(extended_boards)}"

            for i in range(len(x_batch)):
                self.data.append((x_batch[i], y_batch[i], extended_boards[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, board = self.data[idx]
        # 1. 使用和TFRecordsConverter完全一致的board转特征方法
        b = card_tools.convert_board_to_nn_feature(board)
        # 2. 拼接 x 和 b（board的特征）
        nn_input = np.zeros(len(x) + len(b), dtype=np.float32)
        nn_input[:len(x)] = x
        nn_input[len(x):] = b
        # 3. mask处理targets，参考TFRecordsConverter中的逻辑
        ranges = x[:-1]  # 忽略最后一个pot的值
        mask = np.ones_like(ranges, dtype=np.float32)
        mask[ranges == 0] = 0
        nn_target = y * mask
        return torch.tensor(nn_input, dtype=torch.float32), torch.tensor(nn_target, dtype=torch.float32)

    def convert_board_to_nn_feature(self, board):
        # 这里直接返回float32类型board特征，如果你有card_tools里更复杂的转换，可以替换此处
        return board.astype(np.float32)


# === 训练函数 ===
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs = inputs.to(model.device)
        targets = targets.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = masked_huber_loss(targets, outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# === 验证函数 ===
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)
            outputs = model(inputs)
            loss = masked_huber_loss(targets, outputs)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--street', type=int, default=2, help='设置当前训练的street值（1: preflop, 2: flop, 3: turn, 4: river）')
    parser.add_argument('--train_type', type=str, default='root_nodes', choices=['root_nodes', 'leaf_nodes'], help='训练类型')
    args = parser.parse_args()

    STREET = args.street
    TRAIN_TYPE = args.train_type

    # 下面的 CFG 也要用新的 STREET 和 TRAIN_TYPE
    CFG = {
        'n_epochs': 100,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'n_workers': 0,
        'model_save_path': f'./data/Models/{STREET_TO_PHASE[STREET]}/weights.{TRAIN_TYPE}.pt',
        "data_path": f"./data/TrainSamples/{STREET_TO_PHASE[STREET]}/{TRAIN_TYPE}_npy",
    }

    dataset = PokerDataset(CFG['data_path'])
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['n_workers'])
    val_loader = DataLoader(val_set, batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['n_workers'])

    # === 初始化模型 ===
    model = ValueNn(
        street=STREET,  # 替换成你训练的 street 值（0,1,2,3）
        pretrained_weights=False,
        approximate='root_nodes',
        trainning_mode=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])

    # === 主训练循环 ===
    best_val_loss = float('inf')

    for epoch in range(CFG["n_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)

        print(f"[Epoch {epoch + 1}/{CFG['n_epochs']}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CFG["model_save_path"])
            print(f"✅ Saved new best model at {CFG['model_save_path']}")

    print("🎉 Training complete.")