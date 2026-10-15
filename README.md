# Poker AI

基于 Python / PyTorch 的双人德州扑克学习项目：训练共享基础策略、针对不同玩家单独适应，并汇总真人对局训练不输入玩家身份的共享模型。

Heads-up Texas Hold'em with a shared neural policy, per-player adaptation, and pooled offline learning from human games.

## 功能

- 52 张牌、翻牌前到河牌的完整双人牌局，六种离散动作和有效筹码限制。
- 对战筹码逐手延续，支持保存、续局和输光后重新开场。
- 基础训练：规则机器人与历史策略对手，普通强化学习及 L2E 启发的近似元学习。
- 个人适应：每位玩家独立模型，默认每 32 手或本场结束时更新。
- 共享训练：持久收集已完成牌局，均衡抽取不同玩家的数据，采用 AWR 启发的离线回归。
- 配对评估：比较基础版与适应版的 `bb/100` 及差值置信区间。

## 快速开始

Windows 上先安装 Python 3.12 并将 `python` 加入 PATH，然后在 PowerShell 中运行：

```powershell
git clone https://github.com/McCa1n/poker-ai.git
cd poker-ai

# 首次运行自动创建 .venv 并安装 CPU PyTorch / NumPy
.\run_poker.ps1 train --epochs 200

# 与自己的专属模型对战
.\run_poker.ps1 play --player Alice

# 只使用共享基础模型，不启用个人适应
.\run_poker.ps1 play --player Alice --base-only

# 查看已收集的经验；收集真人牌局后再运行共享训练
.\run_poker.ps1 replay-info
.\run_poker.ps1 train-shared --epochs 20

# 独立评估
.\run_poker.ps1 evaluate --hands 1000 --adapt-hands 128
```

`200` 是实验起点，不代表已经训练到可靠实力。所有命令和模型更新规则见 [完整使用说明](POKER_README.md)。其他平台可以安装 `requirements-poker.txt` 后使用 `python poker_ai.py`。

## 数据与模型

模型、玩家存档及真人对局数据库保存在本机 `poker_data/`，已被 Git 排除。仓库不包含预训练权重或任何玩家的训练数据。每手使用 AI 当时可见的信息，玩家身份不会成为共享网络输入。

普通对战使用个人模型；`--base-only` 使用启动时加载的共享模型。更新基础模型不会自动覆盖已有个人模型。

## 实现范围

这是教学与实验项目，不是 L2E / AWR 论文的完整复现，也没有 GTO 或实战盈利保证。包含机制和短程流程验证，尚无充分的长期训练、真人泛化或盈利证据。下注大小是离散动作，当前不支持多人桌和任意金额输入。
