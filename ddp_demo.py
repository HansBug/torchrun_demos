import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# 1. 定义模型 - 使用预训练的ResNet18并修改最后的全连接层以适应CIFAR-100
class CIFAR100Model(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR100Model, self).__init__()
        # 使用预训练的ResNet18
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 修改第一个卷积层以适应CIFAR-100的较小输入尺寸
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除最大池化层，因为CIFAR-100图像较小
        self.model.maxpool = nn.Identity()
        # 修改全连接层以适应CIFAR-100的类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# 2. 准备数据加载和增强
def get_data_loaders(rank, world_size, batch_size=128):
    # 数据增强和归一化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # 下载并加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 测试集不需要分布式采样器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler


# 3. 评估函数
def evaluate(model, test_loader, criterion, device, rank, world_size):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 收集所有进程的结果
    metrics = torch.tensor([test_loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    # 计算全局指标
    global_loss = metrics[0].item() / metrics[2].item()
    global_accuracy = metrics[1].item() / metrics[2].item() * 100

    return global_loss, global_accuracy


# 4. 训练函数
def train(rank, world_size, epochs=100, batch_size=128, lr=0.1):
    # 设置TensorBoard
    if rank == 0:
        tb_writer = SummaryWriter(f'runs/cifar100_ddp_ws{world_size}')
        print(f"TensorBoard logs will be saved to: runs/cifar100_ddp_ws{world_size}")

    # 设置设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 创建模型并移动到设备
    model = CIFAR100Model().to(device)

    # 将模型包装为DDP模型
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 获取数据加载器
    train_loader, test_loader, train_sampler = get_data_loaders(rank, world_size, batch_size)

    # 记录最佳准确率，用于保存最佳模型
    best_acc = 0.0

    # 训练循环
    for epoch in range(epochs):
        start_time = time.time()

        # 设置采样器的epoch
        train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 累计统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # # 打印批次进度（仅主进程）
            # if rank == 0 and (batch_idx + 1) % 50 == 0:
            #     print(f"Epoch: {epoch + 1}/{epochs} | Batch: {batch_idx + 1}/{len(train_loader)} | "
            #           f"Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%")

        # 收集所有进程的训练结果
        train_metrics = torch.tensor([train_loss, correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)

        # 计算全局训练指标
        global_train_loss = train_metrics[0].item() / train_metrics[2].item()
        global_train_acc = train_metrics[1].item() / train_metrics[2].item() * 100

        # 评估模型
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, rank, world_size)

        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 计算epoch时间
        epoch_time = time.time() - start_time

        # 记录TensorBoard日志（仅主进程）
        if rank == 0:
            tb_writer.add_scalar('train/loss', global_train_loss, epoch)
            tb_writer.add_scalar('test/loss', test_loss, epoch)
            tb_writer.add_scalar('train/accuracy', global_train_acc, epoch)
            tb_writer.add_scalar('test/accuracy', test_acc, epoch)
            tb_writer.add_scalar('train/learning_rate', current_lr, epoch)
            tb_writer.add_scalar('train/epoch_time', epoch_time, epoch)

            # 记录模型参数直方图
            for name, param in model.named_parameters():
                tb_writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    tb_writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            print(f"Epoch: {epoch + 1}/{epochs} | "
                  f"Train Loss: {global_train_loss:.4f} | Train Acc: {global_train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.module.state_dict(), f"cifar100_best_model_ws{world_size}.pt")
                print(f"Best model saved with accuracy: {best_acc:.2f}%")

        # 同步所有进程
        dist.barrier()

    # 保存最终模型（仅主进程）
    if rank == 0:
        torch.save(model.module.state_dict(), f"cifar100_final_model_ws{world_size}.pt")
        print(f"Training completed. Final model saved.")
        tb_writer.close()


# 5. 初始化进程组
def init_process(rank, world_size, backend='nccl'):
    # 获取环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # 初始化进程组
    dist.init_process_group(backend=backend)

    # 打印进程信息
    print(f"Initialized process {rank} (local_rank: {local_rank}) out of {world_size}")

    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Process {rank} using GPU: {torch.cuda.get_device_name(local_rank)}")


# 6. 主函数
def main():
    # 从环境变量获取信息
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    print(f"Starting process with rank {rank}, local_rank {local_rank}, world_size {world_size}")

    # 初始化进程组
    init_process(rank, world_size)

    # 开始训练
    train(local_rank, world_size)

    # 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
