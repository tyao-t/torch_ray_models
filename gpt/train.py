from loss import evaluate_model, calc_loss_batch
from gpt_model import generate_and_print_sample
import torch, torch.optim, itertools, functools
from text_processor import create_dataloader
from gpt_model import GPTModel
from transformers.trainer_utils import find_executable_batch_size

# Adam: 自适应学习率 指数滑动平均 一阶矩 二阶矩 偏差修正 β1/β2 ε数值稳定 逐元素预条件化 梯度标准差归一化 动量/惯性 噪声平滑 各向异性缩放 非凸优化 尺度不变性 收敛稳健性。Adam通过对梯度做指数滑动平均得到一阶矩和二阶矩，经偏差修正后按逐元素预条件化把平均梯度除以其“标准差”，在非凸、各向异性地形里以自适应学习率平滑噪声、提升数值稳定与收敛稳健性。

# AdamW: AdamW 解耦weight decay 与L2正则区别 非通过梯度项直接衰减w 正则强度λ 不对Norm/偏置/Embedding衰减 泛化改进 大模型默认 与warmup/decay协同 学习率敏感性降低 动量与衰减解耦 参数组 可重复性 全参/指令微调稳定 数值稳定性。AdamW将权重衰减从梯度更新中解耦，在不扭曲动量与自适应缩放的前提下对参数直接收缩，配合warmup/decay在大模型预训练与微调中带来更稳的优化与更好的泛化。

# Learning Rate Warmup: 预热阶段 小学习率起步 线性/多项式上升 动量估计冷启动 二阶矩稳定期 混合精度动态范围 安全区间 防止早期发散/NaN 大batch线性缩放规则 早期高噪声平滑 深残差/注意力放大效应 峰值LR过渡 burn-in LoRA短暖启动 训练曲线平滑。Warmup用小LR逐步升到峰值，让动量与二阶矩在混合精度和大batch情况下先稳定下来，缓解深网络早期的噪声与放大效应，避免一上来就数值发散。

# Cosine Decay: 余弦退火 学习率调度 端点导数为0 先快后慢 探索→利用过渡 模拟退火“降温” 抗震荡 泛化提升 lr_min尾巴 单阶段训练收敛 避免step跳变 与warmup拼接 进度感知t/T 平坦极小值偏好 可继续训练/重启易衔接。Cosine decay在warmup之后让LR随进度先快后慢、平滑衰减到lr_min，起到“降温”与抗震荡作用，常带来更好的泛化并避免台阶式调度的数值冲击。

# Gradient Clipping: 全局范数裁剪 L2范数 阈值max_norm 统一比例缩放 防偶发梯度爆炸 长序列注意力极值 AMP先unscale再clip 梯度累积前裁一次 分布式全局归约 触发率监控 阈值调参 与LR/序列长度权衡 “保险丝”机制 直方图/日志监测 数值稳定提升。Clipping计算全局L2范数并在超过阈值时统一缩放梯度长度，像保险丝一样拦截偶发尖峰（长序列/混合精度常见），需在AMP下先反缩放并监控触发率以调好阈值与学习率。

# 小结：Warmup负责“安全起步”，cosine decay负责“平滑收尾”，Adam/AdamW用一、二阶矩做逐元素自适应步长与动量平滑（AdamW还解耦衰减以利泛化），clipping则在每一步提供“尖峰保护”，共同在可能高曲率与高噪声的非凸地形上实现稳定、可控、泛化更好的训练。
# 二阶矩估计（second moment） 被称为自适应学习率（adaptive learning rate），Root Mean Square Propagation”

# Hyper Param tuning
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],
    "drop_rate": [0.0, 0.1, 0.2],
    "warmup_steps": [10, 20, 30],
    "weight_decay": [0.1, 0.01, 0.0],
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],
    "initial_lr": [0.00005, 0.0001],
    "min_lr": [0.00005, 0.00001, 0.0001],
    "n_epochs": [5, 10, 15, 20, 25],
}

# @find_executable_batch_size(starting_batch_size=512, auto_find_batch_size=True)
# @auto_find_executable_batch_size(starting_batch_size=512)
def train_model(model, train_loader_wo_batch_size, val_loader_wo_batch_size, 
                optimizer, device, *, batch_size, warmup_steps=10, n_epochs=5, 
                eval_freq=10, eval_iter=1, initial_lr=3e-05, min_lr=1e-6, ctx=None):
    train_loader, val_loader = train_loader_wo_batch_size(batch_size), \
        val_loader_wo_batch_size(batch_size)
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    total_training_steps = len(train_loader) * n_epochs

    peak_lr = optimizer.param_groups[0]["lr"]
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment # Warmup

            else:
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps)) 
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) # Cosine annealing

            track_lrs.append(lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # start_context, tokenizer = "To be or not to be", tiktoken.get_encoding("gpt2")
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )

    return train_losses, val_losses, track_tokens_seen, track_lrs

if __name__ == "__main__":

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}. With device {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        if torch.cuda.get_device_capability()[0] >= 7:  # Hopper (9.0+), Ampere (8.0+), Turing (7.5+), Volta (7.0+)
            torch.set_float32_matmul_precision("high")
            print("Uses tensor cores")
        else:
            print("Tensor cores not supported on this GPU. Using default precision.")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}\n")


    # 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    hparam_combs = list(itertools.product(*HPARAM_GRID.values()))
    best_val_loss = torch.inf
    best_hparams = None # {k: v[0] for k, v in HPARAM_GRID.items()}

    text_data = "" # TO BE REPLACED

    tokenizer = tiktoken.get_encoding("gpt2")

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    interrupted = False
    current_config_idx_count = 0

    for combination in hyperparameter_combinations:
        try:
            current_config_idx_count += 1
            print(f"Evaluating configuration {current_config_idx_count} of {total_combinations}")

            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

            GPT_CONFIG_124M = {
                "vocab_size": 50304,
                "context_length": 256,
                "emb_dim": 768,
                "n_heads": 12,
                "n_layers": 12,
                "drop_rate": HPARAM_CONFIG["drop_rate"],
                "qkv_bias": False,
            }

            train_loader_wo_batch_size = functools.partial(
                create_dataloader,
                text=text_data[:split_idx],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=True,
                shuffle=True,
                num_workers=0
            )

            train_loader_wo_batch_size = functools.partial(
                create_dataloader,
                text=text_data[split_idx:],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=False,
                shuffle=False,
                num_workers=0
            )

            model = GPTModel(GPT_CONFIG_124M)
            model = torch.compile(model)
            # model._orig_mod.state_dict(), model.module if DDP
            model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=HPARAM_CONFIG["peak_lr"], 
                weight_decay=HPARAM_CONFIG["weight_decay"],
                fused=True # Fused AdamW optimizer Uses the fused kernels for AdamW
            )

            train_loss, val_loss = train_model(
                model, train_loader_wo_batch_size, train_loader_wo_batch_size, 
                optimizer, device, 
                batch_size=HPARAM_CONFIG["batch_size"],
                n_epochs=HPARAM_CONFIG["n_epochs"],
                eval_iter=1,
                warmup_steps=HPARAM_CONFIG["warmup_steps"],
                initial_lr=HPARAM_CONFIG["initial_lr"],
                min_lr=HPARAM_CONFIG["min_lr"],
                ctx=ctx
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG

        except KeyboardInterrupt:
            break

    print("Hyperparameter search completed.")
    print(f"Best hyperparameters: {best_hparams}")
    print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")

def auto_find_executable_batch_size(starting_batch_size=256):
    def _is_cuda_oom(exception: Exception) -> bool:
        msg = str(exception).lower()
        return (
            isinstance(exception, RuntimeError)
            and (
                "out of memory" in msg
                or "oom" in msg
                or "cuda error" in msg
                or "cudnn" in msg
            )
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            batch_size = kwargs.pop('batch_size', starting_batch_size)

            while batch_size > 0:
                try:
                    return func(batch_size=batch_size, *args, **kwargs)

                except Exception as e:
                    print(f"Exception for batch_size={batch_size}: {type(e).__name__}: {e}")

                    if isinstance(e, MemoryError) or _is_cuda_oom(e):
                        batch_size //= 2
                        print(f"Retrying with batch_size={batch_size}\n")
                        if not torch.cuda.is_available():
                            continue

                        try: torch.cuda.empty_cache()
                        except Exception: pass
                    else:
                        raise

            raise RuntimeError("No executable batch size found, reached zero.")

        return wrapper

    return decorator