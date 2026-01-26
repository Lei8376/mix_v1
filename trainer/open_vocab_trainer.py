from dataclasses import dataclass
from typing import Dict

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from MinkowskiEngine import SparseTensor

from model.criterion import Criteria
from utils.util import AverageMeter


@dataclass
class OpenVocabTrainerConfig:
    num_epochs: int = 10
    base_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    log_dir: str = "runs/open_vocab_3d"
    checkpoint_dir: str = "checkpoints"
    log_every_steps: int = 50
    scheduler_t0: int = 1
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6


class OpenVocabTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: OpenVocabTrainerConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )
        steps_per_epoch = max(1, len(train_loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=steps_per_epoch * config.scheduler_t0,
            T_mult=config.scheduler_t_mult,
            eta_min=config.scheduler_eta_min,
        )

        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.global_step = 0
        self.best_loss = float("inf")

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensor_keys = [
            "coords_3d",
            "feat_3d",
            "ori_coords_3d",
            "binary_label_3d",
            "binary_label_2d",
            "label_2d",
            "img",
            "x_label",
            "y_label",
            "inds_reconstruct",
        ]
        moved = dict(batch)
        for key in tensor_keys:
            moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def _build_sparse_tensor(self, batch: Dict[str, torch.Tensor]) -> SparseTensor:
        return SparseTensor(batch["feat_3d"], batch["coords_3d"])

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_loss = AverageMeter()
            for step, batch in enumerate(self.train_loader):
                batch = self._move_batch_to_device(batch)
                batch["sinput"] = self._build_sparse_tensor(batch)

                results = self.model(batch)
                criteria = Criteria(
                    results,
                    batch,
                    bce_weight=1.0,
                    dice_weight=1.0,
                )
                loss = criteria.loss_pt()

                if loss.item() == 0:
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.config.grad_clip_norm,
                )
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss.update(loss.item())
                self.writer.add_scalar("Loss/Train_Step", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "LR", self.scheduler.get_last_lr()[0], self.global_step
                )
                self.global_step += 1

                if step % self.config.log_every_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                        f"Step [{step}/{len(self.train_loader)}] "
                        f"Loss: {loss.item():.4f} LR: {current_lr:.2e}"
                    )

            self.writer.add_scalar("Loss/Train_Epoch", epoch_loss.avg, epoch)
            print(
                f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                f"Average Loss: {epoch_loss.avg:.4f}"
            )

            if epoch_loss.avg < self.best_loss:
                self.best_loss = epoch_loss.avg
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "loss": self.best_loss,
                    },
                    f"{self.config.checkpoint_dir}/best_model.pth",
                )
                print(f"  -> Saved best model (loss: {self.best_loss:.4f})")

        self.writer.close()

