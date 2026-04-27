import logging
from typing import List, Literal, Optional, Union

import torch
import torch.distributed as dist
import torch.optim.optimizer as Optimizer


class OneCycleAutoLR(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start=0.3,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        cycle_momentum=True,
        base_momentum: Union[float, List[float]] = 0.85,
        max_momentum: Union[float, List[float]] = 0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        auto_lr_warmup_steps=100000,
        ensemble_lr_spread_ratio=0.0,
        model_sync_interval=1000,
        rank=0,
        world_size=1,
        enable_auto_lr: bool = False,
        loss_scale: float = 0.002,
        lr_of_lr: float = 0.5,
        momentum_of_lr: float = 0.9,
    ):
        self.auto_lr_warmup_steps = auto_lr_warmup_steps
        self.ensemble_lr_spread_ratio = ensemble_lr_spread_ratio
        self.model_sync_interval = model_sync_interval
        self.rank = rank
        self.world_size = world_size
        self.detached_loss = 0.0
        self.enable_auto_lr = enable_auto_lr
        self.loss_scale = loss_scale
        self.dividor = max((self.world_size - 1) / 2.0, 0.5)
        self.lr_of_lr = lr_of_lr
        decay_steps = (total_steps - auto_lr_warmup_steps) / model_sync_interval
        self.decay_ratio = 1 - torch.exp(-torch.log(torch.tensor(div_factor / final_div_factor)) / decay_steps).item()
        self.velocity = 0.0
        self.momentum_of_lr = momentum_of_lr
        if self.ensemble_lr_spread_ratio > 0.0 and self.model_sync_interval == 0:
            raise ValueError(
                "model_sync_interval must be greater than 0 when ensemble_lr_spread_ratio is larger than 0.0"
            )
        super().__init__(
            optimizer,
            max_lr,
            total_steps,
            epochs,
            steps_per_epoch,
            pct_start,
            anneal_strategy,
            cycle_momentum,
            base_momentum,
            max_momentum,
            div_factor,
            final_div_factor,
            three_phase,
            last_epoch,
        )
        SEED = 1024
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    def gather_tensor_across_ranks(self, world_size: int, tensor: torch.Tensor):
        """
        loss: scalar tensor on each rank
        returns: dictionary mapping rank -> loss_value
        """
        # Make a tensor buffer for all ranks
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

        # All-gather the loss from every rank
        dist.all_gather(gathered, tensor)

        return gathered

    def update_lr(self, values) -> None:
        for _, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            if isinstance(param_group["lr"], torch.Tensor):
                lr_val = lr.item() if isinstance(lr, torch.Tensor) else lr  # type: ignore[attr-defined]
                param_group["lr"].fill_(lr_val)
            else:
                param_group["lr"] = lr

    def step(self, loss: Optional[torch.Tensor] = None) -> None:
        # Resolve rank/world_size at runtime
        if dist.is_initialized():
            runtime_world_size = dist.get_world_size()
            if runtime_world_size != self.world_size:
                self.rank = dist.get_rank()
                self.world_size = runtime_world_size
                self.dividor = max((self.world_size - 1) / 2.0, 0.5)

                if self.ensemble_lr_spread_ratio > 0.0:
                    local_rank_lr_ratio = (
                        1.0 + self.ensemble_lr_spread_ratio * (self.rank - (self.world_size - 1) / 2.0) / self.dividor
                    )
                    self.base_lrs = [lr * local_rank_lr_ratio for lr in self.base_lrs]
                    logging.info(f"Adjusting learning rates for rank {self.rank} (world_size={self.world_size})")

        self._step_count += 1
        self.last_epoch += 1
        self.detached_loss += loss.detach().clone().float().to(loss.device) if loss is not None else 0.0
        # need to sync model weights across ranks before updating learning rate

        if (
            self.last_epoch < self.auto_lr_warmup_steps
            or (not self.enable_auto_lr)
            or self.ensemble_lr_spread_ratio == 0.0
        ):
            values = self.get_lr()
            self.update_lr(values)
            if (not self.enable_auto_lr) or (self.ensemble_lr_spread_ratio == 0.0):
                return

        if self.ensemble_lr_spread_ratio > 0.0 and self.last_epoch % self.model_sync_interval == 0:
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    # param.data holds the weight tensor.
                    # 1. AllReduce with SUM operation
                    # Sums the parameter data from ALL ranks into the tensor on THIS rank
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                    # 2. Divide by the total number of ranks to get the average
                    param.data.div_(self.world_size)
            logging.info(f"Rank {self.rank}: Successfully averaged model weights across {self.world_size} ranks.")
            if self.enable_auto_lr and self.last_epoch >= self.auto_lr_warmup_steps:
                self.detached_loss = self.detached_loss / self.model_sync_interval
                rank_loss_list = self.gather_tensor_across_ranks(world_size=self.world_size, tensor=self.detached_loss)
                logging.info(f"{self.rank} rank_loss_list: {rank_loss_list}")
                avg_loss = sum(rank_loss_list) / len(rank_loss_list)
                rank_loss_list = [(avg_loss - loss) / self.loss_scale for loss in rank_loss_list]
                rank_loss_tensor = torch.stack(rank_loss_list, dim=0)
                rank_loss_weight = torch.softmax(rank_loss_tensor, dim=0).unsqueeze(-1)
                lr_tensor = torch.tensor(self._last_lr, device=loss.device, dtype=torch.float32)
                rank_lr_list = self.gather_tensor_across_ranks(world_size=self.world_size, tensor=lr_tensor)
                logging.info(f"{self.rank} rank_lr_list: {rank_lr_list}")
                rank_lr_list = torch.stack(rank_lr_list, dim=0)
                weighted_avg_lr = rank_lr_list * rank_loss_weight
                weighted_avg_lr = weighted_avg_lr.sum(dim=0)
                rank_lr_list = torch.mean(rank_lr_list, dim=0)
                delta_lr = weighted_avg_lr - rank_lr_list
                self.velocity = self.momentum_of_lr * self.velocity + (1 - self.momentum_of_lr) * delta_lr
                weighted_avg_lr = rank_lr_list * (1 - self.decay_ratio) + self.velocity * self.lr_of_lr
                values = []
                for lr in weighted_avg_lr:
                    shuffle_indices = torch.randperm(self.world_size).tolist()
                    logging.info(f"{self.rank} shuffle_indices: {shuffle_indices}")
                    random_index = shuffle_indices[self.rank]
                    # make a random combination for different learning rate
                    local_rank_lr_ratio = (
                        1.0
                        + self.ensemble_lr_spread_ratio * (random_index - (self.world_size - 1) / 2.0) / self.dividor
                    )
                    values.append(lr * local_rank_lr_ratio)
                logging.info(f"Adjusting non-embedding learning rate for rank {self.rank}: {values}")
                self.update_lr(values)
                if self.rank == 0:
                    logging.info(f"rank_lr_list: {rank_lr_list}")
                    logging.info(f"rank_loss_weight: {rank_loss_weight}")
                    logging.info(f"average lr rate: {rank_lr_list}")
                    logging.info(f"delta lr rate: {delta_lr}")
                    logging.info(f"weighted_avg_lr: {weighted_avg_lr}")
                    logging.info(f"avg_loss: {avg_loss}")
        if self.last_epoch % self.model_sync_interval == 0:
            self.detached_loss = 0.0  # reset for next interval

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]