import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from typing import Any, cast
# from brax.v1 import envs
import jax
import wandb

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Create configs
    from MRQ.mrq import MRQConfig
    from MRQ.trainer import Trainer, TrainerConfig

    trainer_config = TrainerConfig(**cfg.trainer)
    mrq_config = MRQConfig(**cfg.mrq)
    env_config = cfg.env
        

    # Create trainer
    trainer = Trainer(
        trainer_config=trainer_config,
        mrq_config=mrq_config,
        env_config=env_config,
    )

    wandb_run = None
    if cfg.wandb.enabled:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_config = cast(dict[str, Any], resolved_cfg) if isinstance(resolved_cfg, dict) else {}
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            tags=list(cfg.wandb.tags),
            notes=cfg.wandb.notes,
            mode=cfg.wandb.mode,
            config=wandb_config,
        )

    def _wandb_log(step: int, metrics):
        if wandb_run is not None:
            wandb.log(metrics, step=step)
    
    # Train
    trainer.train(log_fn=_wandb_log)

    if wandb_run is not None:
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    # jax.config.update("jax_debug_nans", True)
    main()