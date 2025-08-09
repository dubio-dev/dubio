# Copyright (c) 2025 Project Jamify
#               2025 Declare-Lab
#               2025 AMAAI Lab
#               2025 Renhang Liu (liurenhang0@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DPO Training Script for DiffRhythm

This script shows how to use the DPO trainer for preference optimization.
"""

from omegaconf import OmegaConf
import sys

from jam.dpo.dpo_trainer import DPOTrainer
from jam.dpo.dpo_cfm import DPOCFM
from jam.model.dit import DiT


def main():
    cfg_cli = OmegaConf.from_dotlist(sys.argv[1:])
    config_path = cfg_cli.get("config", "configs/dpo.yaml")
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, cfg_cli)

    model = DPOCFM(
        transformer=DiT(**cfg.model.dit),
        **cfg.model.cfm
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        data_cfg=cfg.data,
        wandb_cfg=cfg.wandb,
        **cfg.training,
        bnb_optimizer=False,
        config=cfg,
    )
    
    print(f"Starting DPO training")
    print(f"Dataset: {cfg.data.train_dataset.dpo_json_path}")
    print(f"Max steps: {cfg.training.max_steps}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()