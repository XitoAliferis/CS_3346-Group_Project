from dataclasses import dataclass
from Classes.Models.models import BaseModel, BaseModelConfig


@dataclass
class Qwen5BConfig(BaseModelConfig):
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lr: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lora_rank: int = 16
    use_lora: bool = True

class Qwen5B(BaseModel):
    def __init__(self, cfg: Qwen5BConfig | None = None):
        cfg = cfg or Qwen5BConfig()     
        super().__init__(cfg)
