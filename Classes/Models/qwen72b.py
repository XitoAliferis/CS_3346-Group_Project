from dataclasses import dataclass
from Classes.Models.api_models import ApiModel, ApiModelConfig

@dataclass
class Qwen72BConfig(ApiModelConfig):
    model_name: str = "qwen/qwen-2.5-72b-instruct"

class Qwen72B(ApiModel):
    def __init__(self, cfg: Qwen72BConfig | None = None):
        cfg = cfg or Qwen72BConfig()
        super().__init__(cfg)
