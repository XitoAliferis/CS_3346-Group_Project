from dataclasses import dataclass
from api_models import ApiModel, ApiModelConfig

@dataclass
class GPTOSS120BConfig(ApiModelConfig):
    model_name: str = "openai/gpt-oss-120b:free"

class GPTOSS120B(ApiModel):
    def __init__(self, cfg: GPTOSS120BConfig | None = None):
        cfg = cfg or GPTOSS120BConfig()
        super().__init__(cfg)
