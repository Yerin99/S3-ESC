
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.vanilla_blenderbot_small import Model as vanilla_blenderbot_small

# Scheduled Sampling models
from models.strat_blenderbot_small_ss import Model as strat_blenderbot_small_ss
from models.strat_blenderbot_small_lora import Model as strat_blenderbot_small_lora
from models.strat_blenderbot_small_lora_strat import Model as strat_blenderbot_small_lora_strat
from models.strat_blenderbot_small_lora_strat_s3 import Model as strat_blenderbot_small_lora_strat_s3
from models.strat_blenderbot_small_s3_nolora import Model as strat_blenderbot_small_s3_nolora

models = {

    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,

    # Scheduled Sampling models
    'strat_blenderbot_small_ss': strat_blenderbot_small_ss,

    # LoRA expert models
    'strat_blenderbot_small_lora_freeze': strat_blenderbot_small_lora,
    'strat_blenderbot_small_lora_train': strat_blenderbot_small_lora,
    'strat_blenderbot_small_lora_strat': strat_blenderbot_small_lora_strat,

    # S3: Joint Probe + Scheduled Sampling
    'strat_blenderbot_small_lora_strat_s3': strat_blenderbot_small_lora_strat_s3,

    # S3 ablation: -LoRA
    'strat_blenderbot_small_s3_nolora': strat_blenderbot_small_s3_nolora,
}
