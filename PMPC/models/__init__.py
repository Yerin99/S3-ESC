
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.vanilla_blenderbot_small import Model as vanilla_blenderbot_small

# PMPC Model
from models.pmpc_blenderbot_small import Model as pmpc_blenderbot_small
from models.pmpc_blenderbot_small_s3 import Model as pmpc_blenderbot_small_s3

models = {

    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,

    # PMPC
    'pmpc_blenderbot_small': pmpc_blenderbot_small,
    'pmpc_blenderbot_small_s3': pmpc_blenderbot_small_s3,
}
