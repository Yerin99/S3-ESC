
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.dkfpe_blenderbot_small import Model as dkfpe_blenderbot_small
from models.dkfpe_blenderbot_small_s3 import Model as dkfpe_blenderbot_small_s3

models = {
    'strat_blenderbot_small': strat_blenderbot_small,
    'dkfpe_blenderbot_small': dkfpe_blenderbot_small,
    'dkfpe_blenderbot_small_s3': dkfpe_blenderbot_small_s3,
}
