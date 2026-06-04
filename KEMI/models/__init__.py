
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.vanilla_blenderbot_small import Model as vanilla_blenderbot_small
from models.kemi_blenderbot_small import Model as kemi_blenderbot_small
from models.kemi_blenderbot_small_s3 import Model as kemi_blenderbot_small_s3


models = {

    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,
    'kemi_blenderbot_small': kemi_blenderbot_small,
    'kemi_blenderbot_small_s3': kemi_blenderbot_small_s3,

}