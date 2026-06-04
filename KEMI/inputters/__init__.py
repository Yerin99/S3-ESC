
from inputters.strat import Inputter as strat
from inputters.strat_s3 import Inputter as strat_s3
from inputters.vanilla import Inputter as vanilla


inputters = {
    'vanilla': vanilla,
    'strat': strat,
    'strat_s3': strat_s3,
}



