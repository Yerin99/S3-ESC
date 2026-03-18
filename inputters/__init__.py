
from inputters.strat import Inputter as strat
from inputters.strat_lora import Inputter as strat_lora
from inputters.strat_lora_seq import Inputter as strat_lora_seq
from inputters.vanilla import Inputter as vanilla


inputters = {
    'vanilla': vanilla,
    'strat': strat,
    'strat_lora': strat_lora,
    'strat_lora_seq': strat_lora_seq,
}
