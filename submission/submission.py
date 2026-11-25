from layer05_agent import create_agent as layer05
from contract import Overlay as Stage0
from independent_edits import Overlay as Stage1
from grain_delta import Overlay as Stage2
from scope_guard import Overlay as Stage3
from conditions import Overlay as Stage4


def create_agent(checkpoint=None):
    return Stage4(Stage3(Stage2(Stage1(Stage0(layer05(checkpoint))))))
