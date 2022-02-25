import dataclasses
import time
# MAX_MEMORY = 3000
# BATCH_SIZE = 300
# LR = 0.0003
# EPSILON = 0.0
# GAMMA = 0.9
# Rep = 0
# SAVE = 400
# SHOW = 200

@dataclasses.dataclass
class TrainParam:
    buffer_size:int = 5000
    batch_size:int = 512
    warmup_size:int = 2048
    learning_rate:float = 3e-4
    epsilon:float = 0.1
    gamma:float = 0.9
    rep:int = 0
    save_interval:int = 4000
    evaluate_interval:int = 200



@dataclasses.dataclass
class EnvParam:
    seed:int = 3
    x_min:int = 0
    y_min:int = 0
    x_max:int = 15
    y_max:int = 15


    obs_num:int = 20
    action_num:int = 4


@dataclasses.dataclass
class DQNParam:
    hidden_size:list = dataclasses.field(default_factory=lambda:[64, 64])
    input_size:int = EnvParam.obs_num
    output_size:int = EnvParam.action_num

