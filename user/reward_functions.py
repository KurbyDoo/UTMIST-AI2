from environment.agent import *

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''


def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2


class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2


def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height or abs(
        player.body.position.x) > 6.5 else 0.0

    return reward * env.dt


def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState] = BackDashState,
    is_penalty: bool = False
) -> float:
    """
    Applies a penalty for every time frame player is in a specific state.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return (reward * env.dt) * (-1 if is_penalty else 1)


def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward


def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    player_dir = player.body.position.x - player.prev_x
    opponent_dir = opponent.body.position.x - player.body.position.x

    return abs(player_dir) if player_dir * opponent_dir > 0 else -abs(player_dir)


def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return -env.dt
    return env.dt


def penalize_far_attack(env: WarehouseBrawl, max_attack_distance: float = 2.0) -> float:
    """
    Penalizes the player for attacking when far from the opponent.
    
    Args:
        env: The game environment
        max_attack_distance: Maximum distance where attacking is acceptable (in world units)
    
    Returns:
        float: Penalty if attacking while far, 0 otherwise
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    if isinstance(player.state, KOState) or isinstance(opponent.state, KOState):
        return 0.0

    is_attacking = isinstance(player.state, AttackState)
    if not is_attacking: return 0.0
    distance = abs(player.body.position.x - opponent.body.position.x)

    if distance > max_attack_distance:
        penalty = -(distance - max_attack_distance) / 5.0
        return max(penalty, -1.0)

    return 0.0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player': return 1.0
    else: return -1.0


def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player': return -1.0
    else: return 1.0


def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0


def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -30.0
    return 0.0


def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0


def in_air_reward(env: WarehouseBrawl) -> float:
    player_state = env.objects["player"].state
    if isinstance(player_state, InAirState):
        return -0.5
    return 1


def avoid_falling_reward(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]

    if isinstance(player.state, KOState):
        return 0.0

    p_x = player.body.position.x
    if abs(p_x) >= 7.0:
        return -1.0 * env.dt
    
    elif abs(player.body.position.x) < 2.0:
        platform1: Stage = env.objects['platform1']
        # if abs(platform.body.position.x - player.body.position.x) > 0.5:
        #     reward += 1.0 if p_vx * platform.velocity_x > 0 else -2.0
        
        if platform1.body.position.y < player.body.position.y:
            return -1.0 * env.dt
    return 0
    

def norm_op_dist(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    if isinstance(player.state, KOState) or isinstance(opponent.state, KOState):
        return 0.0

    max_dist = 13
    x_diff = abs(opponent.body.position.x - player.body.position.x)
    y_diff = abs(opponent.body.position.y - player.body.position.y)

    return -(x_diff + y_diff) * env.dt / max_dist

        
### Curriculum rewards
def pit_avoidance_reward(env: WarehouseBrawl) -> float:
    """Applies a continuous, strong penalty for being in the center pit zone."""
    player: Player = env.objects["player"]
    if isinstance(player.state, KOState):
        return 0.0

    # The pit is roughly between x=-2.5 and x=2.5
    if abs(player.body.position.x) < 2.5:
        return -1.0 * env.dt
    return 0.0

def edge_avoidance_reward(env: WarehouseBrawl) -> float:
    """Applies a continuous, strong penalty for being near the edges."""
    player: Player = env.objects["player"]
    if isinstance(player.state, KOState):
        return 0.0

    # The pit is roughly between x=-2.5 and x=2.5
    if abs(player.body.position.x) > 7.0:
        return -1.0 * env.dt

    return 0.0

def safe_moving_platform_reward(env: WarehouseBrawl) -> float:
    """Give reward for being on the moving platform"""
    player: Player = env.objects["player"]
    platform: Stage = env.objects["platform1"]

    # if player is on moving platform
    if abs(player.body.position.x - platform.body.position.x) < 1 and player.body.position.y < platform.body.position.y:
        return 1.0 * env.dt
    return 0.0

def avoid_under_moving_platform(env: WarehouseBrawl) -> float:
    """Punish being under the moving platform"""
    player: Player = env.objects["player"]
    platform: Stage = env.objects["platform1"]

    # if player is under moving platform
    if abs(player.body.position.x - platform.body.position.x) < 1 and player.body.position.y > platform.body.position.y:
        return -1.0 * env.dt
    return 0.0
        
def avoid_ko(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    if isinstance(player.state, KOState):
        return -1.0
    return 0.0

def safe_platform_reward(env: WarehouseBrawl) -> float:
    """Gives a positive reward for being on the two main platforms."""
    player: Player = env.objects["player"]
    if isinstance(player.state, KOState):
        return 0.0

    x_pos = abs(player.body.position.x)
    # The safe zones are roughly between 2.5 and 5.5 on either side
    if 2.5 <= x_pos <= 5.5:
        return 0.5 * env.dt
    return 0.0

def avoid_taunt(env: WarehouseBrawl) -> float:
    """Taunting blocks all other inputs creating noise."""

    player : Player = env.objects["player"]

    if player.input.key_status['g'].just_pressed:
        return -1.0

    return 0.0
    # return in_state_reward(env, TauntState, True)

# def avoid_bad_dash(env: WarehouseBrawl) -> float:

#     player : Player = env.objects["player"]

#     if player.input.key_status



def avoid_holding_opposite_keys(env: WarehouseBrawl) -> float:
    """Stop agent from holding 'A' and 'D' keys at the same time"""
    player: Player = env.objects["player"]
    actions = player.cur_action

    # Actions are [Move Left, Move Right, Jump, Attack, Special, Taunt]
    # We assume a threshold of 0.5 to consider a key "pressed"
    move_left_pressed = actions[0] > 0.5
    move_right_pressed = actions[1] > 0.5

    if move_left_pressed and move_right_pressed:
        # Apply a penalty for each frame this occurs
        return -1.0 * env.dt
    
    # if move_left_pressed or move_right_pressed:
    #     return 1.0 * env.dt

    return 0.0
### Stage 2

def stand_still_penalty(env: WarehouseBrawl) -> float:
    """Avoid standing still."""
    return in_state_reward(env, StandingState, True)
