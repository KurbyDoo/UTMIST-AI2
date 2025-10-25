from environment.agent import *
from reward_functions import *

# StandStill Agent
# - Successfully stops falling off the map
# - Chooses to stop moving entirely
class StandStillReward(RewardManager):
    def __init__():
        reward_functions = {
            # 'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
            'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
            # 'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=1.0),
            # 'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
            'in_air_reward': RewTerm(func=in_air_reward, weight=0.01),
            'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
            # 'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
            # 'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.01),
            # 'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
        }
        signal_subscriptions = {
            'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
            'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=20)),
            'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=5)),
            # 'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=10)),
            # 'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=10))
        }
        super(reward_functions, signal_subscriptions)
