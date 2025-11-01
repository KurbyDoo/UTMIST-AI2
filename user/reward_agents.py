from environment.agent import *
from user.reward_functions import *


class BaseReward(RewardManager):
    def __init__(self):
        self.reward_functions = self._get_reward_functions()
        self.signal_subscriptions = self._get_signal_subscriptions()

        super().__init__(self.reward_functions, self.signal_subscriptions)

    def _get_reward_functions(self):
        """Override this method in subclasses to customize rewards"""
        return {
            'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=3.0),
            'key_spam': RewTerm(func=holding_more_than_3_keys, weight=0.5),
            'avoid_taunt': RewTerm(func=avoid_taunt, weight=10)
        }

    def _get_signal_subscriptions(self):
        """Override this method in subclasses to customize signals"""
        return {
            'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=100)),
            'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=20)),
        }


class BasicMovementCurriculum(BaseReward):
    def _get_reward_functions(self):
        return super()._get_reward_functions() | {
            'close_attack': RewTerm(func=reward_close_attack, weight=1.0),
            'conflict_movement': RewTerm(func=avoid_holding_opposite_keys, weight=0.1),
        }
    
    def _get_signal_subscriptions(self):
        return {}


class StopFallingCurriculum(BasicMovementCurriculum):
    def _get_reward_functions(self):
        return super()._get_reward_functions() | {
            'avoid_edge': RewTerm(func=edge_avoidance_reward, weight=3.0),
            'avoid_pit': RewTerm(func=pit_avoidance_reward, weight=0.1),
            'avoid_falling': RewTerm(func=avoid_falling_reward, weight=10.0),
            'avoid_ko': RewTerm(func=avoid_ko, weight=10.0),
            # 'in_air_reward': RewTerm(func=in_air_reward, weight=0.1),
        }
    
    def _get_signal_subscriptions(self):
        """Override this method in subclasses to customize signals"""
        return {
            'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=500)),
            'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=100)),
        }


class TowardsOpponentCurriculum(StopFallingCurriculum):
    def _get_reward_functions(self):
        return super()._get_reward_functions() | {
            'head_to_opponent': RewTerm(func=norm_op_dist, weight=20.0),
            'stationary_penalty': RewTerm(func=stand_still_penalty, weight=2.0),
            'moving_platform_reward': RewTerm(func=safe_moving_platform_reward, weight=1.0),
            'under_moving_platform': RewTerm(func=avoid_under_moving_platform, weight=50.0),
            'hit_opponent': RewTerm(func=damage_interaction_reward, weight=25.0),
        }

# StandStill Agent
# - Successfully stops falling off the map
# - Chooses to stop moving entirely
class StandStillReward(BaseReward):
    def _get_reward_functions(self):
        return super()._get_reward_functions() | {
            'in_air_reward':  RewTerm(func=in_air_reward, weight=0.01),
            'head_to_opponent':  RewTerm(func=head_to_opponent, weight=0.05)
        }


class TowardsOpponent(StandStillReward):
    def _get_reward_functions(self):
        return super()._get_reward_functions() | {
            'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
            'in_air_reward': RewTerm(func=in_air_reward, weight=0.001),
            'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.05),
        }


class AvoidFalling(TowardsOpponent):
    def _get_reward_functions(self):
        return super()._get_reward_functions() | {
            'avoid_falling': RewTerm(func=avoid_falling_reward, weight=0.05),
            'head_to_opponent': RewTerm(func=norm_op_dist, weight=0.001),
        }
    
class BasicHit(AvoidFalling):
    def _get_reward_functions(self):
        return super()._get_reward_functions() | {
            'hit_opponent': RewTerm(func=damage_interaction_reward, weight=0.1),
        }