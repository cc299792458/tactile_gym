from tactile_gym.rl_envs.demo_rl_env_base import demo_rl_env
from tactile_gym.rl_envs.nonprehensile_manipulation.object_balance.object_balance_env import (
    ObjectBalanceEnv,
)


def main():

    seed = int(0)
    num_iter = 10
    max_steps = 10000
    show_gui = True
    show_tactile = False
    render = True
    print_info = False
    image_size = [256, 256]
    env_modes = {
        # which dofs can have movement (environment dependent)
        "movement_mode": "xy",
        # 'movement_mode':'xyz',
        # 'movement_mode':'RxRy',
        # 'movement_mode':'xyRxRy',

        # specify arm
        "arm_type": "ur5",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",


        # whether to use spin plate on initialisation
        "object_mode": "pole",
        # 'object_mode':'ball_on_plate',
        # 'object_mode':'spinning_plate',

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # add variation to joint force for rigid core
        "rand_gravity": False,
        "rand_embed_dist": True,

        # which observation type to return
        'observation_mode': 'oracle',
        # "observation_mode": "tactile",
        # 'observation_mode':'visual',
        # 'observation_mode':'visuotactile',

        # the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    }

    env = ObjectBalanceEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seed for deterministic results
    env.seed(seed)
    # env.action_space.np_random.seed(seed)

    # create controllable parameters on GUI
    action_ids = []
    min_action = env.min_action
    max_action = env.max_action

    if show_gui:
        if env_modes["movement_mode"] == "xy":
            action_ids.append(env._pb.addUserDebugParameter("dx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "xyz":
            action_ids.append(env._pb.addUserDebugParameter("dx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dz", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "RxRy":
            action_ids.append(env._pb.addUserDebugParameter("dRx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRy", min_action, max_action, 0))

        elif env_modes["movement_mode"] == "xyRxRy":
            action_ids.append(env._pb.addUserDebugParameter("dx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dy", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRx", min_action, max_action, 0))
            action_ids.append(env._pb.addUserDebugParameter("dRy", min_action, max_action, 0))

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    main()
