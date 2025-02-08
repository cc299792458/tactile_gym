import os
import cv2
import sys
import gym
import time
import json
import shutil
import pkgutil
import argparse
import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as bc

from tactile_gym.robots.arms.ur5.ur5 import UR5
from tactile_gym.robots.arms.mg400.mg400 import MG400
from tactile_gym.robots.arms.kuka_iiwa.kuka_iiwa import KukaIiwa
from tactile_gym.robots.arms.franka_panda.franka_panda import FrankaPanda

_ASSET = '/home/chichu/Documents/tactile_gym/tactile_gym/assets/'

float_formatter = "{:.6f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})

env_modes_default = {
    "movement_mode": "xyzRxRyRz",
    "control_mode": "TCP_velocity_control",
    "observation_mode": "oracle",
    "reward_mode": "dense",
}

rest_poses_dict = {
    "ur5": {
        "tactip": {
            "standard": np.array([
                0.00,  # world_joint         (fixed)
                0.166827,  # base_joint       (revolute)
                -2.16515,  # shoulder_joint  (revolute)
                -1.64365,  # elbow_joint     (revolute)
                -0.90317,  # wrist_1_joint   (revolute)
                1.57315,  # wrist_2_joint    (revolute)
                1.74001,  # wrist_3_joint    (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00   # tcp_joint           (fixed)
            ]),
        },
        "digit": {
            "standard": np.array([
                0.00,  # world_joint         (fixed)
                0.1666452116249431,  # base_joint       (revolute)
                -2.2334888481855204,  # shoulder_joint  (revolute)
                -1.6642245054428424,  # elbow_joint     (revolute)
                -0.8142762445463524,  # wrist_1_joint   (revolute)
                1.573151527964482,  # wrist_2_joint    (revolute)
                1.7398309441833082,  # wrist_3_joint    (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00   # tcp_joint           (fixed)
            ]),
        },
        "digitac": {
            "standard": np.array([
                0.00,  # world_joint         (fixed)
                0.16664443404149898,  # base_joint       (revolute)
                -2.2242489977536737,  # shoulder_joint  (revolute)
                -1.6618744232210114,  # elbow_joint     (revolute)
                -0.8258663681806591,  # wrist_1_joint   (revolute)
                1.5731514988184077,  # wrist_2_joint    (revolute)
                1.7398302172182332,  # wrist_3_joint    (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00   # tcp_joint           (fixed)
            ]),
        },
    },
    "kuka_iiwa": {
        "tactip": {
            "standard": np.array([
                0.00,  # world_joint          (fixed)
                0.27440,  # lbr_iiwa_joint_1  (revolute)
                1.20953,  # lbr_iiwa_joint_2  (revolute)
                2.66025,  # lbr_iiwa_joint_3  (revolute)
                1.26333,  # lbr_iiwa_joint_4  (revolute)
                -2.50256,  # lbr_iiwa_joint_5 (revolute)
                0.81968,  # lbr_iiwa_joint_6  (revolute)
                2.76347,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00   # tcp_joint            (fixed)
            ])
        },
        "digit": {
            "standard": np.array([
                0.00,  # world_joint          (fixed)
                0.27440,  # lbr_iiwa_joint_1  (revolute)
                1.20953,  # lbr_iiwa_joint_2  (revolute)
                2.66025,  # lbr_iiwa_joint_3  (revolute)
                1.26333,  # lbr_iiwa_joint_4  (revolute)
                -2.50256,  # lbr_iiwa_joint_5 (revolute)
                0.81968,  # lbr_iiwa_joint_6  (revolute)
                2.76347,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00   # tcp_joint            (fixed)
            ])
        },
        "digitac": {
            "standard": np.array([
                0.00,  # world_joint          (fixed)
                0.27440,  # lbr_iiwa_joint_1  (revolute)
                1.20953,  # lbr_iiwa_joint_2  (revolute)
                2.66025,  # lbr_iiwa_joint_3  (revolute)
                1.26333,  # lbr_iiwa_joint_4  (revolute)
                -2.50256,  # lbr_iiwa_joint_5 (revolute)
                0.81968,  # lbr_iiwa_joint_6  (revolute)
                2.76347,  # lbr_iiwa_joint_7  (revolute)
                0.00,  # ee_joint             (fixed)
                0.00,  # tactip_ee_joint      (fixed)
                0.00,  # tactip_tip_to_body  (fixed)
                0.00   # tcp_joint            (fixed)
            ])
        },
    },
    "franka_panda": {
        "tactip": {
            "standard": np.array([
                0.00,  # world_joint         (fixed)
                3.21456,  # panda_joint1     (revolute)
                1.30233,  # panda_joint2     (revolute)
                2.99673,  # panda_joint3     (revolute)
                0.83832,  # panda_joint4     (revolute)
                -2.97647,  # panda_joint5    (revolute)
                2.13176,  # panda_joint6     (revolute)
                4.65986,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00   # tcp_joint           (fixed)
            ]),
        },
        "digit": {
            "standard": np.array([
                0.00,  # world_joint         (fixed)
                3.21456,  # panda_joint1     (revolute)
                1.30233,  # panda_joint2     (revolute)
                2.99673,  # panda_joint3     (revolute)
                0.83832,  # panda_joint4     (revolute)
                -2.97647,  # panda_joint5    (revolute)
                2.13176,  # panda_joint6     (revolute)
                4.65986,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00   # tcp_joint           (fixed)
            ]),
        },
        "digitac": {
            "standard": np.array([
                0.00,  # world_joint         (fixed)
                3.21456,  # panda_joint1     (revolute)
                1.30233,  # panda_joint2     (revolute)
                2.99673,  # panda_joint3     (revolute)
                0.83832,  # panda_joint4     (revolute)
                -2.97647,  # panda_joint5    (revolute)
                2.13176,  # panda_joint6     (revolute)
                4.65986,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00   # tcp_joint           (fixed)
            ]),
        },
    },
    "mg400": {
        "tactip": {
            "standard": np.array([
                0,                      # j1        (fixed)
                1.1199979523765513,     # j2_1         (revolute)
                -0.027746434948259045,  # j3_1         (revolute)
                -1.094390587897371,     # j4_1          (revolute)
                0.000795099112695166,   # j5          (revolute)
                0,                      # ee_joint           (fixed)
                0,                      # tactip_ee_joint           (fixed)
                0,                      # tactip_tip_to_body    (fixed)
                0,                      # tcp_joint (fixed)
                1.120002713232204,      # j2_2 = j2_1         (revolute)
                -1.1199729024887553,   # j3_2 = -j2_1         (revolute)
                1.0922685386653785      # j4_2 = j2_1 + j3_1          (revolute)
            ]),
        },
        "digit": {
            "standard": np.array([
                0,                      # j1        (fixed)
                1.3190166816731614,     # j2_1         (revolute)
                -0.057932730559221525,  # j3_1         (revolute)
                -1.2611243932983605,     # j4_1          (revolute)
                0.0006084288058448784,   # j5          (revolute)
                0,                      # ee_joint           (fixed)
                0,                      # tactip_ee_joint           (fixed)
                0,                      # tactip_tip_to_body    (fixed)
                0,                      # tcp_joint (fixed)
                1.3190195840338783,      # j2_2 = j2_1         (revolute)
                -1.3189925313906967,   # j3_2 = -j2_1         (revolute)
                1.2610906509351185      # j4_2 = j2_1 + j3_1          (revolute)
            ]),
        },
        "digitac": {
            "standard": np.array([
                0,                      # j1        (fixed)
                1.3223687315585777,     # j2_1         (revolute)
                -0.06290495221125363,  # j3_1         (revolute)
                -1.2594762221064615,     # j4_1          (revolute)
                0.0006084288058448784,   # j5          (revolute)
                0,                      # ee_joint           (fixed)
                0,                      # tactip_ee_joint           (fixed)
                0,                      # tactip_tip_to_body    (fixed)
                0,                      # tcp_joint (fixed)
                1.3223720640647498,      # j2_2 = j2_1         (revolute)
                -1.3223720640647498,   # j3_2 = -j2_1         (revolute)
                1.2594757646221153      # j4_2 = j2_1 + j3_1          (revolute)
            ]),
        },
    }
}

def empty_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def check_dir(dir):

    # check save dir exists
    if os.path.isdir(dir):
        str_input = input("Save Directory already exists, would you like to continue (y,n)? ")
        if not str2bool(str_input):
            exit()
        else:
            # clear out existing files
            empty_dir(dir)

def add_assets_path(path):
    return os.path.join(_ASSET, path)

def demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info=False):
    """
    Control loop for demonstrating an RL environment.
    Use show_gui and show_tactile flags for visualising and controlling the env.
    Use render for more direct info on what the agent will see.
    """
    record = False
    if record:
        import imageio

        render_frames = []
        log_id = env._pb.startStateLogging(
            loggingType=env._pb.STATE_LOGGING_VIDEO_MP4, fileName=os.path.join("example_videos", "gui.mp4")
        )

    # collection loop
    for i in range(num_iter):
        r_sum = 0
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        step = 0

        while not d:

            if show_gui:
                a = []
                for action_id in action_ids:
                    a.append(env._pb.readUserDebugParameter(action_id))
            else:
                a = env.action_space.sample()

            # step the environment
            o, r, d, info = env.step(a)

            if print_info:
                print("")
                print("Step: ", step)
                print("Act:  ", a)
                print("Obs:  ")
                for key, value in o.items():
                    if value is None:
                        print("  ", key, ":", value)
                    else:
                        print("  ", key, ":", value.shape)
                print("Rew:  ", r)
                print("Done: ", d)

            # render visual + tactile observation
            if render:
                render_img = env.render()
                if record:
                    render_frames.append(render_img)

            r_sum += r
            step += 1

            q_key = ord("q")
            r_key = ord("r")
            keys = env._pb.getKeyboardEvents()
            if q_key in keys and keys[q_key] & env._pb.KEY_WAS_TRIGGERED:
                exit()
            elif r_key in keys and keys[r_key] & env._pb.KEY_WAS_TRIGGERED:
                d = True

        print("Total Reward: ", r_sum)

    if record:
        env._pb.stopStateLogging(log_id)
        imageio.mimwrite(os.path.join("example_videos", "render.mp4"), np.stack(render_frames), fps=12)

    env.close()

#######################################################################################################################################
#######################################################################################################################################

class BaseTactileEnv(gym.Env):
    def __init__(self, max_steps=250, image_size=[64, 64], show_gui=False, show_tactile=False, arm_type='ur5'):

        # set seed
        self.seed()

        # env params
        self._observation = []
        self._env_step_counter = 0
        self._max_steps = max_steps
        self._image_size = image_size
        self._show_gui = show_gui
        self._show_tactile = show_tactile
        self._first_render = True
        self._render_closed = False
        self.arm_type = arm_type
        # set up camera for rgb obs and debbugger
        self.setup_rgb_obs_camera_params()

        self.connect_pybullet()

        # set vars for full pybullet reset to clear cache
        self.reset_counter = 0
        self.reset_limit = 1000

    def connect_pybullet(self):
        # render the environment
        if self._show_gui:
            self._pb = bc.BulletClient(connection_mode=pb.GUI)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            self._pb.resetDebugVisualizerCamera(
                self.rgb_cam_dist,
                self.rgb_cam_yaw,
                self.rgb_cam_pitch,
                self.rgb_cam_pos,
            )
        else:
            self._pb = bc.BulletClient(connection_mode=pb.DIRECT)
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                self._pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                self._pb.loadPlugin("eglRendererPlugin")

        # bc automatically sets client but keep here incase needed
        self._physics_client_id = self._pb._client

    def seed(self, seed=None):
        self._seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __del__(self):
        self.close()

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()

        if not self._render_closed:
            cv2.destroyAllWindows()

    def setup_observation_space(self):

        obs_dim_dict = self.get_obs_dim()

        spaces = {
            "oracle": gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_dim_dict["oracle"], dtype=np.float32),
            "tactile": gym.spaces.Box(low=0, high=255, shape=obs_dim_dict["tactile"], dtype=np.uint8),
            "visual": gym.spaces.Box(low=0, high=255, shape=obs_dim_dict["visual"], dtype=np.uint8),
            "extended_feature": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim_dict["extended_feature"], dtype=np.float32
            ),
        }

        if self.observation_mode == "oracle":
            self.observation_space = gym.spaces.Dict({"oracle": spaces["oracle"]})

        elif self.observation_mode == "tactile":
            self.observation_space = gym.spaces.Dict({"tactile": spaces["tactile"]})

        elif self.observation_mode == "visual":
            self.observation_space = gym.spaces.Dict({"visual": spaces["visual"]})

        elif self.observation_mode == "visuotactile":
            self.observation_space = gym.spaces.Dict({"tactile": spaces["tactile"], "visual": spaces["visual"]})

        elif self.observation_mode == "tactile_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"tactile": spaces["tactile"], "extended_feature": spaces["extended_feature"]}
            )

        elif self.observation_mode == "visual_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"visual": spaces["visual"], "extended_feature": spaces["extended_feature"]}
            )

        elif self.observation_mode == "visuotactile_and_feature":
            self.observation_space = gym.spaces.Dict(
                {"tactile": spaces["tactile"], "visual": spaces["visual"], "extended_feature": spaces["extended_feature"]}
            )

    def get_obs_dim(self):
        obs_dim_dict = {
            "oracle": self.get_oracle_obs().shape,
            "tactile": self.get_tactile_obs().shape,
            "visual": self.get_visual_obs().shape,
            "extended_feature": self.get_extended_feature_array().shape,
        }
        return obs_dim_dict

    def load_environment(self):

        self._pb.setGravity(0, 0, -9.81)
        self._pb.setPhysicsEngineParameter(
            fixedTimeStep=self._sim_time_step, numSolverIterations=150, enableConeFriction=1, contactBreakingThreshold=0.0001
        )
        self.plane_id = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/plane/plane.urdf"),
            [0, 0, -0.625],
        )
        self.table_id = self._pb.loadURDF(
            add_assets_path("shared_assets/environment_objects/table/table.urdf"),
            [0.50, 0.00, -0.625],
            [0.0, 0.0, 0.0, 1.0],
        )

    def scale_actions(self, actions):

        # would prefer to enforce action bounds on algorithm side, but this is ok for now
        actions = np.clip(actions, self.min_action, self.max_action)

        input_range = self.max_action - self.min_action

        new_x_range = self.x_act_max - self.x_act_min
        new_y_range = self.y_act_max - self.y_act_min
        new_z_range = self.z_act_max - self.z_act_min
        new_roll_range = self.roll_act_max - self.roll_act_min
        new_pitch_range = self.pitch_act_max - self.pitch_act_min
        new_yaw_range = self.yaw_act_max - self.yaw_act_min

        scaled_actions = [
            (((actions[0] - self.min_action) * new_x_range) / input_range) + self.x_act_min,
            (((actions[1] - self.min_action) * new_y_range) / input_range) + self.y_act_min,
            (((actions[2] - self.min_action) * new_z_range) / input_range) + self.z_act_min,
            (((actions[3] - self.min_action) * new_roll_range) / input_range) + self.roll_act_min,
            (((actions[4] - self.min_action) * new_pitch_range) / input_range) + self.pitch_act_min,
            (((actions[5] - self.min_action) * new_yaw_range) / input_range) + self.yaw_act_min,
        ]

        return np.array(scaled_actions)

    def step(self, action):

        # scale and embed actions appropriately
        encoded_actions = self.encode_actions(action)
        scaled_actions = self.scale_actions(encoded_actions)

        self._env_step_counter += 1

        self.robot.apply_action(
            scaled_actions,
            control_mode=self.control_mode,
            velocity_action_repeat=self._velocity_action_repeat,
            max_steps=self._max_blocking_pos_move_steps,
        )

        reward, done = self.get_step_data()

        self._observation = self.get_observation()

        return self._observation, reward, done, {}

    def get_extended_feature_array(self):
        """
        Get feature to extend current observations.
        """
        return np.array([])

    def get_oracle_obs(self):
        """
        Use for sanity checking, no tactile observation just features that should
        be enough to learn reasonable policies.
        """
        return np.array([])

    def get_tactile_obs(self):
        """
        Returns the tactile observation with an image channel.
        """

        # get image from sensor
        tactile_obs = self.robot.get_tactile_observation()

        observation = tactile_obs[..., np.newaxis]

        return observation

    def get_visual_obs(self):
        """
        Returns the rgb image from an environment camera.
        """
        # get an rgb image that matches the debug visualiser
        view_matrix = self._pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.rgb_cam_pos,
            distance=self.rgb_cam_dist,
            yaw=self.rgb_cam_yaw,
            pitch=self.rgb_cam_pitch,
            roll=0,
            upAxisIndex=2,
        )

        proj_matrix = self._pb.computeProjectionMatrixFOV(
            fov=self.rgb_fov,
            aspect=float(self.rgb_image_size[0]) / self.rgb_image_size[1],
            nearVal=self.rgb_near_val,
            farVal=self.rgb_far_val,
        )

        (_, _, px, _, _) = self._pb.getCameraImage(
            width=self.rgb_image_size[0],
            height=self.rgb_image_size[1],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.rgb_image_size[0], self.rgb_image_size[1], 4))

        observation = rgb_array[:, :, :3]
        return observation

    def get_observation(self):
        """
        Returns the observation dependent on which mode is set.
        """
        # init obs dict
        observation = {}

        # check correct obs type set
        if self.observation_mode not in [
            "oracle",
            "tactile",
            "visual",
            "visuotactile",
            "tactile_and_feature",
            "visual_and_feature",
            "visuotactile_and_feature",
        ]:
            sys.exit("Incorrect observation mode specified: {}".format(self.observation_mode))

        # use direct pose info to check if things are working
        if "oracle" in self.observation_mode:
            observation["oracle"] = self.get_oracle_obs()

        # observation is just the tactile sensor image
        if "tactile" in self.observation_mode:
            observation["tactile"] = self.get_tactile_obs()

        # observation is rgb environment camera image
        if any(obs in self.observation_mode for obs in ["visual", "visuo"]):
            observation["visual"] = self.get_visual_obs()

        # observation is mix image + features (pretending to be image shape)
        if "feature" in self.observation_mode:
            observation["extended_feature"] = self.get_extended_feature_array()

        return observation

    def render(self, mode="rgb_array"):
        """
        Most rendering handled with show_gui, show_tactile flags.
        This is useful for saving videos.
        """

        if mode != "rgb_array":
            return np.array([])

        # get the rgb camera image
        rgb_array = self.get_visual_obs()

        # get the current tactile images and reformat to match rgb array
        tactile_array = self.get_tactile_obs()
        tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)

        # resize tactile to match rgb if rendering in higher res
        if self._image_size != self.rgb_image_size:
            tactile_array = cv2.resize(tactile_array, tuple(self.rgb_image_size))

        # concat the images into a single image
        render_array = np.concatenate([rgb_array, tactile_array], axis=1)

        # setup plot for rendering
        if self._first_render:
            self._first_render = False
            if self._seed is not None:
                self.window_name = "render_window_{}".format(self._seed)
            else:
                self.window_name = "render_window"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # plot rendered image
        if not self._render_closed:
            render_array_rgb = cv2.cvtColor(render_array, cv2.COLOR_BGR2RGB)
            cv2.imshow(self.window_name, render_array_rgb)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(self.window_name)
                self._render_closed = True

        return render_array

class TactileSensor:
    def __init__(
        self,
        pb,
        robot_id,
        tactile_link_ids,
        image_size=[128, 128],
        turn_off_border=False,
        t_s_name='tactip',
        t_s_type="standard",
        t_s_core="no_core",
        t_s_dynamics={},
        show_tactile=True,
        t_s_num=int(0),
    ):

        self._pb = pb
        self.robot_id = robot_id
        self.tactile_link_ids = tactile_link_ids
        self._show_tactile = show_tactile
        self.t_s_name = t_s_name
        self.t_s_type = t_s_type
        self.t_s_core = t_s_core
        self.t_s_dynamics = t_s_dynamics
        self.image_size = image_size
        self.turn_off_border = turn_off_border
        self.t_s_num = t_s_num

        self.load_reference_images()
        self.setup_camera_info()
        # self.save_reference_images()
        self.connect_t_s()

        # if self.t_s_type in ["standard", "mini_standard", "flat", "right_angle"]:
        if self.t_s_name in ["tactip", "digit", "digitac"]:
            self.turn_off_t_s_collisions()

    def turn_off_t_s_collisions(self):
        """
        Turn off collisions between t_s base and rest of the envs,
        useful for speed of training due to mininmising collisions
        """
        self._pb.setCollisionFilterGroupMask(self.robot_id, self.tactile_link_ids["body"], 0, 0)
        if self.t_s_name == 'tactip':
            if self.t_s_type in ["right_angle", "mini_right_angle", "forward"]:
                self._pb.setCollisionFilterGroupMask(self.robot_id, self.tactile_link_ids["adapter"], 0, 0)

        if self.t_s_core == "no_core":
            self._pb.setCollisionFilterGroupMask(self.robot_id, self.tactile_link_ids["tip"], 0, 0)

        # if self.t_s_name == "digit":
        #     self._pb.setCollisionFilterGroupMask(
        #         self.robot_id, self.tactip_link_ids['mask'], 0, 0)

    def load_reference_images(self):
        # get saved reference images
        border_images_path = add_assets_path(os.path.join("robot_assets", self.t_s_name, "reference_images"))

        saved_file_dir = os.path.join(
            border_images_path,
            self.t_s_type,
            str(self.image_size[0]) + "x" + str(self.image_size[0]),
        )

        nodef_gray_savefile = os.path.join(saved_file_dir, "nodef_gray.npy")
        nodef_dep_savefile = os.path.join(saved_file_dir, "nodef_dep.npy")
        border_mask_savefile = os.path.join(saved_file_dir, "border_mask.npy")

        # load border images from simulation
        self.no_deformation_gray = np.load(nodef_gray_savefile)
        self.no_deformation_dep = np.load(nodef_dep_savefile)
        self.border_mask = np.load(border_mask_savefile)

        # plt the reference images for checking
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(self.no_deformation_gray, cmap='gray')
        # axs[1].imshow(self.no_deformation_dep, cmap='gray')
        # axs[2].imshow(self.border_mask, cmap='gray')
        # plt.show(block=True)
        # exit()

    def save_reference_images(self):

        # grab images for creating border from simulation
        no_deformation_rgb, no_deformation_dep, no_deformation_mask = self.get_imgs()
        no_deformation_gray = cv2.cvtColor(no_deformation_rgb.astype(np.float32), cv2.COLOR_BGR2GRAY)

        # convert mask from link/base ids to ones/zeros for border/not border
        mask_base_id = no_deformation_mask & ((1 << 24) - 1)
        mask_link_id = (no_deformation_mask >> 24) - 1
        border_mask = (mask_base_id == self.robot_id) & (mask_link_id == self.tactile_link_ids["body"]).astype(np.uint8)

        # create save file
        border_images_path = add_assets_path(os.path.join("robot_assets", self.t_s_name, "reference_images"))

        saved_file_dir = os.path.join(
            border_images_path,
            self.t_s_type,
            str(self.image_size[0]) + "x" + str(self.image_size[0]),
        )

        # create new directory
        check_dir(saved_file_dir)
        os.makedirs(saved_file_dir, exist_ok=True)

        # save file names
        nodef_gray_savefile = os.path.join(saved_file_dir, "nodef_gray.npy")
        nodef_dep_savefile = os.path.join(saved_file_dir, "nodef_dep.npy")
        border_mask_savefile = os.path.join(saved_file_dir, "border_mask.npy")

        # save border images from simulation
        np.save(nodef_gray_savefile, no_deformation_gray)
        np.save(nodef_dep_savefile, no_deformation_dep)
        np.save(border_mask_savefile, border_mask)

        exit()

    def setup_camera_info(self):
        """
        set parameters that define images from internal camera.
        """
        if self.t_s_name == 'tactip':
            if self.t_s_type in ["standard", "mini_standard", "flat", "right_angle", "mini_right_angle", "forward"]:
                self.focal_dist = 0.065
                self.fov = 60

        elif self.t_s_name == 'digit':
            if self.t_s_type in ["standard", "right_angle", "forward"]:
                self.focal_dist = 0.0015
                self.fov = 40
        elif self.t_s_name == 'digitac':
            if self.t_s_type in ["standard", "right_angle", "forward"]:
                self.focal_dist = 0.0015
                self.fov = 40

        self.pixel_width, self.pixel_height = self.image_size[0], self.image_size[1]
        self.aspect, self.nearplane, self.farplane = 1.0, 0.01, 1.0
        self.focal_length = 1.0 / (2 * np.tan((self.fov * (np.pi / 180)) / 2))  # not used but useful to know
        self.projection_matrix = self._pb.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearplane, self.farplane)

    def update_cam_frame(self):

        # get the pose of the t_s body (where camera sits)
        t_s_body_pos, t_s_body_orn, _, _, _, _ = self._pb.getLinkState(
            self.robot_id, self.tactile_link_ids["body"], computeForwardKinematics=True
        )

        # set camera position relative to the t_s body
        if self.t_s_name == 'tactip':
            if self.t_s_type in ["standard", "mini_standard", "flat"]:
                cam_pos = (0, 0, 0.03)
                cam_rpy = (0, -np.pi / 2, np.pi)
            elif self.t_s_type in ["right_angle", "forward"]:
                cam_pos = (0, 0, 0.03)
                cam_rpy = (0, -np.pi / 2, 140 * np.pi / 180)
            elif self.t_s_type in ["mini_right_angle"]:
                cam_pos = (0, 0, 0.001)
                cam_rpy = (0, -np.pi / 2, 140 * np.pi / 180)

        elif self.t_s_name == 'digit':
            if self.t_s_type in ["standard"]:
                cam_pos = (-0.00095, .0139, 0.020)
                cam_rpy = (np.pi, -np.pi/2, np.pi/2)
            elif self.t_s_type in ["right_angle", "forward"]:
                cam_pos = (-0.00095, .0139, 0.005)
                cam_rpy = (np.pi, -np.pi/2, np.pi/2)
        elif self.t_s_name == 'digitac':
            if self.t_s_type in ["standard"]:
                cam_pos = (-0.00095, .0139, 0.020)
                cam_rpy = (np.pi, -np.pi/2, np.pi/2)
            elif self.t_s_type in ["right_angle", "forward"]:
                cam_pos = (-0.00095, .0139, 0.005)
                cam_rpy = (np.pi, -np.pi/2, np.pi/2)

        cam_orn = self._pb.getQuaternionFromEuler(cam_rpy)

        # get the camera frame relative to world frame
        self.camframe_pos, self.camframe_orn = self._pb.multiplyTransforms(t_s_body_pos, t_s_body_orn, cam_pos, cam_orn)

    def camframe_to_worldframe(self, pos, rpy):
        """
        Transforms a pose in camera frame to a pose in world frame.
        """
        pos = np.array(pos)
        rpy = np.array(rpy)
        orn = np.array(self._pb.getQuaternionFromEuler(rpy))

        worldframe_pos, worldframe_orn = self._pb.multiplyTransforms(self.camframe_pos, self.camframe_orn, pos, orn)
        worldframe_rpy = self._pb.getEulerFromQuaternion(worldframe_orn)

        return np.array(worldframe_pos), np.array(worldframe_rpy)

    def camvec_to_worldvec(self, camframe_vec):
        """
        Transforms a vector in work frame to a vector in world frame.
        """
        camframe_vec = np.array(camframe_vec)
        rot_matrix = np.array(self._pb.getMatrixFromQuaternion(self.camframe_orn)).reshape(3, 3)
        worldframe_vec = rot_matrix.dot(camframe_vec)

        return np.array(worldframe_vec)

    def get_imgs(self):
        """
        Pull some images from the synthetic camera
        """

        # update the camera frame
        self.update_cam_frame()

        # calculate view matrix
        foward_vector = self.camvec_to_worldvec([1, 0, 0])
        up_vector = self.camvec_to_worldvec([0, 0, 1])
        cam_target_pos = self.camframe_pos + self.focal_dist * np.array(foward_vector)

        view_matrix = self._pb.computeViewMatrix(
            self.camframe_pos,
            cam_target_pos,
            up_vector,
        )

        # draw a line at these points for debugging
        # extended_cam_pos = self.camframe_pos + np.array(foward_vector)
        # extended_up_pos  = self.camframe_pos + np.array(up_vector)
        # self._pb.addUserDebugLine(self.camframe_pos, extended_cam_pos, [0, 1, 1], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)
        # self._pb.addUserDebugLine(self.camframe_pos, extended_up_pos, [1, 0, 1], parentObjectUniqueId=-1, parentLinkIndex=-1, lifeTime=0.1)

        # projective texture runs faster but gives odd visuals
        flags = self._pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        img_arr = self._pb.getCameraImage(
            self.pixel_width,
            self.pixel_height,
            view_matrix,
            self.projection_matrix,
            renderer=self._pb.ER_BULLET_HARDWARE_OPENGL,
            flags=flags,
        )

        # get images from returned array
        w = img_arr[0]  # width of the image, in pixels
        h = img_arr[1]  # height of the image, in pixels
        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth dataes
        mask = img_arr[4]  # mask dataes

        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        mask = np.reshape(mask, (h, w))

        return rgb, dep, mask

    def t_s_camera(self):
        """
        Pull some images from the synthetic camera and manipulate them to become
        tacitle images.
        """

        # get the current images
        _, cur_dep, cur_mask = self.get_imgs()

        # get the difference between current images and undeformed counterparts
        diff_dep = np.subtract(cur_dep, self.no_deformation_dep)

        # remove noise from depth image
        eps = 1e-4
        diff_dep[(diff_dep >= -eps) & (diff_dep <= eps)] = 0

        # convert depth to penetration
        pen_img = np.abs(diff_dep)

        # convert dep to display format
        max_penetration = 0.05
        pen_img = ((np.clip(pen_img, 0, max_penetration) / max_penetration) * 255).astype(np.uint8)

        # reduce noise by setting all parts of the image where the t_s body is visible to zero
        mask_base_id = cur_mask & ((1 << 24) - 1)
        mask_link_id = (cur_mask >> 24) - 1
        full_mask = (mask_base_id == self.robot_id) & (mask_link_id == self.tactile_link_ids["body"])
        pen_img[full_mask] = 0

        # add border from ref image
        if not self.turn_off_border:
            pen_img[self.border_mask == 1] = self.no_deformation_gray[self.border_mask == 1]

        return pen_img

    def connect_t_s(self):
        """
        Setup plots if enabled.
        """
        # setup plot for rendering
        if self._show_tactile:
            cv2.namedWindow("tactile_window_{}".format(self.t_s_num), cv2.WINDOW_NORMAL)
            self._render_closed = False
        else:
            self._render_closed = True

    def reset(self):
        """
        Reset t_s
        """
        self.reset_tip()
        self.update_cam_frame()

    def reset_tip(self):
        """
        Reset the t_s core parameters here, could perform physics
        randomisations if required.
        """
        if self.t_s_core == "no_core":
            return None

        elif self.t_s_core == "fixed":
            # change dynamics
            self._pb.changeDynamics(
                self.robot_id,
                self.tactile_link_ids["tip"],
                contactDamping=self.t_s_dynamics["damping"],
                contactStiffness=self.t_s_dynamics["stiffness"],
            )
            self._pb.changeDynamics(
                self.robot_id, self.tactile_link_ids["tip"], lateralFriction=self.t_s_dynamics["friction"]
            )

    def process_sensor(self):
        """
        Return an image captured by the sensor.
        Also plot if enabled.
        """
        img = self.t_s_camera()
        # plot rendered image
        if not self._render_closed:
            cv2.imshow("tactile_window_{}".format(self.t_s_num), img)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow("tactile_window_{}".format(self.t_s_num))
                self._render_closed = True

        return img

    def get_observation(self):
        return self.process_sensor()

    def draw_camera_frame(self):
        rpy = [0, 0, 0]
        self._pb.addUserDebugLine(
            self.camframe_pos,
            self.camframe_to_worldframe([0.1, 0, 0], rpy)[0],
            [1, 0, 0],
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            self.camframe_pos,
            self.camframe_to_worldframe([0, 0.1, 0], rpy)[0],
            [0, 1, 0],
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            self.camframe_pos,
            self.camframe_to_worldframe([0, 0, 0.1], rpy)[0],
            [0, 0, 1],
            lifeTime=0.1,
        )

    def draw_t_s_frame(self):
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0.1, 0, 0],
            [1, 0, 0],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.tactile_link_ids["body"],
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0.1, 0],
            [0, 1, 0],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.tactile_link_ids["body"],
            lifeTime=0.1,
        )
        self._pb.addUserDebugLine(
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 1],
            parentObjectUniqueId=self.robot_id,
            parentLinkIndex=self.tactile_link_ids["body"],
            lifeTime=0.1,
        )

class Robot:
    def __init__(
        self,
        pb,
        rest_poses,
        workframe_pos,
        workframe_rpy,
        TCP_lims,
        image_size=[128, 128],
        turn_off_border=False,
        arm_type="ur5",
        t_s_name='tactip',
        t_s_type="standard",
        t_s_core="no_core",
        t_s_dynamics={},
        show_gui=True,
        show_tactile=True,
    ):

        self._pb = pb
        self.arm_type = arm_type
        self.t_s_name = t_s_name
        self.t_s_type = t_s_type
        self.t_s_core = t_s_core

        # load the urdf file
        self.robot_id = self.load_robot()
        if self.arm_type == "ur5":
            self.arm = UR5(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "franka_panda":
            self.arm = FrankaPanda(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "kuka_iiwa":
            self.arm = KukaIiwa(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        elif self.arm_type == "mg400":
            self.arm = MG400(
                pb, self.robot_id, rest_poses, workframe_pos, workframe_rpy, TCP_lims
            )

        else:
            sys.exit("Incorrect arm type specified {}".format(self.arm_type))

        # get relevent link ids for turning off collisions, connecting camera, etc
        tactile_link_ids = {}
        tactile_link_ids['body'] = self.arm.link_name_to_index[self.t_s_name+"_body_link"]
        tactile_link_ids['tip'] = self.arm.link_name_to_index[self.t_s_name+"_tip_link"]

        if t_s_type in ["right_angle", 'forward', 'mini_right_angle', 'mini_forward']:
            if self.t_s_name == 'tactip':
                tactile_link_ids['adapter'] = self.arm.link_name_to_index[
                    "tactip_adapter_link"
                ]
            elif self.t_s_name in ['digitac', 'digit']:
                print("TODO: Add the adpater link after get it into the URDF")

        # connect the sensor the tactip
        self.t_s = TactileSensor(
            pb,
            robot_id=self.robot_id,
            tactile_link_ids=tactile_link_ids,
            image_size=image_size,
            turn_off_border=turn_off_border,
            t_s_name=t_s_name,
            t_s_type=t_s_type,
            t_s_core=t_s_core,
            t_s_dynamics=t_s_dynamics,
            show_tactile=show_tactile,
            t_s_num=1
        )

    def load_robot(self):
        """
        Load the robot arm model into pybullet
        """
        self.base_pos = [0, 0, 0]
        self.base_rpy = [0, 0, 0]
        self.base_orn = self._pb.getQuaternionFromEuler(self.base_rpy)
        robot_urdf = add_assets_path(os.path.join(
            "robot_assets",
            self.arm_type,
            self.t_s_name,
            self.arm_type + "_with_" + self.t_s_type + "_" + self.t_s_name + ".urdf",
        ))
        robot_id = self._pb.loadURDF(
            robot_urdf, self.base_pos, self.base_orn, useFixedBase=True
        )

        return robot_id

    def reset(self, reset_TCP_pos, reset_TCP_rpy):
        """
        Reset the pose of the UR5 and t_s
        """
        self.arm.reset()
        self.t_s.reset()

        # move to the initial position
        self.arm.tcp_direct_workframe_move(reset_TCP_pos, reset_TCP_rpy)
        # print("TCP pos wrt work frame:",reset_TCP_pos)
        self.blocking_move(max_steps=1000, constant_vel=0.001)
        # self.arm.print_joint_pos_vel()

    def full_reset(self):
        self.load_robot()
        self.t_s.turn_off_t_s_collisions()

    def step_sim(self):
        """
        Take a step of the simulation whilst applying neccessary forces
        """

        # compensate for the effect of gravity
        # self.arm.draw_TCP() # only works with visuals enabled in urdf file
        self.arm.apply_gravity_compensation()

        # step the simulation
        self._pb.stepSimulation()

        # debugging
        # self.arm.draw_EE()
        # self.arm.draw_TCP() # only works with visuals enabled in urdf file
        # self.arm.draw_workframe()
        # self.arm.draw_TCP_box()
        # self.arm.print_joint_pos_vel()
        # self.arm.print_TCP_pos_vel()
        # self.arm.test_workframe_transforms()
        # self.arm.test_workvec_transforms()
        # self.arm.test_workvel_transforms()
        # self.t_s.draw_camera_frame()
        # self.t_s.draw_t_s_frame()

    def apply_action(
        self,
        motor_commands,
        control_mode="TCP_velocity_control",
        velocity_action_repeat=1,
        max_steps=100,
    ):

        if control_mode == "TCP_position_control":
            self.arm.tcp_position_control(motor_commands)

        elif control_mode == "TCP_velocity_control":
            self.arm.tcp_velocity_control(motor_commands)

        elif control_mode == "joint_velocity_control":
            self.arm.joint_velocity_control(motor_commands)

        else:
            sys.exit("Incorrect control mode specified: {}".format(control_mode))

        if control_mode == "TCP_position_control":
            # repeatedly step the sim until a target pose is met or max iters
            self.blocking_move(max_steps=max_steps, constant_vel=None)

        elif control_mode in ["TCP_velocity_control", "joint_velocity_control"]:
            # apply the action for n steps to match control rate
            for i in range(velocity_action_repeat):
                self.step_sim()
        else:
            # just do one step of the sime
            self.step_sim()

    def blocking_move(
        self,
        max_steps=1000,
        constant_vel=None,
        pos_tol=2e-4,
        orn_tol=1e-3,
        jvel_tol=0.1,
    ):
        """
        step the simulation until a target position has been reached or the max
        number of steps has been reached
        """
        # get target position
        targ_pos = self.arm.target_pos_worldframe
        targ_orn = self.arm.target_orn_worldframe
        targ_j_pos = self.arm.target_joints

        pos_error = 0.0
        orn_error = 0.0
        for i in range(max_steps):

            # get the current position and veloicities (worldframe)
            (
                cur_TCP_pos,
                cur_TCP_rpy,
                cur_TCP_orn,
                _,
                _,
            ) = self.arm.get_current_TCP_pos_vel_worldframe()

            # get the current joint positions and velocities
            cur_j_pos, cur_j_vel = self.arm.get_current_joint_pos_vel()

            # Move with constant velocity (from google-ravens)
            # break large position move to series of small position moves.
            if constant_vel is not None:
                diff_j = np.array(targ_j_pos) - np.array(cur_j_pos)
                norm = np.linalg.norm(diff_j)
                v = diff_j / norm if norm > 0 else np.zeros_like(cur_j_pos)
                step_j = cur_j_pos + v * constant_vel

                # reduce vel if joints are close enough,
                # this helps to acheive final pose
                if all(np.abs(diff_j) < constant_vel):
                    constant_vel /= 2

                # set joint control
                self._pb.setJointMotorControlArray(
                    self.robot_id,
                    self.arm.control_joint_ids,
                    self._pb.POSITION_CONTROL,
                    targetPositions=step_j,
                    targetVelocities=[0.0] * self.arm.num_control_dofs,
                    positionGains=[self.arm.pos_gain] * self.arm.num_control_dofs,
                    velocityGains=[self.arm.vel_gain] * self.arm.num_control_dofs
                )

            # step the simulation
            self.step_sim()

            # calc totoal velocity
            total_j_vel = np.sum(np.abs(cur_j_vel))

            # calculate the difference between target and actual pose
            pos_error = np.sum(np.abs(targ_pos - cur_TCP_pos))
            orn_error = np.arccos(
                np.clip((2 * (np.inner(targ_orn, cur_TCP_orn) ** 2)) - 1, -1, 1)
            )

            # break if the pose error is small enough
            # and the velocity is low enough
            if (pos_error < pos_tol) and (orn_error < orn_tol) and (total_j_vel < jvel_tol):
                break

    def get_tactile_observation(self):
        return self.t_s.get_observation()

class ExampleArmEnv(BaseTactileEnv):
    def __init__(
        self,
        max_steps=1000,
        image_size=[64, 64],
        env_modes=dict(),
        show_gui=False,
        show_tactile=False,
    ):

        print("---------------------------------------------------------")
        # used to setup control of robot
        self._sim_time_step = 1.0 / 240.0
        self._control_rate = 1.0 / 10.0
        self._velocity_action_repeat = int(np.floor(self._control_rate / self._sim_time_step))
        self._max_blocking_pos_move_steps = 10

        super(ExampleArmEnv, self).__init__(max_steps, image_size, show_gui, show_tactile)

        # set modes from algorithm side
        self.movement_mode = env_modes["movement_mode"]
        self.control_mode = env_modes["control_mode"]
        self.observation_mode = env_modes["observation_mode"]
        self.reward_mode = env_modes["reward_mode"]

        # set which robot arm and sensor to use
        self.arm_type = env_modes["arm_type"]
        self.t_s_name = env_modes["tactile_sensor_name"]
        self.t_s_type = "standard"
        self.t_s_core = "no_core"

        # setup variables
        self.setup_action_space()

        # load environment objects
        self.load_environment()

        # limits
        TCP_lims = np.zeros(shape=(6, 2))
        TCP_lims[0, 0], TCP_lims[0, 1] = -0.1, +0.1  # x lims
        TCP_lims[1, 0], TCP_lims[1, 1] = -0.1, +0.1  # y lims
        TCP_lims[2, 0], TCP_lims[2, 1] = -0.1, +0.1  # z lims
        TCP_lims[3, 0], TCP_lims[3, 1] = -np.pi / 8, np.pi / 8  # roll lims
        TCP_lims[4, 0], TCP_lims[4, 1] = -np.pi / 8, np.pi / 8  # pitch lims
        TCP_lims[5, 0], TCP_lims[5, 1] = -np.pi / 8, np.pi / 8  # yaw lims

        # set workframe
        self.workframe_pos = np.array([0.65, 0.0, 0.05])
        self.workframe_rpy = np.array([-np.pi, 0.0, np.pi / 2])

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[self.arm_type][self.t_s_name][self.t_s_type]

        # load the ur5 with a tactip attached
        self.robot = Robot(
            self._pb,
            rest_poses=rest_poses,
            workframe_pos=self.workframe_pos,
            workframe_rpy=self.workframe_rpy,
            TCP_lims=TCP_lims,
            image_size=image_size,
            turn_off_border=False,
            arm_type=self.arm_type,
            t_s_name=self.t_s_name,
            t_s_type=self.t_s_type,
            t_s_core=self.t_s_core,
            t_s_dynamics={'stiffness': 50, 'damping': 100, 'friction': 10.0},
            show_gui=self._show_gui,
            show_tactile=self._show_tactile,
        )

        # this is needed to set some variables used for initial observation/obs_dim()
        self.reset()

        # set the observation space dependent on
        self.setup_observation_space()

    def setup_action_space(self):

        # these are used for bounds on the action space in SAC and clipping
        # range for PPO
        self.min_action, self.max_action = -0.01, 0.01

        # define action ranges per act dim to rescale output of policy
        if self.control_mode == "TCP_position_control":

            max_pos_change = 0.001  # m per step
            max_ang_change = 1 * (np.pi / 180)  # rad per step

            self.x_act_min, self.x_act_max = -max_pos_change, max_pos_change
            self.y_act_min, self.y_act_max = -max_pos_change, max_pos_change
            self.z_act_min, self.z_act_max = -max_pos_change, max_pos_change
            self.roll_act_min, self.roll_act_max = -max_ang_change, max_ang_change
            self.pitch_act_min, self.pitch_act_max = -max_ang_change, max_ang_change
            self.yaw_act_min, self.yaw_act_max = -max_ang_change, max_ang_change

        elif self.control_mode == "TCP_velocity_control":

            max_pos_vel = 0.01  # m/s
            max_ang_vel = 5.0 * (np.pi / 180)  # rad/s

            self.x_act_min, self.x_act_max = -max_pos_vel, max_pos_vel
            self.y_act_min, self.y_act_max = -max_pos_vel, max_pos_vel
            self.z_act_min, self.z_act_max = -max_pos_vel, max_pos_vel
            self.roll_act_min, self.roll_act_max = -max_ang_vel, max_ang_vel
            self.pitch_act_min, self.pitch_act_max = -max_ang_vel, max_ang_vel
            self.yaw_act_min, self.yaw_act_max = -max_ang_vel, max_ang_vel

        # setup action space
        self.act_dim = self.get_act_dim()
        self.action_space = gym.spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

    def setup_rgb_obs_camera_params(self):
        """
        Setup camera parameters for debug visualiser and RGB observation
        """
        self.rgb_cam_pos = [0.35, 0.0, -0.25]
        self.rgb_cam_dist = 1.0
        self.rgb_cam_yaw = 90
        self.rgb_cam_pitch = -35
        self.rgb_image_size = self._image_size
        self.rgb_fov = 75
        self.rgb_near_val = 0.1
        self.rgb_far_val = 100

    def reset(self):

        # full reset pybullet sim to clear cache, this avoids silent bug where memory fills and visual
        # rendering fails, this is more prevalent when loading/removing larger files
        if self.reset_counter == self.reset_limit:
            self.full_reset()

        self.reset_counter += 1
        self._env_step_counter = 0

        # reset TCP pos and rpy in work frame
        self.robot.reset(reset_TCP_pos=[0, 0, 0], reset_TCP_rpy=[0, 0, 0])

        # get the starting observation
        self._observation = self.get_observation()

        return self._observation

    def full_reset(self):
        self._pb.resetSimulation()
        self.load_environment()
        self.robot.full_reset()
        self.reset_counter = 0

    def encode_actions(self, actions):
        """
        Return actions as np.array in correct places for sending to robot arm
        i.e. NN could be predicting [y, Rz] actions. Make sure they are in
        correct place [1, 5].
        """

        encoded_actions = np.zeros(6)

        if self.movement_mode == "xyzRxRyRz":
            encoded_actions[0] = actions[0]
            encoded_actions[1] = actions[1]
            encoded_actions[2] = actions[2]
            encoded_actions[3] = actions[3]
            encoded_actions[4] = actions[4]
            encoded_actions[5] = actions[5]

        else:
            sys.exit("Incorrect movement mode specified: {}".format(self.movement_mode))

        return encoded_actions

    def get_step_data(self):

        # get rl info
        done = self.termination()

        if self.reward_mode == "sparse":
            reward = self.sparse_reward()

        elif self.reward_mode == "dense":
            reward = self.dense_reward()

        return reward, done

    def termination(self):
        # terminate when max ep len reached
        if self._env_step_counter >= self._max_steps:
            return True
        return False

    def sparse_reward(self):
        return 0.0

    def dense_reward(self):
        return 0.0

    def get_act_dim(self):
        if self.movement_mode == "xyzRxRyRz":
            return 6
        else:
            sys.exit("Incorrect movement mode specified: {}".format(self.movement_mode))

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
        "movement_mode": "xyzRxRyRz",

        # specify arm
        "arm_type": "ur5",
        # "arm_type": "franka_panda",
        # "arm_type": "kuka_iiwa",
        # "arm_type": "mg400",

        # specify tactile sensor
        "tactile_sensor_name": "tactip",
        # "tactile_sensor_name": "digit",
        # "tactile_sensor_name": "digitac",

        # the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        # which observation type to return
        "observation_mode": "oracle",
        # 'observation_mode':'tactile',
        # 'observation_mode':'visual',
        # 'observation_mode':'visuotactile',
        # 'observation_mode':'tactile_and_feature',
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        # the reward type
        # 'reward_mode':'sparse'
        "reward_mode": "dense",
    }

    env = ExampleArmEnv(
        max_steps=max_steps,
        env_modes=env_modes,
        show_gui=show_gui,
        show_tactile=show_tactile,
        image_size=image_size,
    )

    # set seed for deterministic results
    env.seed(seed)
    # env.action_space.np_random.bit_generator.seed(seed)

    # create controllable parameters on GUI
    action_ids = []
    min_action, max_action = env.min_action, env.max_action
    if show_gui:
        action_ids.append(
            env._pb.addUserDebugParameter("dx", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dy", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dz", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dRX", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dRY", min_action, max_action, 0)
        )
        action_ids.append(
            env._pb.addUserDebugParameter("dRZ", min_action, max_action, 0)
        )

    # run the control loop
    demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info)


if __name__ == "__main__":
    main()
