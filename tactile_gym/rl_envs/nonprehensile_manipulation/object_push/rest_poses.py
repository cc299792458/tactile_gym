import numpy as np

rest_poses_dict = {
    "ur5": {
        "tactip":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.21330,  # base_joint         (revolute)
                    -2.12767,  # shoulder_joint     (revolute)
                    -1.83726,  # elbow_joint        (revolute)
                    -0.74633,  # wrist_1_joint      (revolute)
                    1.56940,  # wrist_2_joint       (revolute)
                    -1.78171,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00,  # tcp_joint              (fixed)
                ]
            )
        },
        "digit":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.21330,  # base_joint         (revolute)
                    -2.12767,  # shoulder_joint     (revolute)
                    -1.83726,  # elbow_joint        (revolute)
                    -0.74633,  # wrist_1_joint      (revolute)
                    1.56940,  # wrist_2_joint       (revolute)
                    -1.78171,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00   # tcp_joint              (fixed)
                ]
            )
        },
        "digitac":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.21330,  # base_joint         (revolute)
                    -2.12767,  # shoulder_joint     (revolute)
                    -1.83726,  # elbow_joint        (revolute)
                    -0.74633,  # wrist_1_joint      (revolute)
                    1.56940,  # wrist_2_joint       (revolute)
                    -1.78171,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00   # tcp_joint              (fixed)
                ]
            )
        },
    },

    "mg400": {
        "tactip":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.21330,  # base_joint         (revolute)
                    -2.12767,  # shoulder_joint     (revolute)
                    -1.83726,  # elbow_joint        (revolute)
                    -0.74633,  # wrist_1_joint      (revolute)
                    1.56940,  # wrist_2_joint       (revolute)
                    -1.78171,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00,  # tcp_joint              (fixed)
                ]
            )
        },
        "digit":{
            "right_angle": np.array(
                [
                    0.00,  # world_joint            (fixed)
                    -0.21330,  # base_joint         (revolute)
                    -2.12767,  # shoulder_joint     (revolute)
                    -1.83726,  # elbow_joint        (revolute)
                    -0.74633,  # wrist_1_joint      (revolute)
                    1.56940,  # wrist_2_joint       (revolute)
                    -1.78171,  # wrist_3_joint      (revolute)
                    0.00,  # ee_joint               (fixed)
                    0.00,  # tactip_ee_joint        (fixed)
                    0.00,  # tactip_body_to_adapter (fixed)
                    0.00,  # tactip_tip_to_body    (fixed)
                    0.00   # tcp_joint              (fixed)
                ]
            )
        },
        "digitac":{
            "right_angle": np.array(
                [
                    -0.4739595385398559,     # j1        (revolute)
                    1.2201380728130053,     # j2_1         (revolute)
                    0.2795016299958721,     # j3_1         (revolute)
                    -1.5989291747156857,     # j4_1          (revolute)
                    0.4923537940111881,   # j5          (revolute)
                    0,                      # ee_joint           (fixed)
                    0,                      # tactip_ee_joint           (fixed)
                    0,                      # tactip_tip_to_body    (fixed)
                    0,                      # tcp_joint (fixed)
                    1.2430560021223471,      # j2_2 = j2_1         (revolute)
                    -1.243056114315045 ,   # j3_2 = -j2_1         (revolute)
                    1.5294185070132973      # j4_2 = j2_1 + j3_1          (revolute)
                ]
            )
        },
    },

    "franka_panda": {
        "right_angle": np.array(
            [
                0.00,  # world_joint         (fixed)
                3.02044,  # panda_joint1     (revolute)
                1.33526,  # panda_joint2     (revolute)
                2.69697,  # panda_joint3     (revolute)
                1.00742,  # panda_joint4     (revolute)
                -2.58092,  # panda_joint5    (revolute)
                2.23643,  # panda_joint6     (revolute)
                6.46838,  # panda_joint7     (revolute)
                0.00,  # ee_joint            (fixed)
                0.00,  # tactip_ee_joint     (fixed)
                0.00,  # tactip_body_to_adapter (fixed)
                0.00,  # tactip_tip_to_body (fixed)
                0.00,  # tcp_joint           (fixed)
            ]
        )
    },
    "kuka_iiwa": {
        "right_angle": np.array(
            [
                0.00,  # world_joint            (fixed)
                -0.54316,  # lbr_iiwa_joint_1   (revolute)
                1.12767,  # lbr_iiwa_joint_2    (revolute)
                3.36399,  # lbr_iiwa_joint_3    (revolute)
                1.44895,  # lbr_iiwa_joint_4    (revolute)
                -3.50353,  # lbr_iiwa_joint_5   (revolute)
                0.60381,  # lbr_iiwa_joint_6    (revolute)
                -0.14553,  # lbr_iiwa_joint_7   (revolute)
                0.00,  # ee_joint               (fixed)
                0.00,  # tactip_ee_joint        (fixed)
                0.00,  # tactip_body_to_adapter (fixed)
                0.00,  # tactip_tip_to_body    (fixed)
                0.00,  # tcp_joint              (fixed)
            ]
        )
    },
}
