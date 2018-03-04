from dps.env.advanced import yolo_rl

config = yolo_rl.config.copy()
config.curriculum = config.curriculum[2:]
config.load_path = '/data/dps_data/logs/yolo_rl/rl_error/weights/best_of_stage_1'
