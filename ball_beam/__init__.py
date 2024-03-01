from gym.envs.registration import register

register(
    id='BallBeam-v0', 
    entry_point='ball_beam.envs:BallBeamEnv'
)