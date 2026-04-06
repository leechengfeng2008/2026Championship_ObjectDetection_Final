from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    nt_poll_dt: float
    sort_targets_by_area_desc: bool
    stale_timeout_s: float
    loop_sleep_s: float
    print_every_n_loops: int
    debug: bool
    target_height_m: float
    distance_eps: float
    dedupe_same_ball_error_m: float
    dedupe_max_angle_deg: float

    pile_ball_priority_0to10: float


APP = AppConfig(
    nt_poll_dt=0.02,
    sort_targets_by_area_desc=True,
    stale_timeout_s=0.50,
    loop_sleep_s=0.02,
    print_every_n_loops=10,
    debug=True,
    target_height_m=0.075,   # 球心高度，依你的模型調整
    distance_eps=1e-6, #  distance_from_pitch()  tan 保護閾值

    dedupe_same_ball_error_m=0.10,
    dedupe_max_angle_deg=35.0,

    pile_ball_priority_0to10 = 8.0,
)