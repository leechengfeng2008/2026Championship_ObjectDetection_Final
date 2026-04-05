from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    nt_poll_dt: float
    sort_targets_by_area_desc: bool
    stale_timeout_s: float
    loop_sleep_s: float
    print_every_n_loops: int
    debug: bool


APP = AppConfig(
    nt_poll_dt=0.02,
    sort_targets_by_area_desc=True,
    stale_timeout_s=0.50,
    loop_sleep_s=0.02,
    print_every_n_loops=10,
    debug=True,
)