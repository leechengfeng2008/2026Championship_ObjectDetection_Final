from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    nt_server: str
    cameras: list[str]
    client_name: str
    table_name: str
    robot_pose_topic: str


NETWORK = NetworkConfig(
    nt_server="10.25.6.17", #robrio IP
    
    #during test mode : in single Camera.
    #cameras=["Camera1", "Camera2"],
    cameras=["Camera1"],
    #cameras=["Camera2"],
    client_name="orangepi-multicam",
    table_name="photonvision",
    robot_pose_topic="/AdvantageKit/RealOutputs/RobotState/robotPose",
)