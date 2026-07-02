"""
ROS 2 bridge — read robot state and send navigation goals to the 3 carters.

This runs in the ROS 2 environment (source /opt/ros/<distro>/setup.bash first), NOT in
the agent's pip venv. The agent calls into this via a simple interface; for the hackathon
the cleanest integration is to run this as a small service the agent talks to, or import
its FleetBridge if the agent shares the ROS 2 env.

See docs/05-isaac-sim-ros2.md for topic names and patterns.

Run standalone test:
    source /opt/ros/humble/setup.bash
    python3 carter_agent.py
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

NAMESPACES = ("carter1", "carter2", "carter3")


class CarterAgent(Node):
    """Reads one Carter's odometry and sends it NavigateToPose goals."""

    def __init__(self, namespace: str):
        super().__init__("carter_agent", namespace=namespace)
        self.ns = namespace
        self.latest_pose: tuple[float, float, float, float] | None = None

        # Namespaced node → relative "odom" resolves to /<ns>/odom.
        self.create_subscription(Odometry, "odom", self._odom_cb, 10)
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def _odom_cb(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.latest_pose = (p.x, p.y, q.z, q.w)

    def send_goal(self, x: float, y: float, yaw_z: float = 0.0, yaw_w: float = 1.0) -> None:
        self._nav_client.wait_for_server()
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = "map"          # goals are in the MAP frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.z = float(yaw_z)
        ps.pose.orientation.w = float(yaw_w)
        goal.pose = ps
        self.get_logger().info(f"[{self.ns}] goal ({x:.2f}, {y:.2f})")
        fut = self._nav_client.send_goal_async(goal, feedback_callback=self._fb)
        fut.add_done_callback(self._goal_response)

    def _fb(self, msg) -> None:
        self.get_logger().info(
            f"[{self.ns}] distance_remaining={msg.feedback.distance_remaining:.2f}")

    def _goal_response(self, future) -> None:
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn(f"[{self.ns}] goal rejected")
            return
        gh.get_result_async().add_done_callback(
            lambda f: self.get_logger().info(
                f"[{self.ns}] done status={f.result().status}"))


class FleetBridge:
    """
    Convenience wrapper the agent's tools can call:
      bridge.positions()            -> {"carter1": (x, y), ...}
      bridge.send("carter1", x, y)  -> dispatch a nav goal
    """

    def __init__(self):
        rclpy.init()
        self.agents = {ns: CarterAgent(ns) for ns in NAMESPACES}
        self.executor = MultiThreadedExecutor()
        for a in self.agents.values():
            self.executor.add_node(a)

    def spin_some(self, timeout_sec: float = 0.1) -> None:
        self.executor.spin_once(timeout_sec=timeout_sec)

    def positions(self) -> dict[str, tuple[float, float] | None]:
        self.spin_some()
        return {
            ns: (a.latest_pose[0], a.latest_pose[1]) if a.latest_pose else None
            for ns, a in self.agents.items()
        }

    def send(self, robot: str, x: float, y: float) -> None:
        self.agents[robot].send_goal(x, y)

    def shutdown(self) -> None:
        for a in self.agents.values():
            a.destroy_node()
        rclpy.shutdown()


def main() -> None:
    bridge = FleetBridge()
    # quick manual test: drive each robot somewhere
    bridge.send("carter1", 5.0, 3.0)
    bridge.send("carter2", -2.0, 4.0)
    bridge.send("carter3", 1.0, -6.0)
    try:
        bridge.executor.spin()
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
