import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    package_name = 'elevation_mapping_cupy'
    share_dir = get_package_share_directory(package_name)

    core_param_path = os.path.join(
        share_dir, 'config', 'core', 'core_param.yaml')
    robot_param_path = os.path.join(
        share_dir, 'config', 'setups', 'alphatruck', 'base.yaml')

    for path in (core_param_path, robot_param_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Config file not found: {path}')

    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz alongside the mapping node'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='',
        description='Path to an RViz config file (only used if launch_rviz:=true)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use /clock (simulation time) instead of wall clock'
    )

    elevation_mapping_node = Node(
        package=package_name,
        executable='elevation_mapping_node.py',
        name='elevation_mapping_node',
        output='screen',
        parameters=[
            core_param_path,
            robot_param_path,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
    )

    return LaunchDescription([
        launch_rviz_arg,
        rviz_config_arg,
        use_sim_time_arg,
        elevation_mapping_node,
        rviz_node,
    ])
