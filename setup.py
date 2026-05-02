from setuptools import setup
from glob import glob
package_name = 'amr_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, package_name + '.utils'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marco Gadallah',
    maintainer_email='marco.gadallah@smail.inf.h-brs.de',
    description='AMR Perception ROS2 Package',
    license='MIT',
    tests_require=['pytest'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/amr_perception']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/maps', glob('maps/*')),
    ],    
    entry_points={
        'console_scripts': [
            'potential_field_planner = amr_perception.potential_field_planner:main',
            'astar_planner = amr_perception.astar_planner:main',
            'planner_coordinator = amr_perception.planner_coordinator:main',
            'particle_filter = amr_perception.particle_filter:main',
                  
        ],
    },
)
