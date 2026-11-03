from setuptools import find_packages, setup

package_name = 'AgileAutonomy_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='michael',
    maintainer_email='michaelbryanjahanto@gmail.com',
    description='Agile Autonomy (Loquercio et al. 2021) baseline for KAIROS',
    license='TODO: License declaration',
    entry_points={'console_scripts': []},
)
