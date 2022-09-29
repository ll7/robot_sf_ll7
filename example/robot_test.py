import os
from robot_sf.utils.utilities import clear_pycache
from robot_sf.robot_env import RobotEnv


def main():
    env = RobotEnv(difficulty=2)
    env.reset()

    print(env.robot.map.peds_sim_env.max_population_for_new_individual)
    print(env.robot.map.peds_sim_env.max_population_for_new_group)

    for _ in range(600):
        print(env.robot.map.peds_sim_env.peds.size())
        env.step([0,0])

    env.robot.map.peds_sim_env.peds.ped_states
    env.render(mode = 'plot').show()
    
    current_folder = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    clear_pycache(current_folder)
 

if __name__ == "__main__":
    main()
