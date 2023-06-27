from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from agent.dqnagent import DQNAgent
from epsilon_profile import EpsilonProfile
from networks import MLP # Il faudra choisir notre réseau de neuronne, pour le moment j'ai mis MLP
#from networks import CNN

# test once by taking greedy actions based on Q values
def test_game(env: SpaceInvaders, agent: DQNAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = env.reset() if (same) else env.reset()
        if display:
            env.render()

        for step in range(max_steps):
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = env.step(action)

            if display:
                time.sleep(speed)
                env.render()

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                break
            state = next_state
    return n_steps, sum_rewards

def main():
    """ INSTANCIE LE JEU """ 
    game = SpaceInvaders(display=True)
    gamma = 1. #Je ne sais pas à quoi il sert exactement dans notre cas
    
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    
    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
    n_episodes = 2000
    max_steps = 50
    alpha = 0.001
    eps_profile = EpsilonProfile(1.0, 0.1)
    final_exploration_episode = 500

    # Hyperparamètres de DQN
    batch_size = 32
    replay_memory_size = 1000
    target_update_frequency = 100
    tau = 1.0

    """ INSTANCIE LE RESEAU DE NEURONES """
    # Hyperparamètres
    state = game.get_state() 
    model = MLP(state[0], state[1], state[2], state[3], game.na) #distance_joueur, distance_balle, sens_invader, bullet_state, nombre d'actions
    #model = CNN(env.ny, env.nx, env.nf, env.na)
    agent = DQNAgent(model, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
    agent.learn(game, n_episodes, max_steps)
    #controller = DQNAgent(game.na)
    test_game(game, agent, max_steps=15, nepisodes=10, speed=0.1, display=True)

    state = game.reset()
    while True:
        action = RandomAgent.select_action(state) #temporaires, je n'ai pas trouvé comment passer du réseau de neuronne à une action choisie 
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()
