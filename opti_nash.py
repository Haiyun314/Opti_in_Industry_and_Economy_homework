import numpy as np
import scipy.optimize as opt
import multiprocessing
import time


class Game:
    """
    pay_off: player A's payoff
    scores: based on opponent strategy, and trials, the final scores received by player A
    player_descion: player's strategy probability distribution

    """
    pay_off = np.array(
        [[-2, 1, 2],
         [2, -1, 0],
         [1, 0, -2]]
    )
    scores = 0
    player_descion = [0, 0, 0, 1, 1, 1, 1, 1, 2]

    def __init__(self, num_of_trials: int, num_of_threads: int):
        """

        :param num_of_trials: the total number of gaming trials
        :param num_of_threads: number of parallelizing thread, as default
        equal to the number of cores, it will reduce performance when threads large than cores,
        because frequently communication between cpus
        """
        assert num_of_trials >= 10000, f'the input number of trials:{num_of_trials} must larger than 10000'
        assert num_of_threads >= 1, f'the input number of threads{num_of_threads} must larger than 0'
        self.number_of_trials = num_of_trials
        self.num_of_threads = num_of_threads

    def decision(self, trial):
        """
        based on the players' strategy, compute the scores achieved by player a
        :param trial: number of trials at each core
        :return: the accumulation scores at the cores
        """
        COUNT = trial
        local_score = 0
        while COUNT:
            COUNT -= 1
            playera_descion = np.random.choice(self.player_descion)
            playerb_descion = np.random.choice(self.player_descion)
            local_score += self.pay_off[playera_descion, playerb_descion]
        return local_score

    def run_with_thread(self):
        """
        parallelize the trials into multi thread, and compute the total scores
        """
        base = self.number_of_trials // self.num_of_threads
        trials = [base] * (self.num_of_threads - 1)
        trials.append(self.number_of_trials - base * (self.num_of_threads - 1))
        try:
            with multiprocessing.Pool(processes=self.num_of_threads) as pool:
                result = pool.map(self.decision, trials)
        except Exception as e:
            print(f"Error creating multiprocessing.Pool: {e}")
            self.decision(self.number_of_trials)
        self.scores = sum(result)

    def result(self):
        print(f'final scores {self.scores / self.num_of_threads}, expected pay '
              f'off {self.scores / self.number_of_trials} '
              f'total trials = {self.number_of_trials}')


def opti_strategy(pay_off_matrix, player_position: bool):
    """

    :param pay_off matrix: outcomes of a strategic interaction between two or more decision-makers
    :param player_position:
    :return:
    """
    # pay off matrix of player A
    pay_off = pay_off_matrix

    # opponent pay off matrix
    pay_off_op = -1 * pay_off.transpose()

    # z coefficient
    z_coe = np.array([-1, -1, -1]).reshape((3, 1))

    # Coefficients of the objective function to be minimized
    c = np.array([0, 0, 0, -1])

    # Coefficients of the inequality constraints (Ax >= b)
    if not player_position:  # if player position 0
        Ine_M = -1 * np.append(pay_off, z_coe, axis=1)
    else:  # if player position 1
        Ine_M = -1 * np.append(pay_off_op, z_coe, axis=1)

    Ine_b = np.array([0, 0, 0])

    # Coefficients of the equality constraints (Ax = b)
    E_M = np.array([[1, 1, 1, 0]])
    E_b = np.array([1])

    # Bounds for variables (1>= xi >= 0)
    x1 = (0, 1)
    x2 = (0, 1)
    x3 = (0, 1)
    z = (-10, 10)

    # Solve the linear programming problem
    result = opt.linprog(c, A_ub=Ine_M, b_ub=Ine_b, A_eq=E_M, b_eq=E_b, bounds=[x1, x2, x3, z], method='highs')

    # Print the results
    print("Status:", result.message)
    print("Optimal Values (x1, x2, x3):", result.x[:-1])
    print("expected scores can be achieved", result.fun[-1])


if __name__ == '__main__':
    pay_off = np.array([[-2, 1, 2],
                        [2, -1, 0],
                        [1, 0, -2]]
                       )
    player = 0  # player position 0 or 1
    opti_strategy(pay_off, bool(player))

    try:
        number_of_threads = multiprocessing.cpu_count()
    except Exception as e:
        print(f'Error detecting CPU count: {e}')
        number_of_threads = 1
    Start_Game = Game(1000000, number_of_threads)
    start = time.perf_counter()
    Start_Game.run_with_thread()
    print(f'performance of {number_of_threads} threads is {time.perf_counter() - start}')
    Start_Game.result()
