import numpy as np
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt



def generate_products(num_products: int) -> tuple:

    np.random.seed(42)

    characteristics = np.zeros((num_products, 4))
    prices = np.zeros(num_products)

    # Группы продуктов
    food_groups = {
        'овощи': {'protein_range': (1, 3), 'fat_range': (0, 1), 'carb_range': (3, 10), 'calories_range': (20, 50),
                  'price_range': (80, 300)},
        'фрукты': {'protein_range': (0, 2), 'fat_range': (0, 1), 'carb_range': (10, 20), 'calories_range': (40, 80),
                   'price_range': (120, 500)},
        'мясо': {'protein_range': (15, 25), 'fat_range': (5, 15), 'carb_range': (0, 2), 'calories_range': (150, 250),
                 'price_range': (400, 1200)},
        'зерновые': {'protein_range': (8, 12), 'fat_range': (1, 3), 'carb_range': (60, 75),
                     'calories_range': (300, 400), 'price_range': (60, 250)},
        'молочные': {'protein_range': (3, 8), 'fat_range': (2, 5), 'carb_range': (4, 6), 'calories_range': (50, 100),
                     'price_range': (90, 350)}
    }

    groups = list(food_groups.keys())

    for i in range(num_products):
        group = np.random.choice(groups)
        params = food_groups[group]

        characteristics[i, 0] = np.random.uniform(*params['protein_range'])
        characteristics[i, 1] = np.random.uniform(*params['fat_range'])
        characteristics[i, 2] = np.random.uniform(*params['carb_range'])
        characteristics[i, 3] = np.random.uniform(*params['calories_range'])
        prices[i] = np.random.uniform(*params['price_range'])

    return characteristics, prices



N = 100  # количество продуктов
k = 7  # количество продуктов в рационе
MAX_BUDGET = 2000

# Нормы
NORM = np.array([60, 50, 200, 1800])  # белки, жиры, углеводы, калории


products, prices = generate_products(N)

# Вероятности генетических операторов
MUTATION_PROBABILITY = 0.3
CROSSOVER_PROBABILITY = 0.8


class Individual:
    def __init__(self, genome: np.ndarray):
        self.genome = genome
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self) -> float:
        """приспособленность особи"""
        if np.sum(self.genome) != k:
            return float('inf')

        # для выбранных продуктов остаются хар-ки, для невыбранных 0, суммарное значение выбранных
        total_characteristics = np.sum(products * self.genome[:, np.newaxis], axis=0)
        total_price = np.sum(prices * self.genome)

        # Отклонение от нормы
        deviation = np.sum(np.abs(total_characteristics - NORM) / NORM)

        # Штраф за превышение бюджета
        budget_penalty = max(0, total_price - MAX_BUDGET) * 0.1

        return deviation + budget_penalty


def create_random_individual() -> Individual:
    """Создает случайную особь с ровно k продуктами"""
    genome = np.zeros(N, dtype=int)
    indices = np.random.choice(N, size=k, replace=False)
    genome[indices] = 1
    return Individual(genome)


def select_parents(population: list) -> tuple:
    """Турнир"""
    tournament_size = 3
    tournament1 = np.random.choice(population, size=tournament_size, replace=False)
    tournament2 = np.random.choice(population, size=tournament_size, replace=False)
    parent1 = min(tournament1, key=lambda x: x.fitness)
    parent2 = min(tournament2, key=lambda x: x.fitness)
    return parent1, parent2



def single_point_crossover(parent1: Individual, parent2: Individual) -> tuple:
    """Одноточечный"""
    point = np.random.randint(1, N - 1)
    child1_genome = np.concatenate([parent1.genome[:point], parent2.genome[point:]])
    child2_genome = np.concatenate([parent2.genome[:point], parent1.genome[point:]])
    child1_genome = adjust_genome(child1_genome, k)
    child2_genome = adjust_genome(child2_genome, k)
    return Individual(child1_genome), Individual(child2_genome)


def two_point_crossover(parent1: Individual, parent2: Individual) -> tuple:
    """Двухточечный"""
    points = sorted(np.random.choice(N, size=2, replace=False))
    child1_genome = np.concatenate([
        parent1.genome[:points[0]],
        parent2.genome[points[0]:points[1]],
        parent1.genome[points[1]:]
    ])
    child2_genome = np.concatenate([
        parent2.genome[:points[0]],
        parent1.genome[points[0]:points[1]],
        parent2.genome[points[1]:]
    ])
    child1_genome = adjust_genome(child1_genome, k)
    child2_genome = adjust_genome(child2_genome, k)
    return Individual(child1_genome), Individual(child2_genome)


def uniform_crossover(parent1: Individual, parent2: Individual) -> tuple:
    """Равномерный"""
    mask = np.random.randint(0, 2, size=N)
    child1_genome = np.where(mask, parent1.genome, parent2.genome)
    child2_genome = np.where(mask, parent2.genome, parent1.genome)
    child1_genome = adjust_genome(child1_genome, k)
    child2_genome = adjust_genome(child2_genome, k)
    return Individual(child1_genome), Individual(child2_genome)


def adjust_genome(genome: np.ndarray, target_k: int) -> np.ndarray:
    """чтобы было ровно k"""
    current_k = np.sum(genome)
    if current_k == target_k:
        return genome

    adjusted_genome = genome.copy()
    ones_indices = np.where(adjusted_genome == 1)[0]
    zeros_indices = np.where(adjusted_genome == 0)[0]

    if current_k > target_k:
        to_remove = np.random.choice(ones_indices, size=current_k - target_k, replace=False)
        adjusted_genome[to_remove] = 0
    else:
        to_add = np.random.choice(zeros_indices, size=target_k - current_k, replace=False)
        adjusted_genome[to_add] = 1

    return adjusted_genome


#мутации
def random_replacement(individual: Individual) -> Individual:
    """Случайная"""
    genome = individual.genome.copy()
    ones_indices = np.where(genome == 1)[0]
    zeros_indices = np.where(genome == 0)[0]

    if len(ones_indices) > 0 and len(zeros_indices) > 0:
        to_remove = np.random.choice(ones_indices)
        to_add = np.random.choice(zeros_indices)
        genome[to_remove] = 0
        genome[to_add] = 1

    return Individual(genome)


def swap_mutation(individual: Individual) -> Individual:
    """Обмен двух случайных продуктов"""
    genome = individual.genome.copy()
    ones_indices = np.where(genome == 1)[0]
    zeros_indices = np.where(genome == 0)[0]

    if len(ones_indices) > 0 and len(zeros_indices) > 0:
        to_remove = np.random.choice(ones_indices)
        to_add = np.random.choice(zeros_indices)
        genome[to_remove] = 0
        genome[to_add] = 1

    return Individual(genome)


def inversion_mutation(individual: Individual) -> Individual:
    """Инверсия последовательности"""
    genome = individual.genome.copy()
    start, end = sorted(np.random.choice(N, size=2, replace=False))
    genome[start:end + 1] = genome[start:end + 1][::-1]
    genome = adjust_genome(genome, k)
    return Individual(genome)


def evolve_population(population: list, crossover_method: callable, mutation_method: callable) -> list:
    """Эволюционирует популяцию на одно поколение"""
    new_population = []

    while len(new_population) < len(population):
        parent1, parent2 = select_parents(population)

        # Кроссовер
        if np.random.rand() < CROSSOVER_PROBABILITY:
            child1, child2 = crossover_method(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        # Мутация
        if np.random.rand() < MUTATION_PROBABILITY:
            child1 = mutation_method(child1)
        if np.random.rand() < MUTATION_PROBABILITY:
            child2 = mutation_method(child2)

        new_population.extend([child1, child2])

    return new_population[:len(population)]


def run_experiment(crossover_method: callable, mutation_method: callable, label: str,
                   population_size: int = 50, generations: int = 100) -> list:
    """Запускает и возвращает историю лучшей приспособленности"""
    np.random.seed(42)
    population = [create_random_individual() for _ in range(population_size)]
    best_fitness_history = []

    for generation in range(generations):
        population = evolve_population(population, crossover_method, mutation_method)
        best_fitness = min(ind.fitness for ind in population)
        best_fitness_history.append(best_fitness)

    return best_fitness_history


# Основная программа
if __name__ == "__main__":
    experiments = [
        (single_point_crossover, random_replacement, "Single Point + Random Replacement"),
        (single_point_crossover, swap_mutation, "Single Point + Swap Mutation"),
        (single_point_crossover, inversion_mutation, "Single Point + Inversion Mutation"),
        (two_point_crossover, random_replacement, "Two Point + Random Replacement"),
        (two_point_crossover, swap_mutation, "Two Point + Swap Mutation"),
        (two_point_crossover, inversion_mutation, "Two Point + Inversion Mutation"),
        (uniform_crossover, random_replacement, "Uniform + Random Replacement"),
        (uniform_crossover, swap_mutation, "Uniform + Swap Mutation"),
        (uniform_crossover, inversion_mutation, "Uniform + Inversion Mutation"),
    ]

    # Запуск экспериментов и сбор результатов
    experiment_results = []

    for crossover, mutation, label in experiments:
        history = run_experiment(crossover, mutation, label)
        experiment_results.append((label, history))

    # Создание графика
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_results)))

    for i, (label, history) in enumerate(experiment_results):
        plt.plot(history, label=label, color=colors[i], linewidth=2, alpha=0.8)

    plt.title('Сравнение методов генетического алгоритма для составления рациона', fontsize=14, fontweight='bold')
    plt.xlabel('Поколение')
    plt.ylabel('Лучшая приспособленность')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Сохраняем график в файл вместо показа
    plt.tight_layout()
    plt.savefig('genetic_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("График сохранен в файл 'genetic_algorithm_comparison.png'")