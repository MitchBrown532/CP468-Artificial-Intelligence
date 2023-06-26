from sklearn.model_selection import GridSearchCV
from NQueens import GeneticAlgorithmSolver

def tune_genetic_algorithm():
    # Define the parameter grid
    param_grid = {
        'population_size': [50, 100, 200],
        'generations': [200, 400, 600],
        'mutation_rate': [0.1, 0.2, 0.3]
    }

    # Create an instance of the genetic algorithm solver
    ga_solver = GeneticAlgorithmSolver()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=ga_solver, param_grid=param_grid, scoring='neg_mean_squared_error')

    # Fit the data to find the best hyperparameters
    grid_search.fit()

    # Get the best hyperparameters and the corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_


    print("ding dong:", best_params, "bing bong:", best_score)
    return best_params, best_score

tune_genetic_algorithm()