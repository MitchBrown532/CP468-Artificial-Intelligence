from collections import deque
import math
import pandas as pd

# For calculating Euclidean distance - will use to determine distance between 2 given cities
def calculate_distance(city1, city2):
    x1, y1 = city1['latitude'], city1['longitude']
    x2, y2 = city2['latitude'], city2['longitude']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) # Pythagorean theorem

# For creating a distance matrix
def create_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = [[0] * num_cities for _ in range(num_cities)] # Declare initial distances to 0, create square matrix
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j: # Keep matrix values on the diagonal equal to 0 since distance from city to itself is 0
                distance_matrix[i][j] = calculate_distance(cities.loc[i], cities.loc[j])
    return distance_matrix

# Breadth-first search (BFS)
def tsp_bfs(distance_matrix):
    num_cities = len(distance_matrix)
    visited = [False] * num_cities
    min_distance = float('inf')
    min_path = []

    queue = deque([(0, 0, [0])])  # Queue of lists, ind 0 of entry is next city index, ind 1 is the distance from the current city to the next city, and ind 2 tracks the current path
    while queue: # Iterates until queue is empty
        curr_city, distance, path = queue.popleft()

        if len(path) == num_cities: # If new path through all cities has been created
            distance += distance_matrix[curr_city][0]  # Add distance from last city back to the starting city
            if distance < min_distance: # Update if new shorter path has been found
                min_distance = distance
                min_path = path[:]
            continue

        visited[curr_city] = True
        for next_city in range(num_cities):
            if not visited[next_city]:
                queue.append((next_city, distance + distance_matrix[curr_city][next_city], path + [next_city]))

        visited[curr_city] = False

    return min_distance, [cities['name'][i] for i in min_path]

def tsp_dfs(distance_matrix):
    num_cities = len(distance_matrix)
    visited = [False] * num_cities
    min_distance = float('inf')
    min_path = []

    stack = deque([(0, 0, [0])])  # Stack of lists, ind 0 of entry is next city index, ind 1 is the distance from the current city to the next city, and ind 2 tracks the current path
    while stack: # Iterates until stack is empty
        curr_city, distance, path = stack.popleft()

        if len(path) == num_cities: # If new path through all cities has been created
            distance += distance_matrix[curr_city][0]  # Add distance from last city back to the starting city
            if distance < min_distance: # Update if new shorter path has been found
                min_distance = distance
                min_path = path[:]
            continue

        visited[curr_city] = True
        for next_city in range(num_cities):
            if not visited[next_city]:
                stack.appendleft((next_city, distance + distance_matrix[curr_city][next_city], path + [next_city])) # Using appendleft to create stack data structure effect with double-ended queue

        visited[curr_city] = False

    return min_distance, [cities['name'][i] for i in min_path]

# Read the CSV file using pandas to make a dataframe. - 50 is full set, 10 is miniset (for testing)
cities = pd.read_csv('city_data_50.csv')
# cities = pd.read_csv('city_data_10.csv')
# print(cities.head())

# Create the distance matrix
distance_matrix = create_distance_matrix(cities)
# print(distance_matrix)

# Solve TSP using BFS
print("Working on BFS..........")
bfs_min_distance, bfs_min_path = tsp_bfs(distance_matrix)
print("BFS - Min Distance:", bfs_min_distance)
print("BFS - Min Path:", bfs_min_path)

# Solve TSP using DFS
print("Working on DFS..........")
dfs_min_distance, dfs_min_path = tsp_dfs(distance_matrix)
print("DFS - Min Distance:", dfs_min_distance)
print("DFS - Min Path:", dfs_min_path)

print("------------------------------------")
print('Compare:')
print("DFS: ", dfs_min_distance)
print("BFS: ", bfs_min_distance)
print("------------------------------------")