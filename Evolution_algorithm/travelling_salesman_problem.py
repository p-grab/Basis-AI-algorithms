from random import sample, choice, random, seed
from matplotlib import pyplot as plt
from copy import deepcopy
import time
import statistics

FINAL_DIST_LIST = []
TIME_LIST = []


class City:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name
    
    def distance_to_city(self, city):
        dis_x = abs(self.x - city.x)
        dis_y = abs(self.y - city.y)
        return ((dis_x ** 2) + (dis_y ** 2)) ** 0.5


class Route:
    def __init__(self, path):
        self.path = path
        self.full_distance = self.calculate_path_distance()

    def calculate_path_distance(self):
        distance = 0
        for c in range (0, len(self.path)):
            city_from = self.path[c]
            city_to = None
            if c + 1 != len(self.path):
                city_to = self.path[c + 1]
            else:
                city_to = self.path[0]
            distance += city_from.distance_to_city(city_to)
        self.full_distance = distance
        return distance


def create_route(cities_List):
    while(True):
        path = sample(cities_List, len(cities_List))
        if cities_List[0] == path[0]:
            break
    return Route(path)


def get_base_populaton(population_size, cities_List):
    base_populaton = []
    for i in range(population_size):
        base_populaton.append(create_route(cities_List))
    return base_populaton


def selection_process(base_populaton, population_size: int):
    selected_population = []

    for i in range(population_size):
        pick1 = choice(base_populaton)
        pick2 = choice(base_populaton)
        #pick3 = choice(base_populaton)
        pk = [pick1, pick2]
        picks = [pick1.full_distance, pick2.full_distance]
        min_id = picks.index(min(picks))
        selected_population.append(pk[min_id])
    if (len(base_populaton) != len(selected_population)):
        raise RuntimeError('Failed in selection process')
    return selected_population


def mutation(selected_population, mutation_rate):
    mutated_population = []
    mutated_population = selected_population.copy()
    for route in mutated_population:
        rnd = random()
        if rnd <= mutation_rate:
            while(True):
                index1 = int(random() * len(route.path))
                index2 = int(random() * len(route.path))
                if index1 !=0 and index2 != index1:
                    break
            city1 = route.path[index1]
            city2 = route.path[index2]
            route.path[index2] = city1
            route.path[index1] = city2
        route.calculate_path_distance()
    return mutated_population
    

def run_genetic_alg(population_size, mutation_rate, generations, cityList):
    average_path_value = []
    min_path_value = []
    best_inv_list = []
    best_inv = None
    
    for i in range(0, generations):
        if i == 0:
            base_population = get_base_populaton(population_size, cityList)
        minimum = 9999999
        sum = 0
        n = 0
        for p in base_population:
            sum += p.full_distance
            if p.full_distance < minimum:
                minimum = p.calculate_path_distance()
                if best_inv is None:
                    copy_path = p.path.copy()
                    best_inv = Route(copy_path)
                    best_inv.calculate_path_distance()
                if best_inv.full_distance > p.full_distance:
                    copy_path = p.path.copy()
                    best_inv = Route(copy_path)
                    best_inv.calculate_path_distance()
            n += 1
        
        best_inv_list.append(best_inv)
        if minimum !=0  and minimum != 9999999:
            min_path_value.append(minimum)
        average_path_value.append(sum / len(base_population))

        selected_population = selection_process(base_population, population_size)
        muted_population = mutation(selected_population, mutation_rate)
        base_population = muted_population

    best_gen = min_path_value.index(min(min_path_value))
    names = [c.name for c in best_inv.path]
    print(names)
    print(round(best_inv.full_distance, 2))
    print(f'Best gen {best_gen}')
    FINAL_DIST_LIST.append(round(best_inv.full_distance, 2))
    print("\n")
    gen = [i for i in range(generations)]

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot([c.x for c in cityList], [c.y for c in cityList], 'o')
    best_inv.path.append(cityList[0])
    ax2.plot([c.x for c in best_inv.path], [c.y for c in best_inv.path],'.b-', label=[c.name for c in best_inv.path])
    distances = [d.full_distance for d in best_inv_list]
    ax3.plot(gen, distances)
    ax1.set_title('Cities')
    ax3.set_title('Generation')
    plt.show()

cityList = []

#TRACES:

# RANDOM
seed(10)
for i in range(0,30):
    cityList.append(City(x=int(random() * 100), y=int(random() * 100), name = i))
# for i in range(0,10):
#     cityList.append(City(x=int(random() * 30) + 100, y=int(random() * 30) + 100, name = i))
# for i in range(0,10):
#     cityList.append(City(x=int(random() * 30) + 200, y=int(random() * 30) + 200, name = i))
seed()

#REGULAR
# x = 100
# for i in range(4):
#     x += 10
#     y = 100
#     for j in range(5):
#         y += 10
#         cityList.append(City(x,y, name = str(i)+","+str(j)))

#RECTANGLE (SIMPLE)
# x = 0
# x=-100
# i = 0
# #for i in range(6):
# while(i < 9):
#     x += 100
#     y = 0
#     i+= 4
#     for j in range(2):
#         y += 30
#         cityList.append(City(x,y, name = str(i)+","+str(j)))

for i in range(1):
    start = time.process_time()
    run_genetic_alg(population_size=150, mutation_rate=0.01, generations=2000, cityList=cityList)
    stop = time.process_time()
    TIME_LIST.append(round(stop - start, 3))
sum_wyn = 0
sum_czas = 0
print('indeks\tdistans\tczas')
for i in range(len(FINAL_DIST_LIST)):
    print(f'{i}\t{FINAL_DIST_LIST[i]}\t{TIME_LIST[i]}')
    sum_wyn += FINAL_DIST_LIST[i]
    sum_czas += TIME_LIST[i]

print(f'min {round(min(FINAL_DIST_LIST),3)}\tmax {round(max(FINAL_DIST_LIST),3)}\tśr {round(sum_wyn / len(FINAL_DIST_LIST), 3)}\todch {round(statistics.stdev(FINAL_DIST_LIST), 3)}')
print(f'min {round(min(TIME_LIST),3)}\tmax {round(max(TIME_LIST),3)}\tśr {round(sum_czas / len(TIME_LIST),3)}\todch {round(statistics.stdev(TIME_LIST), 3)}')
