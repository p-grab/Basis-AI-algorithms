from matplotlib import pyplot as plt
import numpy as np
import time
import random


def calculate_Himmelblau_value(x, y):
    return pow((x *x  + y - 11), 2) + pow((x + y * y - 7), 2)


def generate_random_point():
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    return [x, y]


def derivatives_value_Himmelblau(x,y):
    dx = 2 * (2 * x * (x * x + y - 11) + x + y * y - 7)
    dy = 2 * (x * x + 2 * y * (x + y * y - 7) + y - 11)
    return [dx, dy]


def steepest_gradient_descent(starting_point, B=0.015, itereation_number=1000, precision=1.e-12):
    [x, y] = starting_point
    change_in_step = 10
    t = 0
    while (t < itereation_number and change_in_step > precision):
        seen_points_SGD.append([x, y])
        prev_point = [x, y]
        d = derivatives_value_Himmelblau(x, y)
        [x, y] = [x - B * d[0], y - B *d[1]]
        change_in_step = pow((prev_point[0] - x) * (prev_point[0] - x)  + (prev_point[1] - y) * (prev_point[1] - y), 0.5)
        t += 1
    print(f"Number of iterations SGD:   {t}")
    print(f"Found minimum value:    {round(calculate_Himmelblau_value(x, y), 4)}")
    return x, y


def calc_reverse_hesjan(x, y):
    dxx = 12 * x * x + 4 * y - 42
    dyy = 4 * x + 12 * y * y - 26
    dxy = 4 * (x + y)
    hes = np.array([[dxx, dxy],
                    [dxy, dyy]])
    hes_reverse = np.linalg.inv(hes)
    return hes_reverse


def newton_method(starting_point, B=0.5, itereation_number=1000, precision=1.e-12):
    [x, y] = starting_point
    change_in_step = 10
    t = 0
    while (t < itereation_number and change_in_step > precision):
        seen_points_NEWTON.append([x, y])
        prev_point = [x, y]
        q = np.array(derivatives_value_Himmelblau(x, y))
        d = np.matmul(calc_reverse_hesjan(x, y), q)
        [x, y] = [x - B * d[0], y - B *d[1]]
        change_in_step = pow((prev_point[0] - x) * (prev_point[0] - x)  + (prev_point[1] - y) * (prev_point[1] - y), 0.5)
        t += 1
    print(f"Number of iterations Newton:   {t}")
    print(f"Found minimum value:    {round(calculate_Himmelblau_value(x, y), 4)}")
    return x, y


def make_plot(seen_points_SGD, seen_points_NEWTON):
    x_STG_list = []
    y_STG_list = []
    x_N_list = []
    y_N_list = []
    
    for point in seen_points_SGD:
        x_STG_list.append(point[0])
        y_STG_list.append(point[1])

    for point in seen_points_NEWTON:
        x_N_list.append(point[0])
        y_N_list.append(point[1])
    plt.clf()
    plt.plot(x_STG_list, y_STG_list, label="path to reach minumum STG")
    plt.plot(x_N_list, y_N_list, label="path to reach minumum NEWTON")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Path to find minimum')
    HIMMELBLAU_MIN_VAL = [3,-2.8055118, -3.779310, 3.584428], [2, 3.131312, -3.283186, -1.84126]
    plt.plot(HIMMELBLAU_MIN_VAL[0], HIMMELBLAU_MIN_VAL[1], "o", label="minumium")
    plt.plot(-0.2708, 0.923,"x", label="maximum")
    plt.legend(loc='upper right', ncol=1, fancybox=True, shadow=False, fontsize='x-small')
    plt.savefig(fname=f'STG_NEWTON_COMPARER_{i}')
    #plt.show()


if __name__ == "__main__":
    while(True):
        #Menu
        print("-------------------------------------------")
        print("\tSGD AND NEWTON METHODS COMPARER")
        print("1. Compare 'B' value (const starting point)")
        print("2. Compare starting point (const 'B' value)")
        print("3. Both starting point and 'B' value changing")
        print("0. Exit")
        option = input()
        if(option == "0"): break
        if(option not in ["1", "2", "3"]): continue
        try:
            working_times = int(input("How many times you want to compare methods? (between 1-50)\n"))
            max_iteration_times_SGD = int(input("How many maximum iteration steps in SGD method?\n"))
            max_iteration_times_N = int(input("How many maximum iteration steps in Newton method?\n"))
        except:
            print("WRONG INPUTS")
            break
        if (working_times <=0 or working_times > 50 or max_iteration_times_SGD <=0 or max_iteration_times_N <=0):
            print("WRONG INPUTS")
            break
        if option == "1":   one_point = True
        else:   one_point = False

        if (option == "1" or option == "3"):
            B_for_SGD_min = float(input("Input minumum 'B' value for SGD method\n"))
            B_for_SGD_max = float(input("Input maximum 'B' value for SGD method\n"))
            B_for_N_min = float(input("Input minumum 'B' value for Newton method\n"))
            B_for_N_max = float(input("Input maximum 'B' value for Newton method\n"))
            B_step_SGD = (B_for_SGD_max - B_for_SGD_min) / working_times
            B_step_N = (B_for_N_max - B_for_N_min) / working_times
            if (B_for_SGD_max - B_for_SGD_min <= 0 or B_for_N_max - B_for_N_min <= 0):
                print("WRONG INPUTS")
                break
        if (option == "2"):
            B_for_SGD = float(input("Input 'B' value for SGD method\n"))
            B_for_N = float(input("Input 'B' value for Newton method\n"))
            if (min([B_for_SGD, B_for_N]) <= 0):
                print("WRONG INPUTS")
                break
        #Run program and measure time
        i = 1
        while(i <= working_times):
            seen_points_SGD = []
            seen_points_NEWTON = []
            if one_point:
                if i == 1:
                    start_point = generate_random_point() 
            else:
                start_point = generate_random_point() 
            if (option == "1" or option == "3"):
                B_for_SGD = B_for_SGD_min + (i-1) * B_step_SGD
                B_for_N = B_for_N_min + (i-1) * B_step_N
            
            print(f"\nSTEP {i}")
            print("\t\tSGD METHOD")
            print(f"'B' value: {round(B_for_SGD, 3)}")
           
            start = time.process_time()
            found_point_SGD = steepest_gradient_descent(start_point, B=B_for_SGD, itereation_number=max_iteration_times_SGD)
            stop = time.process_time()
            steepest_gradient_descent_time = round(stop - start, 3)
            print(f"Found point:[{round(found_point_SGD[0], 4), round(found_point_SGD[1], 4), }]")
            print(f"Time: {steepest_gradient_descent_time} [s]")
            
            print("\t\tNEWTON METHOD")
            print(f"'B' value: {round(B_for_N, 3)}")

            start = time.process_time()
            found_point_N = newton_method(start_point, B=B_for_N, itereation_number=max_iteration_times_N)
            stop = time.process_time()
            newton_method_time = round(stop - start, 3)
            print(f"Found point:{round(found_point_N[0], 4), round(found_point_N[1], 4), }")
            print(f"Time: {newton_method_time} [s]")

            make_plot(seen_points_SGD, seen_points_NEWTON)
            i += 1
        break