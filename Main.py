import os
import numpy as np
from PIL import Image, ImageDraw
import random as rd
from numpy.random import choice, random
import json


POPULATION_SIZE = 200
GENERATIONS = 70000
NUMBER_CIRCLES = 600


UNIFORM_CROSSOVER_PROB = 0.5

### mutation amount 0.05
MUTATION_X_Y_AMOUNT = 0.02
MUTATION_RADIUS_AMOUNT = 0.02
MUTATION_COLOR_AMOUNT = 0.02

### mutation probability
MUTATION_PROBABILITY = 0.7
MUTATION_PROBABILITY_X_Y_AMOUNT = 0.3
MUTATION_PROBABILITY_RADIUS_AMOUNT = 0.3
MUTATION_PROBABILITY_COLOR_AMOUNT = 0.3

### Add and delete probability
CIRCLES_ADD_PROBABLITY = 1
CIRCLES_ADD_TYPE_PROBABLITY = 0.5
CIRCLES_DELETE_PROBABLITY = 0.3



FILEPATH = os.path.abspath(__file__)
FILEDIR = FILEPATH.replace(os.path.basename(FILEPATH), "")

FILENAME = "fiji.png"

og_image_file = Image.open(FILEDIR + FILENAME, "r")
og_image = np.array(og_image_file, dtype=np.uint64)

width, height = og_image_file.size
print(width, height)

blank = Image.new("RGB", (width, height), (255, 255, 255, 255))


# to get cordinates from decimal to x0,y0 x1, y1
def get_ellipse_coordinates(x, y, radius_norm, width, height):
    def map_coordinates_to_ellipse(x, y, radius_norm, width, height):
        canvas_x, canvas_y = int(x * width), int(y * height)
        canvas_radius = int(radius_norm * min(width, height))
        return [
            (canvas_x - canvas_radius, canvas_y - canvas_radius),
            (canvas_x + canvas_radius, canvas_y + canvas_radius),
        ]

    ellipse_coords = map_coordinates_to_ellipse(x, y, radius_norm, width, height)
    return ellipse_coords


# Mapping the value from [0, 1] to [0.008, 0.11] used for radius
def map_rad(value):
    mapped_value = 0.008 + (value * 0.035)
    return mapped_value


# Mapping the value from [0, 1] to [0, 360] for hue
def map_h(value):
    # Ensure hue is within the valid range [0, 360]
    mapped_value = int(value * 360) % 360
    return mapped_value


# Mapping the value from [0, 1] to [0, 100] for saturation and lightness
def map_sl(value):
    # Ensure saturation and lightness are within the valid range [0, 100]
    mapped_value = max(min(value * 100, 100), 0)
    rounded_value = round(mapped_value, 2)  # Round to two decimal places
    return rounded_value


def draw(chroma):
    image = Image.new("RGBA", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    for gene in chroma:
        x, y, r, h, s, l = gene
        coords = get_ellipse_coordinates(x, y, map_rad(r), width, height)
        circle_color = (
            "hsl(" + str(map_h(h)) + "," + str(map_sl(s)) + "%," + str(map_sl(l)) + "%)"
        )
        draw.ellipse(coords, fill=circle_color, outline=None, width=1)
    # print(x,y,r,h,s,l)

    # image.show()
    return np.array(image, dtype=np.uint64)


def draw_display(chroma):
    image = Image.new("RGBA", (width, height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for gene in chroma:
        x, y, r, h, s, l = gene
        coords = get_ellipse_coordinates(x, y, map_rad(r), width, height)
        circle_color = (
            "hsl(" + str(map_h(h)) + "," + str(map_sl(s)) + "%," + str(map_sl(l)) + "%)"
        )
        draw.ellipse(coords, fill=circle_color, outline=None, width=1)

    image.show()

def draw_save(chroma):
    image = Image.new("RGBA", (width, height), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for gene in chroma:
        x, y, r, h, s, l = gene
        coords = get_ellipse_coordinates(x, y, map_rad(r), width, height)
        circle_color = (
            "hsl(" + str(map_h(h)) + "," + str(map_sl(s)) + "%," + str(map_sl(l)) + "%)"
        )
        draw.ellipse(coords, fill=circle_color, outline=None, width=1)

    return image


def fitness(generated_image):
    # Ensure both images have the same shape
    if og_image.shape != generated_image.shape:
        raise ValueError("Images must have the same dimensions")

    pix1 = np.array(og_image, dtype=np.uint64)
    pix2 = np.array(generated_image, dtype=np.uint64)
    return round(np.sqrt(np.square(pix1 - pix2).sum(axis=-1)).sum(), 8)

def draw_fitness(chroma, chroma2):
    pix = draw(chroma)
    fit = fitness(pix)
    pix2 = draw(chroma2)
    fit2 = fitness(pix2)
    return fit, fit2


def crossover(parent1, parent2):
    # Randomly select a crossover point (assuming the chromosome has the same length)
    crossover_point = rd.randint(1, len(parent1) - 1)

    # Create children by combining genetic information from parents
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def one_point_crossover(parent1, parent2):
    # Select a random crossover point
    crossover_point = np.random.randint(1, len(parent1) - 1)  # Exclude ends

    # Perform crossover to create two children
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]), axis=0)
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]), axis=0)

    return child1, child2

def uniform_crossover(parent1, parent2):
    max_length = max(len(parent1), len(parent2))
    
    child1, child2 = [], []
    for i in range(max_length):
        if i < len(parent1) and i < len(parent2):
            if random() < UNIFORM_CROSSOVER_PROB :
                child1.append(parent2[i])
                child2.append(parent1[i])
            else:
                child1.append(parent1[i])
                child2.append(parent2[i])
        elif i < len(parent1):
            child1.append(parent1[i])
            child2.append(parent1[i])
        else:
            child1.append(parent2[i])
            child2.append(parent2[i])
    
    return child1, child2


def convert_to_numpy_array(list_of_arrays):
    # Find the maximum length among the arrays
    max_length = max(len(arr) for arr in list_of_arrays)
    
    # Fill shorter arrays with zeros to make them equal in length
    equal_length_arrays = [np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in list_of_arrays]
    
    # Convert the list of arrays into a single NumPy array
    numpy_array = np.array(equal_length_arrays)
    
    return numpy_array

def convert_to_numpy_array2(list1, list2):
    valof1 = convert_to_numpy_array(list1)
    valof2 = convert_to_numpy_array(list2)
    return valof1,valof2

def value_picker():
    values = [1.2, 1.15, 1.1, 1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    return rd.choice(values)

def mutate_X(gene, scaler):
    x_change = np.random.uniform(-MUTATION_X_Y_AMOUNT * scaler, MUTATION_X_Y_AMOUNT * scaler)
    gene[0] = min(max(gene[0] + x_change, 0), 1)
    return gene

def mutate_Y(gene, scaler):
    y_change = np.random.uniform(-MUTATION_X_Y_AMOUNT * scaler, MUTATION_X_Y_AMOUNT * scaler)
    gene[1] = min(max(gene[1] + y_change, 0), 1)
    return gene

def mutate_radius(gene, scaler):
    gene[2] = min(max(gene[2] + np.random.uniform(-MUTATION_RADIUS_AMOUNT * scaler, MUTATION_RADIUS_AMOUNT * scaler), 0), 1)
    return gene

def mutate_color_h(gene, scaler):
    gene[3] = min(max(gene[3] + np.random.uniform(-MUTATION_COLOR_AMOUNT * scaler, MUTATION_COLOR_AMOUNT * scaler), 0), 1)
    return gene

def mutate_color_s(gene, scaler):
    gene[4] = min(max(gene[4] + np.random.uniform(-MUTATION_COLOR_AMOUNT * scaler, MUTATION_COLOR_AMOUNT * scaler), 0), 1)
    return gene

def mutate_color_l(gene, scaler):
    gene[5] = min(max(gene[5] + np.random.uniform(-MUTATION_COLOR_AMOUNT * scaler, MUTATION_COLOR_AMOUNT * scaler), 0), 1)
    return gene

def new_gene_make(parent1, parent2):
    # Select a random crossover point
    crossover_point = np.random.randint(1, len(parent1) - 1)  # Exclude ends
    if np.random.randint(0, 2) == 1:
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]), axis=0)
    else:
        child1 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]), axis=0)

    return child1

def apply_random_mutations(gene, scaler, num_mutations):
    mutation_functions = [mutate_X, mutate_Y, mutate_radius, mutate_color_h, mutate_color_s, mutate_color_l]
    selected_mutations = np.random.choice(mutation_functions, num_mutations, replace=False)
    for mutation in selected_mutations:
        gene = mutation(gene, scaler)
    return gene   

def do_nothing(gene, scaler):
    # This function does nothing and returns the gene array as is
    return gene

def mutation(chromo):
    changed_chromo = []
    
    for i in chromo:
        ##### to change the scale of the mutation not always same
        if random() < 0.75:
            scaler = value_picker()
        else:
            scaler = 1
        if random() < MUTATION_PROBABILITY:
            changed_chromo.append(apply_random_mutations(i, scaler, np.random.randint(1, 7)))
        else:
            changed_chromo.append(i)

    if random() < CIRCLES_DELETE_PROBABLITY:
            length = len(changed_chromo)
            random_number = np.random.randint(0, length - 1)
            changed_chromo = np.delete(changed_chromo, random_number, axis=0)
    else:
        new_gene = np.around(np.random.random((6)), decimals=10)  
        changed_chromo = np.vstack((changed_chromo, new_gene))
        # if random() < CIRCLES_ADD_TYPE_PROBABLITY:
        #     new_gene = np.around(np.random.random((6)), decimals=10)  
        #     changed_chromo = np.vstack((changed_chromo, new_gene))
        # else:
        #     rd1 = chromo[rd.randint(1, len(chromo)-1)]
        #     rd2 = chromo[rd.randint(1, len(chromo)-1)]
        #     new_gene = new_gene_make(rd1, rd2)
        #     changed_chromo = np.vstack((changed_chromo, new_gene))
        
    return changed_chromo

# randomize position of the circles ex front to back
def randomizer(chromo):
    np.random.shuffle(chromo)
    return chromo

population = []
next_Gen_Population = []

for _ in range(POPULATION_SIZE):
    chromosome = np.around(np.random.random((NUMBER_CIRCLES, 6)), decimals=10)
    pix = draw(chromosome)
    fit = fitness(pix)

    # append fitness and chrmo
    population.append((fit, chromosome))

for generation in range(GENERATIONS):
    new_population = []
    for g in population:
        new_population.append(g)

    population.sort(key=lambda x: x[0])

    print(f"=== Generation {generation+1} ===")
    print(f"Best Fitness: {population[0][0]}")
    if generation % 60 == 0:

        file_path = os.path.join(FILEDIR, "population_data.json")  # File path using FILEDIR

        # Convert NumPy arrays to lists in the population data
        population_list = [(fit, chromo.tolist()) for fit, chromo in population]

        # Save population data to a JSON file
        with open(file_path, 'w') as file:
            json.dump(population_list, file)


        ig = draw_save(population[0][1])
        ig.save(FILEDIR+f'{generation}.png')
    print(len(population[0][1]))

    next_generation = population[: POPULATION_SIZE // 2]

    #print(len(next_generation))

    for i in range(POPULATION_SIZE // 4):
        parent1 = next_generation[i * 2][1]
        parent2 = next_generation[i * 2 + 1][1]
        child1, child2 = uniform_crossover(parent1, parent2)

        ch1, ch2 = convert_to_numpy_array2(child1, child2)
        mut1 = mutation(ch1)
        mut2 = mutation(ch2)
        c1, c2 = convert_to_numpy_array2(mut1, mut2)

        fit1, fit2 = draw_fitness(c1,c2)
        next_generation.append((fit1 , c1))
        next_generation.append((fit2 , c2))


    population = next_generation    
    #print("next_Gen_Population", population)

