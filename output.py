import os
import numpy as np
from PIL import Image, ImageDraw
import random as rd
from numpy.random import choice, random
import json


FILEPATH = os.path.abspath(__file__)
FILEDIR = FILEPATH.replace(os.path.basename(FILEPATH), "")

population = []


FILENAME = "fuji.png"

og_image_file = Image.open(FILEDIR + FILENAME, "r")
# save image as uint64 np array
og_image = np.array(og_image_file, dtype=np.uint64)

# Get width, height of the original image
width, height = og_image_file.size
print(width, height)

# draw a canvas with the circles and return for saving
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
    mapped_value = 0.008 + (value * 0.045)
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

file_path = os.path.join(FILEDIR, "population_data.json")
# Read and print the contents of the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)
    for fit, chromo_list in data:
        chromo_array = np.array(chromo_list)
        population.append((fit, chromo_array))

population = [(fit, np.array(chromo)) for fit, chromo in population]

# draw out the 5 best images
for i in range(5):
    ig = draw_save(population[0][1])
    ig.save(FILEDIR+f'best{i+1}.png')