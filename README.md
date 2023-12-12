# Computational-Art-with-Genetic-algorithm
  This program aims to use a Genetic algorithm to recreate an image using circles.
![fuji](https://github.com/ZionC27/Computational-Art-with-Genetic-algorithm/assets/56661548/8df77624-816e-48e1-85e5-7beb0b83880b)
![21800](https://github.com/ZionC27/Computational-Art-with-Genetic-algorithm/assets/56661548/a167c99a-2d5a-41b7-9800-391bd6164578)
21800 Generations, 1000 Initial circles Population 200

## Requirements to run: 
Python: version 3.9 and above should work

Pillow (PIL): version 10.1.0 and above can be installed with "pip install Pillow"

Numpy: version1.22 and above can be installed with "pip install numpy"

## File structure
**Main.py**: used to start the initial run, choose a PNG image, put it in the same directory, find "FILENAME" in the code, change your filename.png, and run.

**Main_ct.py**: used to continue running if the program is stopped. To continue find "FILENAME" in the code and change it to your filename.png then find the last generated image number and enter it at "start_offset = " and run.

**output.py**: used to display the best 5 images find "FILENAME" in the code and change your filename.png then run.

## Notes 

The image file must be a PNG.

Constants and values 
```
  POPULATION_SIZE
  GENERATIONS
  NUMBER_CIRCLES
  UNIFORM_CROSSOVER_PROB
  MUTATION_X_Y_AMOUNT
  MUTATION_RADIUS_AMOUNT
  MUTATION_COLOR_AMOUNT
  MUTATION_PROBABILITY
  CIRCLES_ADD_PROBABLITY
  CIRCLES_DELETE_PROBABLITY

  def map_rad(value):
      # min max for radius
      mapped_value = min + (value * max)
      return mapped_value
```
They can all be modified to get different outcomes.
