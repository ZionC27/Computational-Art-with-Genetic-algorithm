# Computational-Art-with-Genetic-algorithm


This project employs a genetic algorithm to iteratively generate an image resembling an original image. The methodology involves several key steps and processes:

## Methodology Overview

### A. Population Creation
- Chromosomes encapsulate genes representing circles to be drawn on the canvas.
- Initial creation of chromosomes involves random values for X, and Y coordinates, radius of the circle, and HSL values. ()
- Flexibility allows adjusting the number of circles and population size for experimentation.

### B. Drawing the Image
- Utilizes extracted genes to draw circles based on their parameters.
- Maps 1 or 0 values to appropriate HSL parameters, radius, and coordinates.
- Dynamic adjustment of radius for optimal outcomes.

### C. Fitness Function
- Compares original and generated images pixel-wise.
- Utilizes Root Mean Square Error (RMSE) for quantifying image similarity.
- Provides a fitness score indicating proximity between the images.

### D. Generation Loop
- Sorts chromosomes based on fitness, retains top performers and discards the lower half.
- Periodically stores populations to a JSON file for progress backup.
- Executes crossover and mutation functions for subsequent generations.

### E. Crossover
- Randomly selects parent chromosomes and performs uniform crossover for gene swapping.

### F. Mutation
- Operates on genes within chromosomes, introducing alterations based on assigned probabilities.
- Allows changes in coordinates, radius, and HSL values with adjustable magnitudes.
- Includes addition or deletion of genes, favoring a higher chance of adding new genes.

## Conclusion
This algorithmic approach aims to progressively generate an image resembling the original by refining circles' attributes using genetic operations like crossover and mutation. The fitness function enables efficient evaluation, guiding the iterative process towards image resemblance.


## Requirements to run: 
Python: version 3.9 and above should work

Pillow (PIL): version 10.1.0 and above can be installed with
```
pip install Pillow
```
Numpy: version1.22 and above can be installed with
```
pip install numpy
```

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
## Gallery
![fuji](https://github.com/ZionC27/Computational-Art-with-Genetic-algorithm/assets/56661548/8df77624-816e-48e1-85e5-7beb0b83880b)
![21800](https://github.com/ZionC27/Computational-Art-with-Genetic-algorithm/assets/56661548/a167c99a-2d5a-41b7-9800-391bd6164578)
21800 Generations, 1000 Initial circles Population 200
