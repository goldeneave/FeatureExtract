# !pip install colormath


from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
from PIL import Image
import numpy as np

def color_contrast(image):
    # Convert the image to a numpy array
    image = np.array(image)
    
    # Get the shape of the image
    rows, cols, channels = image.shape
    
    # Initialize a list to store the color differences
    color_diffs = []
    
    # Loop over each pair of pixels in the image
    for row in range(rows):
        for col in range(cols):
            for i in range(row, rows):
                for j in range(col, cols):
                    if (i, j) != (row, col):
                        # Get the RGB values of the pixels
                        rgb1 = image[row, col]
                        rgb2 = image[i, j]
                        
                        # Convert RGB values to CIELAB color space
                        color1 = sRGBColor(*rgb1)
                        color2 = sRGBColor(*rgb2)
                        lab1 = convert_color(color1, LabColor)
                        lab2 = convert_color(color2, LabColor)
                        
                        # Calculate the color difference between the colors in CIELAB space
                        color_diff = delta_e_cie2000(lab1, lab2)
                        
                        # Add the color difference to the list
                        color_diffs.append(color_diff)
    
    # Take the average of the color differences
    color_contrast = np.mean(color_diffs)
    
    return color_contrast

# Load an image using the Image module
image = Image.open(img_path)

# Calculate the color contrast for the image
color_contrast = color_contrast(image)

print("Color Contrast:", color_contrast)
