"""
Utility functions for printing colored text to the console.
"""
import random
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# Define available foreground colors
foreground_colors = [
    Fore.BLACK, Fore.RED, Fore.GREEN, Fore.YELLOW, 
    Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE,
    Fore.LIGHTBLACK_EX, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX,
    Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX,
    Fore.LIGHTCYAN_EX, Fore.LIGHTWHITE_EX
]

# Define available background colors
background_colors = [
    Back.BLACK, Back.RED, Back.GREEN, Back.YELLOW, 
    Back.BLUE, Back.MAGENTA, Back.CYAN, Back.WHITE,
    Back.LIGHTBLACK_EX, Back.LIGHTRED_EX, Back.LIGHTGREEN_EX,
    Back.LIGHTYELLOW_EX, Back.LIGHTBLUE_EX, Back.LIGHTMAGENTA_EX,
    Back.LIGHTCYAN_EX, Back.LIGHTWHITE_EX
]

# Define styles
styles = [Style.NORMAL, Style.BRIGHT, Style.DIM]

def print_random_color(text):
    """
    Print text with random foreground color, background color, and style.
    
    Args:
        text: Text string to print
    """
    fg_color = random.choice(foreground_colors)
    bg_color = random.choice(background_colors)
    style = random.choice(styles)
    print(fg_color + bg_color + style + text)
