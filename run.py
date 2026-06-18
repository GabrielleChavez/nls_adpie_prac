import handwriting_tasks as hw
import eyetracking_tasks as et
import process_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    # Load and process data

    completed_data = process_data.get_completed_paths(verbose=True)


    # competed_data contains all eyetracking and handwriting processed data
    # below we explore the keys inside the dictionary
    for key in completed_data.keys():
        print(key)
        print(completed_data[key].keys())   
        