import numpy as np

class ADPIE:
    PATH_1 = {
        'x': np.array([0, 0, 0.368, 0.736]),
        'y': np.array([ 0, -0.613333, -0.613333, -0.613333]),
        'point' : [7, 12, 13, 14]
         }
    PATH_2 = {
        'x': np.array([ 0,  0, -0.368, -0.368, -0.368, -0.736]), 
        'y': np.array([ 0,  0.613333,  0.613333,  0., -0.613333, -0.613333]),
        'point' : [7, 2, 1, 6, 11, 10]
        }
    PATH_3 = {
        'x': np.array([ 0, -0.368, -0.736, -0.736, -0.368,  0,  0.368]), 
        'y': np.array([0, 0, 0, 0.613333, 0.613333, 0.613333,0.613333]),
        'point': [7, 6, 5, 0, 1, 2, 3]
        }
    PATH_4 = {
        'x': np.array([0, 0, 0.368, 0.736, 0.736, 0.368, 0.368, 0]),
        'y': np.array([ 0,  0.613333,  0.613333,  0.613333,  0,  0,-0.613333, -0.613333]),
        'point' : [7, 2, 3, 4, 9, 8, 13, 12]
        }
    PATH_5 = {
        'x': np.array([0, 0.368, 0.736, 0.368, 0]),
        'y': np.array([ 0, -0.613333,  0,  0.613333,  0]),
        'point' : [7, 13, 9, 3, 7]
        }
    PATH_6 = {
        'x': np.array([ 0, -0.368, -0.736, -0.368,  0,  0.368,  0.736]), 
        'y': np.array([ 0,  0.613333,  0, -0.613333,  0, -0.613333,-0.613333]),
        'point' : [7, 1, 5, 11, 7, 13, 14]
        }
    PATH_7 = {
        'x': np.array([ 0, -0.368, -0.368,  0,  0.368,  0.736,  0.368,  0]), 
        'y': np.array([ 0,  0, -0.613333, -0.613333, -0.613333,  0, 0.613333,  0]),
        'point' : [7, 6, 11, 12, 13, 9, 3, 7]
        }

    ALL_PATHS = [PATH_1, PATH_2, PATH_3, PATH_4, PATH_5, PATH_6, PATH_7]

    TASKS = ["Video_recollection_task_1", "Pursuit_path_recollection", "Semantic_association", "Word_list_recall_a", "word_list_recall_b", "Copy_cube", "Draw_clock", "Drawing_speaking", "Trial_B_MoCA" ]

    PRESSURE_MAX = 31126.0 
    TIME_MAX = 24.42422109999461

    pursuit_path = {
        "Pursuit_path_recollection1" : PATH_1,
        "Pursuit_path_recollection2" : PATH_2,
        "Pursuit_path_recollection3" : PATH_3,
        "Pursuit_path_recollection4" : PATH_4,
        "Pursuit_path_recollection5" : PATH_5,
        "Pursuit_path_recollection6" : PATH_6,
        "Pursuit_path_recollection7" : PATH_7,
    }

    radii = {
            "Pursuit_path_recollection1" : 0.11089473684210525,
            "Pursuit_path_recollection2" : 0.175,
            "Pursuit_path_recollection3" : 0.07426315789473684,
            "Pursuit_path_recollection4" : 0.028473684210526314,
            "Pursuit_path_recollection5" : 0.06510526315789474,
            "Pursuit_path_recollection6" : 0.028473684210526314,
            "Pursuit_path_recollection7" : 0.010157894736842105,
        }
