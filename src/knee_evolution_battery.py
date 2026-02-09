import os, sys
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import types
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import get_args_parser, prediction, data_loading


class Knee_Curve:

    def __init__(self, args):
        super(Knee_Curve, self).__init__()
        #self.action = args.action
        #self.pol_order = args.pol_order
        self.group = args.group
        self.dataset = args.dataset
        self.input = args.input
        self.to_predict = args.to_predict
        #self.charge_policies = args.include_charge_policies
        #self.cycle_evaluation = False
        self.paper = args.paper

        # Datu-basea aukeratu (NASA/MENDELEY/Prognosis)
        self.df_battery, self.df_summary = data_loading(self.dataset)
    
        
        # Zikloen analisia
        # if self.cycle_evaluation == True:
        #     self.every_Qmax_eval()
        #     self.every_knee_eval()
        #     self.plot_comparing_models()

        # if self.paper == 1: #Begoren 1. paperrerako.
            # Bizi-zikloa edo degradazio kurbaren knee-point puntua aurresateko
        prediction(self.dataset, self.df_summary, self.df_battery, self.input, self.to_predict, 100, self.group) #, self.charge_policies)


args = get_args_parser().parse_args()
Knee_Curve(args)
