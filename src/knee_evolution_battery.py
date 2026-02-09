import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import get_args_parser, prediction, data_loading


class Knee_Curve:

    def __init__(self, args):
        super(Knee_Curve, self).__init__()

        self.group = args.group
        self.dataset = args.dataset
        self.input = args.input
        self.to_predict = args.to_predict

        # Datu-basea aukeratu (NASA/MENDELEY/Prognosis)
        self.df_battery, self.df_summary = data_loading(self.dataset)

        prediction(self.dataset, self.df_summary, self.df_battery, self.input, self.to_predict, 100, self.group)


args = get_args_parser().parse_args()
Knee_Curve(args)
