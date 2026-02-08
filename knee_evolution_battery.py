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
from utils import get_args_parser, prediction, data_loading


class Knee_Curve:

    def __init__(self, args):
        super(Knee_Curve, self).__init__()
        self.action = args.action
        self.pol_order = args.pol_order
        self.group = args.group
        self.dataset = args.dataset
        self.input = args.input
        self.to_predict = args.to_predict
        self.charge_policies = args.include_charge_policies
        self.cycle_evaluation = False
        self.paper = args.paper

        # Datu-basea aukeratu (NASA/MENDELEY/Prognosis)
        self.df_battery, self.df_summary = data_loading(self.dataset)
    
        
        # Zikloen analisia
        if self.cycle_evaluation == True:
            self.every_Qmax_eval()
            self.every_knee_eval()
            self.plot_comparing_models()

        if self.paper == 1: #Begoren 1. paperrerako.
            # Bizi-zikloa edo degradazio kurbaren knee-point puntua aurresateko
            prediction(self.dataset, self.df_summary, self.df_battery, self.input, self.to_predict, 100, self.group, self.charge_policies)
    
        
                   
              
    def every_Qmax_eval(self):
                    bat_names = self.df_battery.bat_name.unique()

                    # Create a colormap and normalize
                    cmap = cm.get_cmap('tab20')  # You can use other colormaps like 'viridis', 'plasma', etc.
                    norm = mcolors.Normalize(vmin=0, vmax=len(bat_names) - 1)

                    # Assign a unique color to each battery
                    battery_colors = {name: cmap(norm(i)) for i, name in enumerate(bat_names)}

                    # Create a figure and a set of subplots
                    fig, ax = plt.subplots(figsize=(10, 6))  # Use plt.subplots() to get both fig and ax

                    # Plot each battery with its assigned color
                    for name in self.df_battery.bat_name.unique():
                        df = self.df_battery[self.df_battery['bat_name'] == name].reset_index(drop=True)
                        ax.scatter(df['cycle'], df['Q_value'], s=10, color=battery_colors[name])

                    ax.set_xlabel('Cycle')
                    ax.set_ylabel('Capacity')
                    #ax.legend(title="Batteries")  # Add a legend to identify batteries
                    plt.title("Battery Capacity vs Cycle")  # Add a title to the plot
                    plt.show()

                    # Definir los colores para las 8 categorías
                    color_map = {
                        ("Slow C1", 1): "firebrick",
                        ("Slow C1", 2): "orangered",
                        ("Standard C1", 1): "olive",
                        ("Standard C1", 2): "lime",
                        ("Fast C1", 1): "teal",
                        ("Fast C1", 2): "darkturquoise",
                        ("Faster C1", 1): "indigo",
                        ("Faster C1", 2): "fuchsia"
                    }

                    marker_map = {1: 'o', 2: '^'}  # One-Step = Círculo, Two-Step = Triángulo
                    
                    fig, axes = plt.subplots(4, 2, figsize=(12, 16))  # Crear subgráficas 4x2
                    axes = axes.flatten()  # Convertir a array 1D para facilitar la iteración

                    # Para controlar los manejadores de leyenda
                    legend_handles = []
                    legend_labels = []

                    for idx, ((charge_policy, step_type), color) in enumerate(color_map.items()):
                        ax = axes[idx]  # Seleccionar el subplot correspondiente
                        
                        for name in self.df_battery.bat_name.unique():
                            df = self.df_battery[self.df_battery['bat_name'] == name].reset_index(drop=True)
                            if len(df.knee):  # Asegurarse de que haya datos
                                charge_category = df.charge_policy.iloc[0]
                                one_step = df.one_step.iloc[0]
                                
                                is_target = (charge_category == charge_policy) and (one_step == step_type)

                                # Resaltar la categoría objetivo, otras en gris transparente
                                plot_color = color if is_target else "gainsboro"
                                alpha_value = 1.0 if is_target else 0.01  # Color fuerte para la categoría objetivo, gris desvanecido para las demás

                                marker = marker_map.get(one_step, 'o')

                                # Crear el gráfico de dispersión en el subplot correspondiente
                                ax.scatter(df['cycle'], df['Q_value'], s=10, color=plot_color, marker=marker, alpha=alpha_value)

                        # Añadir la etiqueta (title) como label en cada gráfico
                        ax.text(0.5, 0.95, f"{charge_policy} - {'One-Step' if step_type == 1 else 'Two-Step'}", 
                                ha='center', va='top', fontsize=10, transform=ax.transAxes, color=color)

                        # Almacenar las entradas de la leyenda solo para cada categoría única
                        if (charge_policy, step_type) not in legend_labels:
                            legend_labels.append((charge_policy, step_type))
                            legend_handles.append(plt.Line2D([0], [0], marker=marker_map.get(step_type, 'o'), color='w', markerfacecolor=color, markersize=8))

                        # Establecer el label de x solo en la última fila (índices 6,7)
                        if idx >= 6:
                            ax.set_xlabel('Cycle')
                        
                        # Establecer el label de y solo en la primera columna (índices 0, 2, 4, 6)
                        if idx % 2 == 0:
                            ax.set_ylabel('Capacity')

                    # Añadir la leyenda solo una vez al lado derecho
                    # plt.legend(handles=legend_handles, labels=[f"{label} - {'One-Step' if step == 1 else 'Two-Step'}" for label, step in legend_labels], title="Charge Policy & Steps", bbox_to_anchor=(1.05, 1), loc='upper left')

                    plt.tight_layout()
                    plt.show()

                    '''# Capacity ratio plot (histogram)
                    # 1) Calcular la dif entre Qd100 y Qd2
                    ratios = []
                    for bat in self.df_battery.bat_name.unique():
                        df = self.df_battery[self.df_battery.bat_name == bat]
                        if not df[df.cycle == 100].empty and not df[df.cycle == 2].empty:
                            ratio = df[df.cycle == 100]['max_Qd'].values / df[df.cycle == 2]['max_Qd'].values
                            ratios.append(ratio[0])  # Assuming there's only one value per cycle
                    plt.figure()
                    plt.hist(ratios, bins=30, edgecolor='black', linewidth=1.5)
                    plt.xlabel('Ratio of max Qd (Cycle 100 / Cycle 2)')
                    plt.ylabel('Frequency')
                    plt.title('Histogram of Ratios')
                    plt.savefig("./Results/life_cycle_prediction/capacity_degradation_ratio.png")
                    plt.show()'''
                    return

    def every_knee_eval(self):
                    bat_names = self.df_battery.bat_name.unique()

                    # Create a colormap and normalize
                    cmap = cm.get_cmap('tab20')  # You can use other colormaps like 'viridis', 'plasma', etc.
                    norm = mcolors.Normalize(vmin=0, vmax=len(bat_names) - 1)

                    # Assign a unique color to each battery
                    battery_colors = {name: cmap(norm(i)) for i, name in enumerate(bat_names)}

                    # Create a figure and a set of subplots
                    fig, ax = plt.subplots(figsize=(8, 6))  # Use plt.subplots() to get both fig and ax

                    # Plot each battery with its assigned color
                    for name in self.df_battery.bat_name.unique():
                        df = self.df_battery[self.df_battery['bat_name'] == name].reset_index(drop=True)
                        ax.scatter(df['cycle'], df['knee'], s=10, color=battery_colors[name])
                    ax.set_xlabel(r'$\text{Cycle}$', fontsize=16)
                    ax.set_ylabel(r'$\text{Discharge Capacity (Q) at Knee points}$', fontsize=14) 

                    #ax.legend(title="Batteries")  # Add a legend to identify batteries
                    #plt.title("Battery knees vs Cycle")  # Add a title to the plot
                    plt.tight_layout()
                    plt.show()


                    # Definir los colores para las 8 categorías
                    color_map = {
                        ("Slow C1", 1): "firebrick",
                        ("Slow C1", 2): "orangered",
                        ("Standard C1", 1): "olive",
                        ("Standard C1", 2): "lime",
                        ("Fast C1", 1): "teal",
                        ("Fast C1", 2): "darkturquoise",
                        ("Faster C1", 1): "indigo",
                        ("Faster C1", 2): "fuchsia"
                    }

                    marker_map = {1: 'o', 2: '^'}  # One-Step = Círculo, Two-Step = Triángulo
                    
                    fig, axes = plt.subplots(4, 2, figsize=(8, 6))  # Crear subgráficas 4x2
                    axes = axes.flatten()  # Convertir a array 1D para facilitar la iteración

                    # Para controlar los manejadores de leyenda
                    legend_handles = []
                    legend_labels = []

                    for idx, ((charge_policy, step_type), color) in enumerate(color_map.items()):
                        ax = axes[idx]  # Seleccionar el subplot correspondiente
                        
                        for name in self.df_battery.bat_name.unique():
                            df = self.df_battery[self.df_battery['bat_name'] == name].reset_index(drop=True)
                            if len(df.knee):  # Asegurarse de que haya datos
                                charge_category = df.charge_policy.iloc[0]
                                one_step = df.one_step.iloc[0]
                                
                                is_target = (charge_category == charge_policy) and (one_step == step_type)

                                # Resaltar la categoría objetivo, otras en gris transparente
                                plot_color = color if is_target else "gainsboro"
                                alpha_value = 1.0 if is_target else 0.01  # Color fuerte para la categoría objetivo, gris desvanecido para las demás

                                marker = marker_map.get(one_step, 'o')

                                # Crear el gráfico de dispersión en el subplot correspondiente
                                ax.scatter(df['cycle'], df['knee'], s=10, color=plot_color, marker=marker, alpha=alpha_value)

                        # Añadir la etiqueta (title) como label en cada gráfico
                        ax.text(0.5, 0.95, f"{charge_policy} - {'One-Step' if step_type == 1 else 'Two-Step'}", 
                                ha='center', va='top', fontsize=10, transform=ax.transAxes, color=color)

                        # Almacenar las entradas de la leyenda solo para cada categoría única
                        if (charge_policy, step_type) not in legend_labels:
                            legend_labels.append((charge_policy, step_type))
                            legend_handles.append(plt.Line2D([0], [0], marker=marker_map.get(step_type, 'o'), color='w', markerfacecolor=color, markersize=8))

                        # Establecer el label de x solo en la última fila (índices 6,7)
                        if idx >= 6:
                            ax.set_xlabel('Cycle', fontsize = 14)
                        
                        # Establecer el label de y solo en la primera columna (índices 0, 2, 4, 6)
                        if idx % 2 == 0:
                            ax.set_ylabel('Qd X coord', fontsize=14)

                    # Añadir la leyenda solo una vez al lado derecho
                    # plt.legend(handles=legend_handles, labels=[f"{label} - {'One-Step' if step == 1 else 'Two-Step'}" for label, step in legend_labels], title="Charge Policy & Steps", bbox_to_anchor=(1.05, 1), loc='upper left')

                    plt.tight_layout()
                    plt.show()
                    return
                
    def plot_comparing_models(self):
        cycle_life_capacity = pd.read_pickle('./data/processed/NASA/results_total_cycles_capacity.pkl')
        cycle_life_knees = pd.read_pickle('./data/processed/NASA/results_total_cycles_knees.pkl')
        knee_capacity = pd.read_pickle('./data/processed/NASA/results_cycle_knee_of_capacity.pkl')
        knee_knees = pd.read_pickle('./data/processed/NASA/results_cycle_knee_of_knees.pkl')
        # Graficar
        '''plt.figure(figsize=(8, 6))
        plt.plot(range(30, 100 + 1, 10), cycle_life_capacity.errors_RMSE, label="Cycle-life with capacity", marker='o')
        plt.fill_between(range(30, 100 + 1, 10), cycle_life_capacity.lower_bounds, cycle_life_capacity.upper_bounds, color='b', alpha=0.2) # , label="95% CI")

        plt.plot(range(30, 100 + 1, 10), cycle_life_knees.errors_RMSE, label="Cycle-life with knees", marker='o')
        plt.fill_between(range(30, 100 + 1, 10), cycle_life_knees.lower_bounds, cycle_life_knees.upper_bounds, color='r', alpha=0.2) #, label="95% CI")
        plt.xlabel("Number of cycles", fontsize=16)
        plt.ylabel("RMSE", fontsize=16)
        # plt.title("Error predicting Cycle-life vs Number of cycles")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(range(30, 100 + 1, 10), knee_capacity.errors_RMSE, label="Knee-point with capacity", marker='o')
        plt.fill_between(range(30, 100 + 1, 10), knee_capacity.lower_bounds, knee_capacity.upper_bounds, color='b', alpha=0.2) # , label="95% CI")

        plt.plot(range(30, 100 + 1, 10), knee_knees.errors_RMSE, label="Knee-point with knees", marker='o')
        plt.fill_between(range(30, 100 + 1, 10), knee_knees.lower_bounds, knee_knees.upper_bounds, color='r', alpha=0.2) #, label="95% CI")
        plt.xlabel("Number of cycles", fontsize=14)
        plt.ylabel("RMSE", fontsize=14)
        plt.title("Error predicting degradation knee-pont vs Number of cycles")
        plt.legend()
        plt.tight_layout()
        plt.show()'''


        plt.figure(figsize=(8, 6))

        plt.plot(range(30, 100 + 1, 10), cycle_life_capacity.errors_RMSE, label="Cycle-life cut-off voltage", marker='o')
        plt.fill_between(range(30, 100 + 1, 10), cycle_life_capacity.lower_bounds, cycle_life_capacity.upper_bounds, color='b', alpha=0.2) # , label="95% CI")

        plt.plot(range(30, 100 + 1, 10), cycle_life_knees.errors_RMSE, label="Cycle-life knees", marker='o')
        plt.fill_between(range(30, 100 + 1, 10), cycle_life_knees.lower_bounds, cycle_life_knees.upper_bounds, color='orange', alpha=0.2) #, label="95% CI")

        plt.plot(range(30, 100 + 1, 10), knee_capacity.errors_RMSE, label="Knee-point cut-off voltage", marker='^')
        plt.fill_between(range(30, 100 + 1, 10), knee_capacity.lower_bounds, knee_capacity.upper_bounds, color='green', alpha=0.2) # , label="95% CI")

        plt.plot(range(30, 100 + 1, 10), knee_knees.errors_RMSE, label="Knee-point knees", marker='^')
        plt.fill_between(range(30, 100 + 1, 10), knee_knees.lower_bounds, knee_knees.upper_bounds, color='r', alpha=0.2) #, label="95% CI")

        
        # Plot the MPE values on the second y-axis (right side)
        #ax2.plot(range(30, 100 + 1, 10), cycle_life_capacity.errors_MPE, label="Cycle-life cut-off voltage (MPE)", linestyle='--', color='b')
        #ax2.plot(range(30, 100 + 1, 10), cycle_life_knees.errors_MPE, label="Cycle-life knees (MPE)", linestyle='--', color='orange')
        #ax2.plot(range(30, 100 + 1, 10), knee_capacity.errors_MPE, label="Knee-point cut-off voltage (MPE)", linestyle='--', color='green')
        #ax2.plot(range(30, 100 + 1, 10), knee_knees.errors_MPE, label="Knee-point knees (MPE)", linestyle='--', color='r')

        # Set labels for both axes
        plt.xlabel("Number of cycles", fontsize=16)
        plt.ylabel("RMSE", fontsize=16)
        plt.legend(fontsize=16)

        ax2 = plt.gca().twinx()
        ax2.set_ylabel("MPE", fontsize=16)

        # Add the legend
        
        #ax2.legend(fontsize=16, loc='upper right')

        plt.tight_layout()
        plt.show()
        return

args = get_args_parser().parse_args()
Knee_Curve(args)
#knee_2 = Knee_Curve(args)