import argparse
import torch
import shap
import os, sys
import glob
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestRegressor
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from skorch import NeuralNetRegressor
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.knee_calculator_cycle import knee_calculator, knee_calculator_karga, knee

########################################################
#         RF / XGBOOST / NN grafikoak   100ziklo hartuz
########################################################

########################
#to_predict = knee_point
########################
def ereduengrafikoa_kneepoint():
    deskargako_datuak_kp = pd.read_pickle('./data/processed/NASA/NASA_deskarga/deskarga_kp_prediction.pkl')
    deskargako_datuak_kp['Dataset'] = 'Deskarga' #Dataset zutabe gehitu eta bertan deskarga idatzi
    kargako_datuak_kp = pd.read_pickle('./data/processed/NASA/NASA_karga/karga_kp_prediction.pkl')
    kargako_datuak_kp['Dataset'] = 'Karga' #Dataset zutabea gehitu eta bertan karga idatzi

    # Bi Dataframeak Batu, horrela df_plot_batua biak elkartuz sortu
    df_plot_batuak_kp = pd.concat([kargako_datuak_kp, deskargako_datuak_kp], ignore_index=True) 
    # Datuen Egitura Aldatu
    df_long_kp = pd.melt(
        df_plot_batuak_kp,
        id_vars=['Y_test', 'Dataset'],
        value_vars=['Y_pred_rf', 'Y_pred_XGB', 'Y_pred_NN'],
        var_name='Eredua', # Zutabe berria: Ereduaren izena
        value_name='Y_pred'  # Zutabe berria: Aurreikuspen balioa
    )
    eredu_mapa = {'Y_pred_rf': 'Random Forest', 'Y_pred_XGB': 'XGBoost', 'Y_pred_NN': 'NN'}
    df_long_kp['Eredua'] = df_long_kp['Eredua'].map(eredu_mapa)

    # Ikurrak zehaztu: Karga -> Borobila ('o'), Deskarga -> Triangelua ('^')
    markers_dict = {'Karga': 'o', 'Deskarga': '^'} 
    kolore_eskala = {'Random Forest': 'tab:blue', 'XGBoost': 'tab:red', 'NN': 'tab:green'}

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x='Y_test', y='Y_pred', data=df_long_kp,hue='Eredua', style='Dataset', markers=markers_dict, palette= kolore_eskala, ax=ax, alpha=0.6,s=80)
        
    # --- Erreferentzia Lerroa eta Etiketak ---
    min_val = df_plot_batuak_kp['Y_test'].min()
    max_val = df_plot_batuak_kp['Y_test'].max()
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=2, label="Perfect Fit (y=x)")

    # --- Azken Ukituak ---
    plt.xlabel("Benetako Balioak (Knee Point)", fontsize=12)
    plt.ylabel("Aurreikusitako Balioak (Knee Point)", fontsize=12)
    plt.title("Ereduen aurreikuspenen konparaketa (Karga vs Deskarga) - Knee Point", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    '''
    # Legenda automatikoki sortu (modeloak eta dataset ikurrak barne)
    handles, labels = ax.get_legend_handles_labels()
    # Eremu asko daudenez, legendaren zatiak bereiz ditzakegu (Modeloak vs Datasetak)
    eredu_kopurua = 3
    dataset_kopurua = 2
    # Modeloen etiketak (lehenengo sarrerak)
    model_handles = handles[0 : eredu_kopurua]
    model_labels = labels[0 : eredu_kopurua]
    # Dataset etiketak (ondorengo sarrerak)
    dataset_handles = handles[eredu_kopurua : eredu_kopurua + dataset_kopurua]
    dataset_labels = labels[eredu_kopurua : eredu_kopurua + dataset_kopurua]
    perfect_fit_handle = handles[-1]
    perfect_fit_label = labels[-1]
    all_handles = model_handles + dataset_handles + [perfect_fit_handle]
    all_labels = model_labels + dataset_labels + [perfect_fit_label]
    # Legendak banatuta marraztu garbitasun hobea lortzeko
    ax.legend(
        all_handles, 
        all_labels, 
        loc='upper left', 
        bbox_to_anchor=(1.05, 1)
    )
    '''
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./results/ereduen_aurreikuspenaren_konparaketa/knee_point aurreikusi.png')

#ereduengrafikoa_kneepoint() 

#EREDUKA --> Aukerak = 'Random Forest' / ' XGBoost' / 'NN'
def ereduengrafikoa_kneepoint_EREDUKA(eredua):
    deskargako_datuak_kp = pd.read_pickle('./data/processed/NASA/NASA_deskarga/deskarga_kp_prediction.pkl')
    deskargako_datuak_kp['Dataset'] = 'Deskarga'
    kargako_datuak_kp = pd.read_pickle('./data/processed/NASA/NASA_karga/karga_kp_prediction.pkl')
    kargako_datuak_kp['Dataset'] = 'Karga' 
    df_plot_batuak_kp = pd.concat([kargako_datuak_kp, deskargako_datuak_kp], ignore_index=True)

    df_long_kp = pd.melt(df_plot_batuak_kp,id_vars=['Y_test', 'Dataset'],value_vars=['Y_pred_rf', 'Y_pred_XGB', 'Y_pred_NN'],var_name='Eredua',value_name='Y_pred')
    
    # Eredu mapaketa (Legenda argiagoa izateko)
    eredu_mapa = {'Y_pred_rf': 'Random Forest', 'Y_pred_XGB': 'XGBoost', 'Y_pred_NN': 'NN'}
    df_long_kp['Eredua'] = df_long_kp['Eredua'].map(eredu_mapa)

    eredu_hautatua = eredua
    df_eredua = df_long_kp[df_long_kp['Eredua'] == eredu_hautatua].copy()
    markers_dict = {'Karga': 'o', 'Deskarga': '^'} 
    # Koloreak Dataset-aren arabera bereiziko dira
    kolore_deskarga_karga = {'Karga': 'darkblue', 'Deskarga': 'darkred'}
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Y_test', y='Y_pred', data=df_eredua,hue='Dataset', style='Dataset',markers=markers_dict,palette=kolore_deskarga_karga,ax=ax, alpha=0.6,s=80)
    min_val = df_plot_batuak_kp['Y_test'].min()
    max_val = df_plot_batuak_kp['Y_test'].max()
    ax.plot([min_val, max_val], [min_val, max_val], 
            linestyle="--", color="black", linewidth=2, 
            label="Egokitzapen perfektua (y=x)")
    ax.set_xlabel("Benetako Balioak (Knee Point)", fontsize=12)
    ax.set_ylabel("Aurreikusitako Balioak (Knee Point)", fontsize=12)
    ax.set_title(f"{eredu_hautatua} Ereduaren Aurreikuspena (Karga vs Deskarga)", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./results/ereduen_aurreikuspenaren_konparaketa/{eredua}_kneepoint_aurreikusi.png')

# ereduengrafikoa_kneepoint_EREDUKA('Random Forest')
# ereduengrafikoa_kneepoint_EREDUKA('XGBoost')
# ereduengrafikoa_kneepoint_EREDUKA('NN')


########################
#to_predict = cycle_life
########################
def ereduengrafikoa_cyclelife(hasi):
    deskargako_datuak_cl = pd.read_pickle('./data/processed/NASA/NASA_deskarga/deskarga_cl_prediction.pkl')
    deskargako_datuak_cl['Dataset'] = 'Deskarga' #Dataset zutabe gehitu eta bertan deskarga idatzi
    kargako_datuak_cl = pd.read_pickle('./data/processed/NASA/NASA_karga/karga_cl_prediction.pkl')
    kargako_datuak_cl['Dataset'] = 'Karga' #Dataset zutabea gehitu eta bertan karga idatzi

    # Bi Dataframeak Batu, horrela df_plot_batua biak elkartuz sortu
    df_plot_batuak_cl = pd.concat([kargako_datuak_cl, deskargako_datuak_cl], ignore_index=True) 
    df_long_cl = pd.melt(
        df_plot_batuak_cl,
        id_vars=['Y_test', 'Dataset'],
        value_vars=['Y_pred_rf', 'Y_pred_XGB', 'Y_pred_NN'],
        var_name='Eredua', # Zutabe berria: Ereduaren izena
        value_name='Y_pred'  # Zutabe berria: Aurreikuspen balioa
    )
    # Ikurrak zehaztu: Karga -> Borobila ('o'), Deskarga -> Triangelua ('^')
    markers_dict = {'Karga': 'o', 'Deskarga': '^'} 
    kolore_eskala_cl = {'Random Forest': 'tab:blue', 'XGBoost': 'tab:red', 'NN': 'tab:green'}

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x='Y_test', y='Y_pred', data=df_long_cl, hue='Eredua', style='Dataset',   markers=markers_dict, palette=kolore_eskala_cl, ax=ax, alpha=0.6,s=80)
    # --- Erreferentzia Lerroa eta Etiketak ---
    min_val_cl = df_plot_batuak_cl['Y_test'].min()
    max_val_cl = df_plot_batuak_cl['Y_test'].max()
    ax.plot([min_val_cl, max_val_cl], [min_val_cl, max_val_cl], linestyle="--", color="black", linewidth=2, label="Egokitzapen perfektua (y=x)")

    # --- Azken Ukituak ---
    plt.xlabel("Benetako Balioak (Cycle life)", fontsize=12)
    plt.ylabel("Aurreikusitako Balioak (Cycle life)", fontsize=12)
    plt.title("Ereduen aurreikuspenen konparaketa (Karga vs Deskarga) - Cycle life", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Legenda automatikoki sortu (modeloak eta dataset ikurrak barne)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./results/ereduen_aurreikuspenaren_konparaketa/cycle_life aurreikusi.png')

#EREDUKA --> Aukerak = 'Random Forest' / ' XGBoost' / 'NN'

def ereduengrafikoa_cyclelife_EREDUKA(eredua):
    deskargako_datuak_cl = pd.read_pickle('./data/processed/NASA/NASA_deskarga/deskarga_cl_prediction.pkl')
    deskargako_datuak_cl['Dataset'] = 'Deskarga' #Dataset zutabe gehitu eta bertan deskarga idatzi
    kargako_datuak_cl = pd.read_pickle('./data/processed/NASA/NASA_karga/karga_cl_prediction.pkl')
    kargako_datuak_cl['Dataset'] = 'Karga' #Dataset zutabea gehitu eta bertan karga idatzi

    # Bi Dataframeak Batu, horrela df_plot_batua biak elkartuz sortu
    df_plot_batuak_cl = pd.concat([kargako_datuak_cl, deskargako_datuak_cl], ignore_index=True) 
    df_long_cl = pd.melt(df_plot_batuak_cl,id_vars=['Y_test', 'Dataset'],value_vars=['Y_pred_rf', 'Y_pred_XGB', 'Y_pred_NN'],var_name='Eredua', value_name='Y_pred'  )
    # Eredu mapaketa (Legenda argiagoa izateko)
    eredu_mapa = {'Y_pred_rf': 'Random Forest', 'Y_pred_XGB': 'XGBoost', 'Y_pred_NN': 'NN'}
    df_long_cl['Eredua'] = df_long_cl['Eredua'].map(eredu_mapa)

    eredu_hautatua = eredua
    df_eredua = df_long_cl[df_long_cl['Eredua'] == eredu_hautatua].copy()
    markers_dict = {'Karga': 'o', 'Deskarga': '^'} 
    # Koloreak Dataset-aren arabera bereiziko dira
    kolore_deskarga_karga = {'Karga': 'darkblue', 'Deskarga': 'darkred'}
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Y_test', y='Y_pred', data=df_eredua,hue='Dataset', style='Dataset',markers=markers_dict,palette=kolore_deskarga_karga,ax=ax, alpha=0.6,s=80)
    min_val = df_plot_batuak_cl['Y_test'].min()
    max_val = df_plot_batuak_cl['Y_test'].max()
    ax.plot([min_val, max_val], [min_val, max_val], 
            linestyle="--", color="black", linewidth=2, 
            label="Egokitzapen perfektua (y=x)")
    ax.set_xlabel("Benetako Balioak (Bizi-zikloa)", fontsize=12)
    ax.set_ylabel("Aurreikusitako Balioak (Bizi-zikloa)", fontsize=12)
    ax.set_title(f"{eredu_hautatua} Ereduaren Aurreikuspena (Karga vs Deskarga)", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./results/ereduen_aurreikuspenaren_konparaketa/{eredua}__cyclelife_aurreikusi.png')

#ereduengrafikoa_cyclelife_EREDUKA('Random Forest')
# ereduengrafikoa_cyclelife_EREDUKA('XGBoost')
# ereduengrafikoa_cyclelife_EREDUKA('NN')


###########################################
#       RMSEaren grafikoa egiteko
###########################################

def rmse_grafikoa(eredua):
    # Ziklo kopuruak (X ardatza)
    cycles = np.array([100, 115, 130, 145, 160])

    # 1. KARGA -> CYCLE_LIFE (Lerro Urdina, Circle markatzailea)
    rmse_karga_cl_rf = np.array([364.5154, 359.9484, 337.0202, 329.6466, 314.4514])
    rmse_karga_cl_xg = np.array([395.2552, 387.065, 384.4245, 385.0798, 380.3439])
    rmse_karga_cl_nn = np.array([367.119, 362.967, 345.0431, 338.0853, 317.3153]) 

    # 2. KARGA -> KNEE_POINT (Lerro Berdea, Triangle markatzailea)
    rmse_karga_kp_rf = np.array([256.5848, 254.4119, 231.9204, 227.1565, 214.2694])
    rmse_karga_kp_xg = np.array([255.9685, 251.0967, 235.347, 257.8394, 257.8714])
    rmse_karga_kp_nn = np.array([262.4591, 256.6282, 245.4827, 234.3264, 223.6936]) 

    # 3. DESKARGA -> CYCLE_LIFE (Lerro Laranja, Circle markatzailea)
    rmse_deskarga_cl_rf = np.array([325.7143, 332.4873, 320.6124, 288.7768, 270.682])
    rmse_deskarga_cl_xg = np.array([309.2677, 297.9942, 275.3498, 302.3164, 296.7573])
    rmse_deskarga_cl_nn = np.array([306.0515, 291.9090, 257.188, 251.9132, 240.2034]) 

    # 4. DESKARGA -> KNEE_POINT (Lerro Gorria, Triangle markatzailea)
    rmse_deskarga_kp_rf = np.array([246.8482, 245.4718, 223.0496, 211.115, 186.6782])
    rmse_deskarga_kp_xg = np.array([228.4492, 238.2927, 222.3095, 216.2534, 193.2164])
    rmse_deskarga_kp_nn = np.array([235.4609, 217.8786, 196.3811, 186.8321, 171.8457]) 


    # Ereduen datuak: [Karga_CL, Karga_KP, Deskarga_CL, Deskarga_KP]
    ereduak_datuak = {
        "rf": [rmse_karga_cl_rf, rmse_karga_kp_rf, rmse_deskarga_cl_rf, rmse_deskarga_kp_rf],
        "xg": [rmse_karga_cl_xg, rmse_karga_kp_xg, rmse_deskarga_cl_xg, rmse_deskarga_kp_xg],
        "nn": [rmse_karga_cl_nn, rmse_karga_kp_nn, rmse_deskarga_cl_nn, rmse_deskarga_kp_nn]
    }
    
    # Ereduaren izen irakurgarriak
    ereduen_mapa = {"rf": "Random Forest","xg": "XGBoost","nn": "FNN"}
    
    # Hautatutako eredua berreskuratu
    if eredua.lower() not in ereduak_datuak:
        print(f"Errorea: '{eredua}' eredua ez da aurkitu datuetan.")
        return
    
    rmse_data = ereduak_datuak[eredua.lower()]
    model_title = ereduen_mapa[eredua.lower()]

    
    # GRAFIKOA SORTU (2x1 Subplot)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'{model_title} ereduaren RMSE balioen konparaketa', fontsize=16, fontweight='bold')
    
    # 1. BIZI-ZIKLOA AURRESATEKO SUBPLOTA (Cycle Life - CL)
    
    # Kargako CL (Indizea 0) - Lerro Urdina, Circle
    ax1.plot(cycles, rmse_data[0], marker='o', color='tab:blue', 
             label='Kargako datuak (Bizi-zikloa)')
    
    # Deskargako CL (Indizea 2) - Lerro Laranja, Circle
    ax1.plot(cycles, rmse_data[2], marker='o', color='tab:orange', 
             label='Deskargako datuak (Bizi-zikloa)')
    
    # Bizi-zikloko lerro horizontala (Deskarga CL, 100 ziklo)
    if rmse_data[2].size > 0:
        deskarga_cl_100 = rmse_data[2][0]
        ax1.axhline(y=deskarga_cl_100, color='black', linestyle='--', linewidth=1.5,
                    label=f'Deskarga - 100 ziklo')
    
    ax1.set_title('Bizi-zikloa aurresatea', fontsize=14)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. KNEE-POINT AURRESATEKO SUBPLOTA (KP)
    
    # Kargako KP (Indizea 1) - Lerro Berdea, Triangle
    ax2.plot(cycles, rmse_data[1], marker='^', color='tab:green', 
             label='Kargako datuak (Knee-point)')
    
    # Deskargako KP (Indizea 3) - Lerro Gorria, Triangle
    ax2.plot(cycles, rmse_data[3], marker='^', color='tab:red', 
             label='Deskargako datuak (Knee-point)')
    
    # Knee-point-eko lerro horizontala (Deskarga KP, 100 ziklo)
    if rmse_data[3].size > 0:
        deskarga_kp_100 = rmse_data[3][0]
        ax2.axhline(y=deskarga_kp_100, color='black', linestyle='--', linewidth=1.5,
                    label=f'Deskarga - 100 ziklo')
    
    ax2.set_title('Knee-point aurresatea', fontsize=14)
    ax2.set_xlabel('Ziklo kopurua', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)

    # Goiko grafikoaren (ax1) x ardatzean ziklo kopurua
    ax1.tick_params(labelbottom=True)
    ax1.set_xticks(cycles)
    ax1.set_xticklabels(cycles)

    # Beheko grafikoan ere (ax2) ziurtatu agertzen direla
    ax2.set_xticks(cycles)
    ax2.set_xticklabels(cycles)

    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Y ardatzaren mugak finkatu (grafikoen arteko konparaketa hobea izateko)
    # Datu guztiak konbinatu, numpy array hutsa ez bada
    all_rmse_data = [d for d in rmse_data if d.size > 0]
    if all_rmse_data:
        # Datu guztietako min eta max balioak aurkitu
        y_min = min(map(min, all_rmse_data))
        y_max = max(map(max, all_rmse_data))
        
        # Muga orokorrak ezarri, tarte bat utziz
        # y_min 50etik gora ezarri behar bada
        y_limit_min = max(50, y_min - 20)
        y_limit_max = y_max + 20
        
        # Bi subplot-etan muga berdinak ezarri
        ax1.set_ylim(y_limit_min, y_limit_max)
        ax2.set_ylim(y_limit_min, y_limit_max)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    fitxategi_izena = f'rmse_subplot_konparaketa_{eredua.lower()}.png'
    path = os.path.join('./results/rmse_konparaketa', fitxategi_izena) 
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    plt.savefig(path)
    plt.close(fig) 
    print(f"'{model_title}' ereduko grafikoa (subplot-ak) gorde da hemen: {path}")


# rmse_grafikoa('rf')
# rmse_grafikoa('xg')
# rmse_grafikoa('nn')

def rmse_grafikoa_ing(eredua):
    # Ziklo kopuruak (X ardatza)
    cycles = np.array([100, 115, 130, 145, 160])

    # 1. KARGA -> CYCLE_LIFE (Lerro Urdina, Circle markatzailea)
    rmse_karga_cl_rf = np.array([364.5154, 359.9484, 337.0202, 329.6466, 314.4514])
    rmse_karga_cl_xg = np.array([395.2552, 387.065, 384.4245, 385.0798, 380.3439])
    rmse_karga_cl_nn = np.array([367.119, 362.967, 345.0431, 338.0853, 317.3153]) 

    # 2. KARGA -> KNEE_POINT (Lerro Berdea, Triangle markatzailea)
    rmse_karga_kp_rf = np.array([256.5848, 254.4119, 231.9204, 227.1565, 214.2694])
    rmse_karga_kp_xg = np.array([255.9685, 251.0967, 235.347, 257.8394, 257.8714])
    rmse_karga_kp_nn = np.array([262.4591, 256.6282, 245.4827, 234.3264, 223.6936]) 

    # 3. DESKARGA -> CYCLE_LIFE (Lerro Laranja, Circle markatzailea)
    rmse_deskarga_cl_rf = np.array([325.7143, 332.4873, 320.6124, 288.7768, 270.682])
    rmse_deskarga_cl_xg = np.array([309.2677, 297.9942, 275.3498, 302.3164, 296.7573])
    rmse_deskarga_cl_nn = np.array([306.0515, 291.9090, 257.188, 251.9132, 240.2034]) 

    # 4. DESKARGA -> KNEE_POINT (Lerro Gorria, Triangle markatzailea)
    rmse_deskarga_kp_rf = np.array([246.8482, 245.4718, 223.0496, 211.115, 186.6782])
    rmse_deskarga_kp_xg = np.array([228.4492, 238.2927, 222.3095, 216.2534, 193.2164])
    rmse_deskarga_kp_nn = np.array([235.4609, 217.8786, 196.3811, 186.8321, 171.8457]) 


    # Ereduen datuak: [Karga_CL, Karga_KP, Deskarga_CL, Deskarga_KP]
    ereduak_datuak = {
        "rf": [rmse_karga_cl_rf, rmse_karga_kp_rf, rmse_deskarga_cl_rf, rmse_deskarga_kp_rf],
        "xg": [rmse_karga_cl_xg, rmse_karga_kp_xg, rmse_deskarga_cl_xg, rmse_deskarga_kp_xg],
        "nn": [rmse_karga_cl_nn, rmse_karga_kp_nn, rmse_deskarga_cl_nn, rmse_deskarga_kp_nn]
    }
    
    # Ereduaren izen irakurgarriak
    ereduen_mapa = {"rf": "Random Forest","xg": "XGBoost","nn": "FNN"}
    
    # Hautatutako eredua berreskuratu
    if eredua.lower() not in ereduak_datuak:
        print(f"Errorea: '{eredua}' eredua ez da aurkitu datuetan.")
        return
    
    rmse_data = ereduak_datuak[eredua.lower()]
    model_title = ereduen_mapa[eredua.lower()]

    
    # GRAFIKOA SORTU (2x1 Subplot)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'Comparison of RMSE values of the {model_title} model', fontsize=16, fontweight='bold')
    
    # 1. BIZI-ZIKLOA AURRESATEKO SUBPLOTA (Cycle Life - CL)
    
    # Kargako CL (Indizea 0) - Lerro Urdina, Circle
    ax1.plot(cycles, rmse_data[0], marker='o', color='tab:blue', 
             label='Charge (cycle life)')
    
    # Deskargako CL (Indizea 2) - Lerro Laranja, Circle
    ax1.plot(cycles, rmse_data[2], marker='o', color='tab:orange', 
             label='Discharge (cycle life)')
    
    # Bizi-zikloko lerro horizontala (Deskarga CL, 100 ziklo)
    if rmse_data[2].size > 0:
        deskarga_cl_100 = rmse_data[2][0]
        ax1.axhline(y=deskarga_cl_100, color='black', linestyle='--', linewidth=1.5,
                    label=f'Discharge - 100 cycles')
    
    ax1.set_title('Cycle-life prediction', fontsize=14)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. KNEE-POINT AURRESATEKO SUBPLOTA (KP)
    
    # Kargako KP (Indizea 1) - Lerro Berdea, Triangle
    ax2.plot(cycles, rmse_data[1], marker='^', color='tab:green', 
             label='Charge (degradation knee point)')
    
    # Deskargako KP (Indizea 3) - Lerro Gorria, Triangle
    ax2.plot(cycles, rmse_data[3], marker='^', color='tab:red', 
             label='Discharge (degradation knee-point)')
    
    # Knee-point-eko lerro horizontala (Deskarga KP, 100 ziklo)
    if rmse_data[3].size > 0:
        deskarga_kp_100 = rmse_data[3][0]
        ax2.axhline(y=deskarga_kp_100, color='black', linestyle='--', linewidth=1.5,
                    label=f'Discharge - 100 cycles')
    
    ax2.set_title('Degradation knee point prediction', fontsize=14)
    ax2.set_xlabel('Number of cycles', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Y ardatzaren mugak finkatu (grafikoen arteko konparaketa hobea izateko)
    # Datu guztiak konbinatu, numpy array hutsa ez bada
    all_rmse_data = [d for d in rmse_data if d.size > 0]
    if all_rmse_data:
        # Datu guztietako min eta max balioak aurkitu
        y_min = min(map(min, all_rmse_data))
        y_max = max(map(max, all_rmse_data))
        
        # Muga orokorrak ezarri, tarte bat utziz
        # y_min 50etik gora ezarri behar bada
        y_limit_min = max(50, y_min - 20)
        y_limit_max = y_max + 20
        
        # Bi subplot-etan muga berdinak ezarri
        ax1.set_ylim(y_limit_min, y_limit_max)
        ax2.set_ylim(y_limit_min, y_limit_max)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    fitxategi_izena = f'rmse_subplot_konparaketa_ing_{eredua.lower()}.png'
    path = os.path.join('./results/rmse_konparaketa_ing', fitxategi_izena) 
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    plt.savefig(path)
    plt.close(fig) 
    print(f"'{model_title}' ereduko grafikoa (subplot-ak) gorde da hemen: {path}")

# rmse_grafikoa_ing('rf')
# rmse_grafikoa_ing('xg')
# rmse_grafikoa_ing('nn')

#############################################################
#        Deskarga/karga knee-point diferentzia (histograma)
#############################################################

def knee_point_diferentzia():
    kargako_datuak = pd.read_pickle('./data/processed/NASA/NASA_karga/DataSetRulEstimation_all_new.pkl')
    karga_q_value = kargako_datuak[['bat_name', 'cycle', 'Q_value']].copy()

    deskargako_datuak = pd.read_pickle('./data/processed/NASA/NASA_deskarga/DataSetRulEstimation_all_new.pkl')
    deskarga_q_value = deskargako_datuak[['bat_name', 'cycle', 'Q_value']].copy()

    
    # Knee Point-a kalkulatu Kargako datuekin
    print("Knee Point-a kalkulatzen (Karga)...")
    knees_karga = knee(karga_q_value, output='Q_value')
    knees_karga = knees_karga.rename(columns={'knee_cycle': 'knee_cycle_karga', 
                                                    'knee_capacity': 'knee_capacity_karga'})
    
    # Knee Point-a kalkulatu Deskargako datuekin
    print("Knee Point-a kalkulatzen (Deskarga)...")
    knees_deskarga = knee(deskarga_q_value, output='Q_value')
    knees_deskarga = knees_deskarga.rename(columns={'knee_cycle': 'knee_cycle_deskarga', #knee_cycle_deskarga knee point-aren zikloaren balioa
                                                          'knee_capacity': 'knee_capacity_deskarga'}) #knee_cycle_deskarga knee_pointean deskarga ahalmenaren balioa
    
    # Batu emaitzak bateriaren izenaren arabera
    df_knees_konparaketa = pd.merge(knees_karga[['bat_name', 'knee_cycle_karga']],
                                    knees_deskarga[['bat_name', 'knee_cycle_deskarga']],
                                    on='bat_name',
                                    how='inner')
    df_knees_konparaketa['Zikloen diferentzia'] = (df_knees_konparaketa['knee_cycle_deskarga'] - df_knees_konparaketa['knee_cycle_karga'])

    #Histograma sortu
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_knees_konparaketa,x='Zikloen diferentzia',
        bins=25, 
        ax=ax
    )
    
    ax.set_xlabel('Knee-point-aren Ziklo Diferentzia (Deskarga - Karga)', fontsize=12)
    ax.set_ylabel('Bateria kopurua', fontsize=12)
    ax.set_title('Knee Point-aren Ziklo-balioen Kenketa', fontsize=14)
    
    # 0-an lerro bertikala
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    # Fitxategia gorde
    path = './results/histogramak/knee_cycle_kenketa_histograma.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    return df_knees_konparaketa

#knee_point_diferentzia()

def knee_point_diferentzia_erlatiboa():
    kargako_datuak = pd.read_pickle('./data/processed/NASA/NASA_karga/DataSetRulEstimation_all_new.pkl')
    karga_q_value = kargako_datuak[['bat_name', 'cycle', 'Q_value']].copy()

    deskargako_datuak = pd.read_pickle('./data/processed/NASA/NASA_deskarga/DataSetRulEstimation_all_new.pkl')
    deskarga_q_value = deskargako_datuak[['bat_name', 'cycle', 'Q_value']].copy()

    
    # Knee Point-a kalkulatu Kargako datuekin
    print("Knee Point-a kalkulatzen (Karga)...")
    knees_karga = knee(karga_q_value, output='Q_value')
    knees_karga = knees_karga.rename(columns={'knee_cycle': 'knee_cycle_karga', 
                                                    'knee_capacity': 'knee_capacity_karga'})
    
    # Knee Point-a kalkulatu Deskargako datuekin
    print("Knee Point-a kalkulatzen (Deskarga)...")
    knees_deskarga = knee(deskarga_q_value, output='Q_value')
    knees_deskarga = knees_deskarga.rename(columns={'knee_cycle': 'knee_cycle_deskarga', #knee_cycle_deskarga knee point-aren zikloaren balioa
                                                          'knee_capacity': 'knee_capacity_deskarga'}) #knee_cycle_deskarga knee_pointean deskarga ahalmenaren balioa
    
    # Batu emaitzak bateriaren izenaren arabera
    df_knees_konparaketa = pd.merge(knees_karga[['bat_name', 'knee_cycle_karga']],
                                    knees_deskarga[['bat_name', 'knee_cycle_deskarga']],
                                    on='bat_name',
                                    how='inner')
    
    #Bateria bakoitzaren bizi-zikloa lortu (ziklo maximoa) --> Bi kasuetan berbera denez, deskargako datuak erabiliko ditugu ziklo kopurua hartzeko.
    azken_zikloa = deskarga_q_value.groupby('bat_name')['cycle'].max().reset_index()
    azken_zikloa.columns = ['bat_name', 'azken_zikloa']

    #Ziklo maximoa dataframe-ra gehitu
    df_knees_konparaketa=pd.merge(df_knees_konparaketa,azken_zikloa, on='bat_name', how='left')

    #Diferentzia normalizatua 
    df_knees_konparaketa['Zikloen diferentzia erlatiboa'] = ((df_knees_konparaketa['knee_cycle_deskarga'] - df_knees_konparaketa['knee_cycle_karga'])/df_knees_konparaketa['azken_zikloa'])*100

    #Histograma sortu
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_knees_konparaketa,x='Zikloen diferentzia erlatiboa',
        bins=25, 
        ax=ax
    )
    
    ax.set_xlabel('Diferentzia erlatiboa (%)', fontsize=12)
    ax.set_ylabel('Bateria kopurua', fontsize=12)
    ax.set_title('Knee Point diferentzia erlatiboa', fontsize=14)
    
    #0-an lerro bertikala gehitu
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    # Fitxategia gorde
    path = './results/histogramak/knee_cycle_erlatiboa_histograma.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    return df_knees_konparaketa

knee_point_diferentzia_erlatiboa()


#################################################################
#        Azken zikloa zein den jakiteko
###############################################################
def azken_zikloa():
    allfiles = glob.glob('./data/processed/NASA/*.csv')
    
    for indfile in range(len(allfiles)):
        # Fitxategia kargatu
        df_battery = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        
        # Azken zikloaren zenbakia lortu (.max() erabiliz)
        last_cycle = df_battery.cycle.max()
        
        # Fitxategiaren izen laburra (bide osoa gabe)
        file_name = allfiles[indfile].split('/')[-1]
        
        # Mezua erakutsi
        output_msg = f'Processing {file_name} - Azken zikloa: {last_cycle}'
        
        # BALDINTZA: 170 baino txikiagoa bada, KONTUZ gehitu
        if last_cycle < 170:
            print(f'{output_msg} --> KONTUZ: Ziklo gutxiegi!')
        else:
            print(output_msg)

# azken_zikloa()

def azken_zikloa_clean():
    file_path = './data/processed/NASA/NASA_karga/DataSetRulEstimation_all_new.pkl'
    df_all_batteries = pd.read_pickle(file_path)
    
    # 2. Bateria bakoitzeko azken zikloa kalkulatu
    last_cycles = df_all_batteries.groupby('bat_name')['cycle'].max()

    # 3. Bateria bakoitzaren emaitzak ikusi eta baldintza egiaztatu
    for bat_name, last_cycle in last_cycles.items():
        output_msg = f'Bateria: {bat_name} - Azken zikloa: {last_cycle}'
        
        # BALDINTZA: 200 baino txikiagoa bada
        if last_cycle < 200:
            print(f'{output_msg} --> KONTUZ: Ziklo gutxiegi!')
        else:
            print(output_msg)

# azken_zikloa_clean()