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
from gplearn.functions import make_function

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knee_calculator_cycle import knee_calculator, knee_calculator_karga, knee

#############################
#       OROKORRAK
#############################

def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knee evolution", add_help=True)
     
    parser.add_argument("--dataset", type=str, default="NASA_deskarga", required= True, choices=["NASA_deskarga", "NASA_karga", "Mendeley", "Zenodo"])
    # NASAren kasurako:
    parser.add_argument("--action", type=str, default="predict_knee", choices=["print_curves", "predict_curves", "predict_knee", "cycle_life"])
    parser.add_argument("--pol_order", type=int, default=20, required= False)
    parser.add_argument("--group", type=str, default=None, required= False, choices=["fast_charging", "two_steps"])
    parser.add_argument("--input", type=str, default="knee-charge", required= False, choices=["cut-of-voltage", "knee-charge", "knee_discharge"])
    parser.add_argument("--to_predict", type=str, default="knee_point", required= False, choices=["cycle_life", "knee_point"])
    parser.add_argument("--include_charge_policies", type=bool, default=False, required= False, choices=[True, False])

    parser.add_argument("--paper", type=int, default=1, choices=[1,2])
    return parser

# Datu-baseak:
    # NASA: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
    # Mendeley: https://data.mendeley.com/datasets/nsc7hnsg4s/2
    # Zenodo: https://zenodo.org/records/10963339

#############################
#      DATA LOADING
#############################

def get_charging_policy(df):
    # Filter charging phase (ignore discharge/rest)
    df_charge = df[df['I'] > 0].copy()

    # Compute C-rate
    df_charge['C_rate'] = df_charge['I'] / 1.1  # Assuming 1.1 Ah battery
    df_charge = df_charge[df_charge["C_rate"] > 0.1]
    df_charge['SOC'] = np.round((df_charge['V'] - 2)/ (3.6-2),3)

    C1 = np.round(max(df_charge.I[:50]),1)

    '''# Find the most common C-rate (C1)
    C1 = df_charge["C_rate"].value_counts().idxmax()
    # Find the index where C1 first appears
    first_C1_index = df_charge[df_charge["C_rate"] == C1].index[0]
    # Select C1 and the next 10 values
    C1_and_next_10 = df_charge.iloc[first_C1_index : first_C1_index + 11]  # 11 to include C1 itself
    C1 = np.round(np.median(C1_and_next_10.I), 3)'''

    # Find C2
    C2 = np.round(df_charge[df_charge['Qc'] <= (0.8*1.1)]['I'][-20:].quantile(0.5),1)  
    if C1 == C2:
        # One-step charge (C1 = C2)
        one_step = 1
    else:
        # Two-step charge
        one_step = 2

    # Classify charging type
    if C1 <= 3.6:
        charge_type = "Slow C1"
    elif 3.6 < C1 <= 4:
        charge_type = "Standard C1"
    elif 4 < C1 <= 5.4:
        charge_type = "Fast C1"
    else:
        charge_type = "Faster C1"
        
    return charge_type, one_step, C1, C2 

def intensity(dataset):
    if dataset =='NASA_deskarga' or dataset == 'NASA_karga':
        allfiles = glob.glob(f'./data/raw/NASA/*.csv')
        current_var = 'I'
        time_var = 't'
    elif dataset =='Zenodo':
        allfiles = glob.glob(f'./data/processed/{dataset}/preprocessed/*.csv')
        current_var = 'current_A'
        time_var = 'relative_time_min'

    charge_policy_info = []
    for indfile in range(len(allfiles)):
        file_path = allfiles[indfile]
        print('Processing File (or battery) ' + file_path)
        df_battery = pd.read_csv(file_path, sep=',', index_col=None)
        df_battery = df_battery.loc[:, ~df_battery.columns.str.contains("Unnamed")]
        df_battery['It'] = (df_battery[current_var]/200.)*(df_battery[time_var])
        
        bat_ind = file_path.split('\\')[1].split('_')[0] if dataset=='NASA_deskarga' or dataset=='NASA_karga' else file_path.split('\\')[1].split('.')[0]
        # os.path.splitext(os.path.basename(file_path_in_zip))[0]

        n_cycles = max(df_battery.cycle.unique())
        if dataset =='NASA_deskarga' or dataset=='NASA_karga':
            df_cycle = df_battery[df_battery['cycle'] == 1]
            charge_type, one_step, C1, C2 = get_charging_policy(df_cycle)  
            D1, D2 = 4, 4
            charge_policy_info.append([bat_ind, n_cycles, charge_type, one_step, C1, C2, D1, D2])
        elif dataset == 'Zenodo':
            policies = df_battery.loc[df_battery.cycle > 1,'description'].reset_index(drop=True)[0]
            C1 = policies.split('and')[0].split('C')[0]
            C2 = policies.split('and')[0].split('C')[0]
            D1 = policies.split('and')[1].split('C')[0]
            D2 = policies.split('and')[1].split('C')[0]
            charge_type, one_step = 'Zenodo', 1
            charge_policy_info.append([bat_ind, n_cycles, charge_type, one_step, C1, C2, D1, D2])


    charge_policy_df = pd.DataFrame(charge_policy_info, columns=['bat_name', 'n_cycles', 'charge_policy', 'one_step', 'C1', 'C2', 'D1', 'D2'])
    if dataset == 'NASA_deskarga' or dataset == 'NASA_karga':
        output_dir = f'./data/processed/NASA/{dataset}'
    else:
        output_dir = f'./data/processed/{dataset}'

    os.makedirs(output_dir, exist_ok=True)
    charge_policy_df.to_pickle(os.path.join(output_dir, 'charge_policies.pkl'))
    return

def load_intesities(dataset):
    #charge_policies = charge_policies.loc[:, ~charge_policies.columns.str.contains("Unnamed")]
    if dataset == 'NASA_karga':
        df = pd.read_pickle(f'./data/processed/NASA/NASA_karga/DataSetRulEstimation_all_new.pkl')
        charge_policies = pd.read_pickle(f'./data/processed/NASA/NASA_karga/charge_policies.pkl')
    elif dataset == 'NASA_deskarga':
        df = pd.read_pickle(f'./data/processed/NASA/NASA_deskarga/DataSetRulEstimation_all_new.pkl')
        charge_policies = pd.read_pickle(f'./data/processed/NASA/NASA_deskarga/charge_policies.pkl')

    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    # Make a copy of df
    df_new = df.copy()

    # Merge charge_policies with df_new based on battery index
    df_new = df_new.merge(charge_policies[['bat_name', 'charge_policy', 'one_step', 'C1', 'C2','D1', 'D2']], 
                        on='bat_name', how='left')

    # Save the new dataset
    if dataset == 'NASA_karga' or dataset == 'NASA_deskarga':
        df_new.to_pickle(f'./data/processed/NASA/{dataset}/RulEstimation_CompleteData.pkl')
    else:
        df_new.to_pickle(f'./data/processed/{dataset}/RulEstimation_CompleteData.pkl')
    return df_new

def data_loading(dataset):
    if dataset == "NASA_karga":
        data_file = './data/processed/NASA/NASA_karga/DataSetRulEstimation_all_new.pkl'                            
        charge_policies = './data/processed/NASA/NASA_karga/charge_policies.pkl'                                   
        data_complete ='./data/processed/NASA/NASA_karga/RulEstimation_CompleteData.pkl'                          
    elif dataset == 'NASA_deskarga':
        data_file = './data/processed/NASA/NASA_deskarga/DataSetRulEstimation_all_new.pkl'                            
        charge_policies = './data/processed/NASA/NASA_deskarga/charge_policies.pkl'                                  
        data_complete ='./data/processed/NASA/NASA_deskarga/RulEstimation_CompleteData.pkl' 
    elif dataset == "Mendeley":
        data_file = './data/processed/MENDELEY/DataSetRulEstimation_all_new.pkl'
        charge_policies = './data/processed/MENDELEY/charge_policies.csv' 
        data_complete ='./data/processed/MENDELEY/RulEstimation_CompleteData.csv'
    elif dataset == 'Zenodo':
        data_file = './data/processed/Zenodo/DataSetRulEstimation_all_new.pkl'
        charge_policies = './data/processed/Zenodo/charge_policies.pkl' 
        data_complete ='./data/processed/Zenodo/RulEstimation_CompleteData.csv'
    
    if os.path.exists(data_complete):
        df_battery = pd.read_pickle(data_complete)
        df_battery = df_battery.loc[:, ~df_battery.columns.str.contains("Unnamed")]
    else:  
        if os.path.exists(data_file):
            df_battery = pd.read_pickle(data_file)
            df_battery = df_battery.loc[:, ~df_battery.columns.str.contains("Unnamed")]
            if os.path.exists(charge_policies):
                df_battery = load_intesities(dataset)
            else:
                intensity(dataset)
                df_battery = load_intesities(dataset)
        else:
            if dataset == "NASA_karga":
                df_battery = knee_calculator_karga(2,3.6)
            elif dataset == "NASA_deskarga":
                df_battery = knee_calculator(2,3.6)
            # elif dataset == "Mendeley":
            #     df_battery = knee_calculator_m()
            # elif dataset == "Zenodo":
            #     df_battery = knee_calculator_p(2.5,4.2)
            
            if os.path.exists(charge_policies):
                df_battery = load_intesities(dataset)
            else:
                intensity(dataset)
                df_battery = load_intesities(dataset)

    
    df_summary = df_battery.groupby('bat_name').agg(
                        total_cycles=('cycle', 'max'),
                        charge_policy=('charge_policy', 'first'),  
                        one_step=('one_step', 'first'),  
                        C1=('C1', 'first'),
                        C2=('C2', 'first')
                    ).reset_index()

    if dataset == 'NASA_karga':
        df_cycle_100 = df_battery[df_battery.cycle == 100][['bat_name', 'max_Qc']]
        df_cycle_10 = df_battery[df_battery.cycle == 10][['bat_name', 'max_Qc']]
        df_summary = df_summary.merge(df_cycle_100, on='bat_name', how='left', suffixes=('', '_100'))
        df_summary = df_summary.merge(df_cycle_10, on='bat_name', how='left', suffixes=('', '_10'))
        df_summary['Qc_diff'] = df_summary['max_Qc'] - df_summary['max_Qc_10']
        df_summary.drop(columns=['max_Qc', 'max_Qc_10'], inplace=True)
    elif dataset == "NASA_deskarga": 
        df_cycle_100 = df_battery[df_battery.cycle == 100][['bat_name', 'max_Qd']]
        df_cycle_10 = df_battery[df_battery.cycle == 10][['bat_name', 'max_Qd']]
        df_summary = df_summary.merge(df_cycle_100, on='bat_name', how='left', suffixes=('', '_100'))
        df_summary = df_summary.merge(df_cycle_10, on='bat_name', how='left', suffixes=('', '_10'))
        df_summary['Qd_diff'] = df_summary['max_Qd'] - df_summary['max_Qd_10']
        df_summary.drop(columns=['max_Qd', 'max_Qd_10'], inplace=True)
    return df_battery, df_summary

###################################
#      AURRESATEAK
###################################

def converter_():
    converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y,
    'sin': lambda x    : np.sin(x),
    'cos': lambda x    : np.cos(x),
    'inv': lambda x: 1/x,
    'sqrt': lambda x: x**0.5,
    'kubo': lambda x: x**3,
    'karratu': lambda x: x**3
    }
    return converter

def represent_equation(equation, converter):
    # Replace symbolic operations with their corresponding Python functions
    for key, func in converter.items():
        equation = equation.replace(key, f"converter['{key}']")
    return equation

def karratu(x):
    return x**2

def kubo(x):
    return x**3

class FNN(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, hidden_layers=4, hidden_neurons=[64, 32, 16, 8], activation=nn.ReLU):
        super(FNN, self).__init__()
        
        # Ziurtatu ezkutuko neuronak geruza kopuru zuzena duela
        if len(hidden_neurons) != hidden_layers:
            raise ValueError("Length of hidden_neurons must match hidden_layers")

        # Geruzak definitu
        layers = [nn.Linear(input_dim, hidden_neurons[0]), activation()]
        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons[i], hidden_neurons[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_neurons[-1], output_dim))  # Output layer
        
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        if isinstance(t, np.ndarray):
            t = torch.tensor(t, dtype=torch.float32)
        if isinstance(t, torch.Tensor) and t.dtype != torch.float32:
            t = t.float()      
        return self.net(t)

def set_seed(seed_value):   
    SEED = int(seed_value)

    # Pythoneko core moduluak
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # NumPy
    np.random.seed(SEED)

    # PyTorch
    torch.manual_seed(SEED)

    # GPU erabiltzen bada (oso garrantzitsua da)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        
  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def mape(actual, forecast):
    # 1. Benetako eta aurreikusitako balioak numpy array bihurtu
    a = np.array(actual)
    f = np.array(forecast)

    # 2. Zatidura kalkulatu puntu bakoitzean (Errore Erlatiboa):
   
    # Zatiketa bektoriala:
    percentage_error = abs(a - f) / a
    
    # 3. Errore Portzentaiaren Batez bestekoa kalkulatu
    mean_percentage_error = np.mean(percentage_error)
    
    # 4. Ehunekotan itzuli (x 100)
    mpe_result = mean_percentage_error * 100
    
    return mpe_result

def cycle_prediction(X,Y, dataset, input_, to_predict, n_features):
    #Datu-basea entrenamentu- eta testatze-multzoetan bereizi
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    ################
    # RANDOM FOREST
    ################
    # 4. Ranfom Forest eredua sortu eta entrenatu
    print("-> Random Forest: Grid Search hasten...")
    rf_oinarria = RandomForestRegressor(random_state=42)
    #Sare-bilaketa egin (hiperparametroen optimizazioa)
    param_grid_rf = {'n_estimators': [50, 100, 200],'max_depth': [10, 20, None],'min_samples_leaf': [1, 5]}
    grid_search_rf = GridSearchCV(estimator=rf_oinarria, param_grid=param_grid_rf, scoring='neg_mean_squared_error',cv=3,n_jobs=-1)
    '''
    Grid search ez bagenu egingo
    rf = RandomForestRegressor(n_estimators=len(X_train.columns), random_state=42)
    rf.fit(X_train, Y_train) 
    Y_pred_rf = rf.predict(X_test)
    '''
    grid_search_rf.fit(X_train, Y_train)
    # Aurkitu diren parametro onenekin eredu onena hartu
    best_rf_model = grid_search_rf.best_estimator_
    # Erabili eredu onena iragarpenak egiteko
    Y_pred_rf = best_rf_model.predict(X_test)
    # Errorea kalkulatu (RMSE eta MAPE)
    rmse_rf = root_mean_squared_error(Y_test, Y_pred_rf, multioutput='raw_values')
    mape_rf = mape(Y_test, Y_pred_rf)

    ###########
    # XGBoost
    ###########
    '''   
    GRID SEARCH EGIN GABE:
    XGB = XGBRegressor()
    XGB.fit(X_train, Y_train)
    Y_pred_XGB = XGB.predict(X_test)
    '''
    #Sare-bilaketa egin
    print("-> XGBoost: Grid Search hasten...")
    xgb_oinarri = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
    param_grid_xgb = {'n_estimators': [100, 200],'learning_rate': [0.01, 0.1],'max_depth': [3, 5, 7]}
    grid_search_xgb = GridSearchCV(estimator=xgb_oinarri,param_grid=param_grid_xgb,scoring='neg_mean_squared_error',cv=3,n_jobs=-1)
    grid_search_xgb.fit(X_train, Y_train)

    # Aurkitu diren parametro onenekin eredu onena hartu
    best_xgb_model = grid_search_xgb.best_estimator_
    # Erabili eredu onena iragarpenak egiteko
    Y_pred_XGB = best_xgb_model.predict(X_test)
    #Errorea kalkulatu (RMSE eta MAPE)
    rmse_XGB = root_mean_squared_error(Y_test, Y_pred_XGB, multioutput='raw_values')
    mape_XGB = mape(Y_test, Y_pred_XGB)
    
    # FNN
    # DataFrame-ak PyTorch Tensors bilakatu
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)  
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).unsqueeze(1)
    model = FNN(input_dim=len(X_train.columns), output_dim=1, hidden_layers=4, hidden_neurons=[20,20,20,10], activation=nn.ReLU)
    # Galera-funtzioa definitu 
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
 
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    Y_pred_NN = model(X_test_tensor)
    Y_pred_NN_np = Y_pred_NN.detach().flatten()
    rmse_NN = root_mean_squared_error(Y_test_tensor, Y_pred_NN_np, multioutput='raw_values')
    mape_NN = mape(Y_test, Y_pred_NN_np)
    
    # Emaitzak erakusti:
    # RMSE eta MAPE balioak:
    print(f"Random Forest RMSE: {rmse_rf[0]:.4f} eta MAPE: {mape_rf}")
    print(f"XGBoost RMSE: {rmse_XGB[0]:.4f} eta MAPE: {mape_XGB}")
    print(f"NeuralNetwork RMSE: {rmse_NN[0]:.4f} eta MAPE: {mape_NN}")


    df_plot = pd.DataFrame({'Y_test': Y_test, 'Y_pred_rf': Y_pred_rf, 'Y_pred_XGB': Y_pred_XGB, 'Y_pred_NN': Y_pred_NN.detach().cpu().numpy().flatten()})
    
    '''if n_features == 100:
        if dataset == 'NASA_karga':
            if to_predict == 'cycle_life':
                df_plot.to_pickle('./data/processed/NASA/NASA_karga/karga_cl_prediction.pkl')
            elif to_predict == 'knee_point':
                df_plot.to_pickle('./data/processed/NASA/NASA_karga/karga_kp_prediction.pkl')
        elif dataset == 'NASA_deskarga':
            if to_predict == 'cycle_life':
                df_plot.to_pickle('./data/processed/NASA/NASA_deskarga/deskarga_cl_prediction.pkl')
            elif to_predict == 'knee_point':
                df_plot.to_pickle('./data/processed/NASA/NASA_deskarga/deskarga_kp_prediction.pkl')
'''
    
    plt.figure(figsize=(8,6))

    # Eredu bakoitzerako grafiko bat:
    sns.scatterplot(x='Y_test', y='Y_pred_rf', data=df_plot, color='yellow', label="Random Forest")
    sns.scatterplot(x='Y_test', y='Y_pred_XGB', data=df_plot, color='blue', label="XGBoost")
    sns.scatterplot(x='Y_test', y='Y_pred_NN', data=df_plot, color='olive', label="Sare Neuronala")

    # Egokitzapen/doikuntza perfektua
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle="--", color="black", label="Egokitzapen perfektua")
    
    # Ardatzak eta izenburua
    plt.xlabel(f'Benetako balioak ({to_predict})')
    plt.ylabel("Aurreikusitako balioak")
    plt.title("Ereduen aurreikuspenen konparaketa")
    plt.legend()
    #plt.show()
    plt.savefig(f'./results/cycle_prediction/grafikoa_{dataset}_{input_},{to_predict},{n_features}.png')

    return


def obtain_pivot_df(df_battery, df_filtered, bats, feature, n_features, to_predict, charge_policies):
    # Ziklo bakoitza zutabe bat izan dadila:
    df_pivot = df_filtered.pivot(index='bat_name', columns='cycle', values=feature).reset_index()

    # Zutabeen izena aldatu, argiago izateko
    df_pivot.columns = ['bat_name'] + [f'cycle_{c}' for c in range(1, n_features+1)] 

    if to_predict  == 'cycle_life':
        df_output = df_battery.groupby('bat_name')['cycle'].max().reset_index()
        df_output.columns = ['bat_name', 'total_cycles']
        df_output = df_output[df_output.bat_name.isin(bats)]
    elif to_predict == 'knee_point':
        df_output = knee(df_battery, feature)
        df_output = df_output[df_output.bat_name.isin(bats)]

    if charge_policies: 
        df_one_step = df_filtered[['bat_name', 'C1', 'C2']].drop_duplicates()
        df_pivot = df_pivot.merge(df_one_step, on='bat_name', how='left')
    df_pivot = df_pivot.merge(df_output, on='bat_name', how='left')
    return df_pivot, df_output.columns[1]

def filter_batteries(df_summary, df_battery, features, grouped):
    if grouped == 'two_steps':
        bats = df_summary[df_summary.one_step == 2]['bat_name']                                                           
    elif grouped == 'fast_charging':
        bats = df_summary[df_summary.charge_policy.isin(["Fast Charge", "Faster Fast Charge"])]['bat_name']              
    else:
        bats = df_summary['bat_name']                                                                                         

    df_filtered = df_battery[df_battery['cycle'] <= features]
    df_filtered = df_filtered[df_filtered.bat_name.isin(bats)]
    
    return df_filtered, bats

def prediction(dataset, df_summary, df_battery, input_, to_predict, n_features, group, charge_policies):
    df_filtered, bats = filter_batteries(df_summary, df_battery, features=n_features, grouped=group)
    if input_ == 'knee-charge' or input_=='knee-discharge': # knee_knees - capacity_knee - knee-discharge
        df_pivot, output_col = obtain_pivot_df(df_battery, df_filtered, bats, 'knee', n_features, to_predict, charge_policies)
    elif input_ == 'cut-of-voltage':
        df_pivot, output_col = obtain_pivot_df(df_battery, df_filtered, bats, 'Q_value', n_features, to_predict, charge_policies)
    cycle_columns = df_pivot.filter(like='cycle_')
    if charge_policies:
        additional_columns = df_pivot[['C1', 'C2']]
        cycle_columns = pd.concat([cycle_columns, additional_columns], axis=1)
    cycle_prediction(df_pivot[cycle_columns.columns], df_pivot[output_col], dataset, input_, to_predict,n_features)
    #elif to_predict == 'knee-point':
    '''if input_ == 'knee-discharge': # knee_knees - capacity_knee
        df_pivot = obtain_pivot_df(df_battery, df_filtered, bats, 'knee', n_features, charge_policies)
        cycle_columns = df_pivot.filter(like='cycle_')
        cycle_prediction(df_pivot[cycle_columns.columns], df_pivot.total_cycles)
    elif input_ == 'cut-off':
        df_pivot = obtain_pivot_df(df_battery, df_filtered, bats, 'Q_max', n_features, charge_policies)
        cycle_columns = df_pivot.filter(like='cycle_')
        cycle_prediction(df_pivot[cycle_columns.columns], df_pivot.total_cycles)'''
    return 
