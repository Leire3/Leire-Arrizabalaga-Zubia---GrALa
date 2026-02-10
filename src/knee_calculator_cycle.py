import glob, warnings, os
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

###################################
#       Beharrezko funtzio batzuk
###################################

def distancia_maxima(x,y):
        m_syn = (y[-1] - y[0]) / (x[-1] - x[0])
        y_syn = y[0] + m_syn * (x - x[0])
        # ynew1 eta y_syn arteko aldea kalkulatzen da --> y_diff_syn 
        y_diff_syn = y - y_syn
        # y_diff_syn maximoa deneko puntuaren indizea identifikatu
        knee_index_syn = [index for index in range(len(y_diff_syn)) if y_diff_syn[index] == y_diff_syn.max()]
        return knee_index_syn

def knee(df_battery_cycle, output):
    result_data = []
    grouped_batteries = df_battery_cycle.groupby('bat_name')
    for bat_name, data_group in grouped_batteries:
    
        ejeX = data_group['cycle']
        ejeY = data_group[output]
        z1 = np.polyfit(ejeX, ejeY, 20)
        f1 = np.poly1d(z1)
        xnew1 = ejeX.values
        ynew1 = f1(xnew1)
        knee_index_syn = distancia_maxima(xnew1, ynew1)
    
        # ukondoa ematen den zikloa
        knee_cycle = xnew1[knee_index_syn[0]]
        # ukondoa ematen den ahalmena
        knee_capacity = ynew1[knee_index_syn[0]]
        result_data.append({
            'bat_name': bat_name,
            'knee_cycle': knee_cycle,
            'knee_capacity': knee_capacity
        })
    df_knees = pd.DataFrame(result_data)
    return df_knees

def codo_1(df_battery_cycle, varX, varY): 
    ejeX = df_battery_cycle[varX]
    ejeY = df_battery_cycle[varY]

    # Maila 20 duen polinomio bat doitu. z1-en polinomioaren koefizienteak gorde.
    z1 = np.polyfit(ejeX, ejeY, 20)
    f1 = np.poly1d(z1)
    # Se generan nuevos valores ynew1 aplicando la función polinómica f1 a los valores de Qd (xnew1)
    xnew1 = ejeX.values
    ynew1 = f1(xnew1)
    # Lerro zuzen bat (y_syn) kalkulatzen da, ynew1 balioen hasierako eta amaierako puntuak m_syn maldarekin lotuz.
    knee_index_syn = distancia_maxima(xnew1, ynew1)
    '''plt.plot(xnew1,ynew1, 'blue', label = 'Estimación')
    plt.plot(df_battery_cycle['Qd'], df_battery_cycle['V'], 'green', label = 'Real')
    plt.axvline(x=xnew1[knee_index_syn[0]], color='red', linestyle='--', label='Knee X')
    plt.xlabel(' Capacidad de descarga ')
    plt.ylabel(' Voltaje ')
    plt.legend()
    plt.show()'''
    return xnew1[knee_index_syn[0]]

def codo_2(df_battery_cycle, varX, varY):
    
    ejeX = df_battery_cycle[varX]
    ejeY = df_battery_cycle[varY]

    z1 = np.polyfit(ejeX, ejeY, 20)
    f1 = np.poly1d(z1)

    xnew1 = ejeX.values
    ynew1 = f1(xnew1)

    knee_index_syn = distancia_maxima(xnew1, ynew1)

    # Bi balio itzultzen dira: X eta Y
    return xnew1[knee_index_syn[0]], ynew1[knee_index_syn[0]]

    return

def knee_per_cycle(df_cycle_, varX, varY):
    knee = codo_1(df_cycle_, varX, varY)
    return knee

def knee_per_cycle2(df_cycle_, varX, varY):
    knee_Q , knee_V = codo_2(df_cycle_, varX, varY)
    return knee_Q, knee_V #Bi balio itzuli


def detect_outliers(data, down, up):
    # Ukondoa inoiz ez da egongo deskarga-ahalmena 0.2 denekoa baino lehenago, ezta 1 ondoren ere.
    data = np.array(data)
    
    # Outlier-ak bilatu
    outlier_indices = [i+1 for i, x in enumerate(data) if x < down or x > up]
    
    return outlier_indices

def distancia_maxima_karga(x,y):
    """
    Knee puntua (kurbaren kodoa) kalkulatzeko metodoa, lerro zuzenarekiko distantzia maximoa bilatuz.
    Karga (kurba ahurra) eta Deskarga (kurba ganbila) kurbetarako balio du.
    """
    
    # Lerro zuzen sintetikoa  kalkulatu
    if x[-1] == x[0]:
        return [0] 
        
    m_syn = (y[-1] - y[0]) / (x[-1] - x[0])
    y_syn = y[0] + m_syn * (x - x[0])
    
    # Kurba eta lerro zuzenaren arteko diferentzia
    y_diff_syn = y - y_syn
    
    # Hautatu Knee puntua:
    # 1. Batez besteko diferentzia negatiboa bada (kurba lerroaren azpian, Karga-kurba tipikoa)
    if np.mean(y_diff_syn) < 0:
        # Knee puntua diferentzia negatibo handiena (minimoa) den lekuan dago.
        max_dist = y_diff_syn.min() 
    # 2. Batez besteko diferentzia positiboa bada (kurba lerroaren gainean, Deskarga-kurba tipikoa)
    else:
        # Knee puntua diferentzia positibo handiena (maximoa) den lekuan dago.
        max_dist = y_diff_syn.max() 
        
    # Indizea lortu
    knee_index_syn = np.where(y_diff_syn == max_dist)[0]
    
    return [knee_index_syn[0]] if knee_index_syn.size > 0 else [0] # Lehen indizea itzuli

def codo_karga(df_battery_cycle, varX, varY):
    
    ejeX = df_battery_cycle[varX].values
    ejeY = df_battery_cycle[varY].values

    z1 = np.polyfit(ejeX, ejeY, 20) 
    f1 = np.poly1d(z1)

    xnew1 = ejeX
    ynew1 = f1(xnew1)

    # Knee puntuaren indizea lortzen da 
    knee_index_syn = distancia_maxima_karga(xnew1, ynew1)

    # Bi balio itzultzen dira: X eta Y (Qc eta V)
    return xnew1[knee_index_syn[0]], ynew1[knee_index_syn[0]]

def knee_per_cycle_karga(df_cycle_, varX, varY):
    # Karga kasurako: varX = 'Qc' eta varY = 'V' izango dira
    knee_Q , knee_V = codo_karga(df_cycle_, varX, varY)
    return knee_Q, knee_V

################################
#         DESKARGAKO KASUA
###############################

def knee_calculator(cut_off_down, cut_off_up):
    clean_data = []
    allfiles = glob.glob('./data/processed/NASA/*.csv')

    for indfile in range(len(allfiles)):    # Para cada batería
        plt.clf()
        print(allfiles[indfile] + ' Fitxategia (edo bateria) prozesatzen' )

        # Datuak kargatu
        df_battery = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_battery.loc[:, ~df_battery.columns.str.contains("Unnamed")]

        # Bateriaren izena
        bat_ind = allfiles[indfile].split('\\')[1].split('_')[0]

        # Grafikorako Vidris koloreak hautatu
        cmap = plt.cm.viridis  
        colors = cmap(np.linspace(0, 1, len(df_battery.cycle.unique())))

        battery_knees = []                                         
        problematic_cycles = []                                     
        cycle_numbers = []                                         
        _Q = []                                                    
        max_Qd, min_Qd = [], []                                     
        mean_V, std_V = [], []                                      
        max_I_charge, min_I_charge = [], []                        
        max_I_discharge, min_I_discharge = [], []                   
        VQ_ratio_mean, VQ_ratio_std, VQ_ratio_median = [],[],[]          

        # Bateria bakoitzerako:
        for ind in df_battery.cycle.unique(): 

            print(f'{ind} zikloan gaude.')
            # 1) Se genera el dataset correspondiente al ciclo
            df_cycle = df_battery[df_battery['cycle'] == ind]
            df = df_cycle[df_cycle['I'] < 0]
            df = df.reset_index(drop=True)

            add_plot = True
            df['dV_dQd'] = np.gradient(df['V'], df['Qd'])

            # "Legezko" tentsio-tartera mugatu
            first_index = df[df['V'] <= cut_off_down].index[0] if (df['V'] <= cut_off_down).any() else len(df['V'])-1
            if ((df["V"] < cut_off_down).any() or (df["V"] > cut_off_up).any()):  
                # Lortu azken indizea non V >= 3.6
                last_index = df[df['V'] >= cut_off_up].index[-1] if (df['V'] >= cut_off_up).any() else 0
                df.loc[:last_index, 'V'] = cut_off_up
                # Lortu lehenengo indizea non V <= 2
                first_index = df[df['V'] <= cut_off_down].index[0] if (df['V'] <= cut_off_down).any() else len(df['V'])-1
                df.loc[first_index:,'V'] = cut_off_down

            if len(df['V']) < 20:                                                           # Mugatu ondoren, datu gutxi baditu --> ezabatu
                print('Este ciclo está mal registrado, hay que eliminarlo')
                problematic_cycles.append(ind)
                battery_knees.append(np.nan)
            elif (df.loc[df['V'] == cut_off_down, 'Qd'] <= 0.2).any():                      # Ahalmena esperotakoa baino askoz txikiagoa bada --> ezabatu
                print('Este ciclo está mal registrado, hay que eliminarlo')
                add_plot = False
                problematic_cycles.append(ind)
                battery_knees.append(np.nan)
            else:                                                                           # Bestela, knee kalkulatu
                knee_onset = knee_per_cycle(df, 'Qd', 'V')
                battery_knees.append(knee_onset)

            # Zikloaren gaineko informazioa gorde:
            cycle_numbers.append(ind)
            _Q.append(df['Qd'][first_index])
            max_Qd.append(max(df['Qd']))
            min_Qd.append(min(df['Qd']))

            # Tentsioaren ezaugarriak
            mean_V.append(df.V.mean())
            std_V.append(df.V.std())  
            # Intentsitatearen ezaugarriak                 
            max_I_discharge.append(df.I.max())
            min_I_discharge.append(df.I.min())
            charge_I=df_cycle.loc[df_cycle.I>=0,'I']
            max_I_charge.append(max(charge_I))
            min_I_charge.append(min(charge_I))  
            
            # Tentsioaren eta deskarga-ahalmenaren arteko erlazioa
            dQ = df.loc[(df.Qd > 0) & (df.Qd < 0.05), 'Qd'] 
            dV = df.loc[(df.Qd > 0) & (df.Qd < 0.05), 'V']
            ratio = dV/dQ    
            VQ_ratio_mean.append(ratio.mean())
            VQ_ratio_std.append(ratio.std())
            VQ_ratio_median.append(ratio.median())

            #Lehenengo eta azken zikloak izendatu legendan
            if ind == 1 and add_plot:
                plt.scatter(df['Qd'], df['V'], color = colors[ind-1], label = 'Ziklo 1', s=0.5)
            elif ind == df_battery.cycle.unique()[-1] and add_plot:
                plt.scatter(df['Qd'], df['V'], color = colors[ind-1], label = f'Ziklo {df_battery.cycle.unique()[-1]}',s=0.5)
            elif add_plot:
                plt.scatter(df['Qd'], df['V'], color = colors[ind-1], s=0.5)
                    
        # Ziklo zaratatsuak ezabatu
        outliers_1 = detect_outliers(battery_knees, 0.2, 1.07)  #Knee inoiz ez da egongo (0.2,1.07) tartetik kanpo

        ############################################################################
        ##################### Degradazio kurba - KNEES ##############################
        ############################################################################

        # Kurba 
        x = df_battery.cycle.unique()
        y = battery_knees

        # Doikuntza polinomioa
        degree = 15
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)

        # Balioak
        y_fit = p(x)

        # Hondarrak eta askatasun graduak
        residuals = y - y_fit
        n = len(y)
        m = len(coeffs)  # Koefiziente kopurua (maila + 1)
        dof = n - m  # Askatasun graduak 

        # Errore estandarra eta konfiantza tartea 
        residual_std_error = np.std(residuals, ddof=m)
        multiplicador = 5  
        conf_interval = multiplicador * residual_std_error

        y_upper = y_fit + conf_interval
        y_lower = y_fit - conf_interval
        outliers_2 = (y < y_lower) | (y > y_upper)

        # Konfiantza tartetik kanpo dauden puntuak outlierrak. Horien zikloak:
        outlier_points_2 = df_battery.cycle.unique()[outliers_2]
        
        ############################################################################
        ################ Degradazio kurba - Cut-off voltage #########################
        ############################################################################
        x_c = df_battery.cycle.unique()
        y_c = _Q

        # Doikuntza polinomioa
        degree = 15
        coeffs_c = np.polyfit(x_c, y_c, degree)
        p_c = np.poly1d(coeffs_c)

        # Balioak
        y_fit_c = p_c(x_c)

        # Hondarrak eta askatasun graduak
        residuals_c = y_c - y_fit_c
        n = len(y_c)
        m = len(coeffs_c)  # Number of coefficients (degree + 1)
        dof = n - m  # Degrees of freedom

        # Errore estandarra eta konfiantza tartea
        residual_std_error_c = np.std(residuals_c, ddof=m)
        multiplicador_c = 2  
        conf_interval_c = multiplicador_c * residual_std_error_c

        y_upper_c = y_fit_c + conf_interval_c
        y_lower_c = y_fit_c - conf_interval_c
        outliers_3 = (y_c < y_lower_c) | (y_c > y_upper_c)

        # Konfiantza tartetik kanpo dauden puntuak outlierrak. Horien zikloak:
        outlier_points_3 = df_battery.cycle.unique()[outliers_3]

        #Outlier guztiak
        total_outliers = np.concatenate([problematic_cycles, outliers_1, outlier_points_2, outlier_points_3])
        for out in np.unique(total_outliers):
            df_cycle = df_battery[df_battery['cycle'] == out]
            df = df_cycle[df_cycle['I'] < 0]
            plt.scatter(df['Qd'], df['V'], color = 'red', label = f'outlier ziklo {out}', s=0.5)
            
        
        plt.title('Deskarga kurba')
        plt.xlabel(r'$\text{Deskarga ahalmena (Q)}$', fontsize = 14)
        plt.ylabel(r'$\text{Tentsioa (V)}$', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/deskargako_grafikuek/deskarga_{bat_ind}.png')        

        
        plt.clf()
        plt.title('Degradazio kurba (Knee puntuak)')
        plt.scatter(df_battery.cycle.unique(), battery_knees, color='blue', s=5, label="Bateriaren knee puntuak")
        plt.scatter(total_outliers, np.array(battery_knees)[(total_outliers-1).astype(int)], color='red', s=5, label="Knee outlierrak")
        plt.plot(x, y_fit, label="Doikuntza polinomioa", color="blue")
        plt.xlabel(r'$\text{Ziklo kopurua}$', fontsize = 14)
        plt.ylabel(r'$\text{Deskarga ahalmena (Q)}$', fontsize=14)
        plt.fill_between(x, y_lower, y_upper, color="lightblue", alpha=0.5, label="Konfiantza-tartea")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/deskargako_grafikuek/deskarga_knees_{bat_ind}.png')


        plt.clf()
        plt.title('Degradazio kurba (Cut off voltage puntuak)')
        plt.scatter(df_battery.cycle.unique(), _Q, color='green', s=5, label="Bateriaren ahalmena")
        plt.scatter(total_outliers, np.array(_Q)[(total_outliers-1).astype(int)], color='red', s=5, label="Ahalmenaren outlierrak")
        plt.plot(x_c, y_fit_c, label="Doikuntza polinomioa", color="green")
        plt.fill_between(x_c, y_lower_c, y_upper_c, color="green", alpha=0.5, label="Konfiantza-tartea")
        plt.xlabel(r'$\text{Ziklo kopurua}$', fontsize = 14)
        plt.ylabel(r'$\text{Deskarga ahalmena (Q)}$', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/deskargako_grafikuek/deskarga_capacity_{bat_ind}.png')
        
        # Outlierrak ez diren zikloak
        new_cicle_number = 1
        for cycle, knee, Q_value, max_Qd_value, min_Qd_value, mean_V_, std_V_, max_I_charge_, min_I_charge_, max_I_discharge_, min_I_discharge_, VQ_ratio_mean_, VQ_ratio_std_, VQ_ratio_median_ in zip(cycle_numbers, battery_knees, _Q, max_Qd, min_Qd, mean_V, std_V, max_I_charge, min_I_charge, max_I_discharge, min_I_discharge, VQ_ratio_mean, VQ_ratio_std, VQ_ratio_median):
            if cycle not in total_outliers:
                clean_data.append([bat_ind, new_cicle_number, knee, Q_value, max_Qd_value, min_Qd_value, mean_V_, std_V_, max_I_charge_, min_I_charge_, max_I_discharge_, min_I_discharge_, VQ_ratio_mean_, VQ_ratio_std_, VQ_ratio_median_])
                new_cicle_number += 1

        
    # Amaierako DataFrame-a
    df_clean = pd.DataFrame(clean_data, columns=['bat_name', 'cycle', 'knee', 'Q_value', 'max_Qd', 'min_Qd','mean_V', 'std_V', 'max_I_charge', 'min_I_charge', 'max_I_discharge', 'min_I_discharge', 'VQ_ratio_mean', 'VQ_ratio_std', 'VQ_ratio_median'])
    path = './data/processed/NASA/NASA_deskarga/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_clean.to_pickle('./data/processed/NASA/NASA_deskarga/DataSetRulEstimation_all_new.pkl')

    return df_clean

def knee_calculator_200_knee_cof(cut_off_down, cut_off_up):
    allfiles = glob.glob('./data/processed/NASA/*.csv')
    knee_points = {}
    for indfile in range(len(allfiles)):    # Para cada batería
        plt.clf()
        print('Processing File (or battery) '+ allfiles[indfile])

        # Cargar datos
        df_battery = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_battery.loc[:, ~df_battery.columns.str.contains("Unnamed")]

        # Nombre de la batería
        bat_ind = allfiles[indfile].split('\\')[1].split('_')[0]

        # Selección del arcoíris para el plot (vidiris)
        cmap = plt.cm.viridis  
        colors = cmap(np.linspace(0, 1, len(df_battery.cycle.unique())))

        # Ziklo garrantzitsuak identifikatu
        ziklo_guztiak = df_battery.cycle.unique()
        azken_zikloa = ziklo_guztiak[-1]
        
        # Marraztu nahi ditugun zikloen multzoa
        irudikatzeko_zikloak_multzoa = set()
        if len(ziklo_guztiak) > 0:
            irudikatzeko_zikloak_multzoa.add(ziklo_guztiak[0])
            irudikatzeko_zikloak_multzoa.add(azken_zikloa)
            for zikloa in ziklo_guztiak:
                if zikloa % 200 == 0:
                    irudikatzeko_zikloak_multzoa.add(zikloa)
        
        irudikatzeko_zikloak = sorted(list(irudikatzeko_zikloak_multzoa))

        battery_knees = []                                          
        problematic_cycles = []                                     

        # Se estudia para cada 
        cutoff_label_added = False
        for ind in df_battery.cycle.unique(): 

            #print(f'Estamos en el ciclo: {ind}')
            # 1) Se genera el dataset correspondiente al ciclo
            df_cycle = df_battery[df_battery['cycle'] == ind]
            df = df_cycle[df_cycle['I'] < 0]
            df = df.reset_index(drop=True)

            add_plot = True
            df['dV_dQd'] = np.gradient(df['V'], df['Qd'])

            # Se acotan los registros al rango de voltage "legal" (criterios en función de la descarga)
            first_index = df[df['V'] <= cut_off_down].index[0] if (df['V'] <= cut_off_down).any() else len(df['V'])-1
            if ((df["V"] < cut_off_down).any() or (df["V"] > cut_off_up).any()):  
                # Obtener el último índice donde V es mayor o igual a 3.6
                last_index = df[df['V'] >= cut_off_up].index[-1] if (df['V'] >= cut_off_up).any() else 0
                df.loc[:last_index, 'V'] = cut_off_up
                # Obtener el primer índice donde V es menor o igual a 2
                first_index = df[df['V'] <= cut_off_down].index[0] if (df['V'] <= cut_off_down).any() else len(df['V'])-1
                df.loc[first_index:,'V'] = cut_off_down

            if len(df['V']) < 20:                                                           # Si una vez acortado tiene muy pocos registros se elimina
                print('Este ciclo está mal registrado, hay que eliminarlo')
                problematic_cycles.append(ind)
                battery_knees.append(10)
            elif (df.loc[df['V'] == cut_off_down, 'Qd'] <= 0.2).any():                      # Si la capacidad en el cut of voltage es muy inferior al resperado se elimina
                print('Este ciclo está mal registrado, hay que eliminarlo')
                add_plot = False
                problematic_cycles.append(ind)
                battery_knees.append(10)
            else:                                                                           # Si no, se calcula el codo
                Q_knee, V_knee = knee_per_cycle2(df, 'Qd', 'V') 
                # Knee koordenatuak lortu eta gorde
                knee_points[ind] = (Q_knee, V_knee)
                

            #Kurbak marraztu
            if ind == 1 and add_plot:
                plt.scatter(df['Qd'], df['V'], color = colors[ind-1], label = 'Ziklo 1', s=0.5)
            elif ind == df_battery.cycle.unique()[-1] and add_plot:
                plt.scatter(df['Qd'], df['V'], color = colors[ind-1], label = f'Ziklo {df_battery.cycle.unique()[-1]}',s=0.5)
            elif add_plot:
                plt.scatter(df['Qd'], df['V'], color = colors[ind-1], s=0.5)
            
            if ind in irudikatzeko_zikloak and add_plot: # Marraztu nahi ditugun zikloetarako bakarrik
                ##############################################
                # Cut-off puntuak
                ############################################

                # Cut-off puntuak kalkulatu
                Q_cutoff = df['Qd'][first_index] if len(df['Qd']) > 0 else 0
                V_cutoff = cut_off_down
                
                # Tamaina kudeatu
                point_size = 50 if ind == 1 or ind == azken_zikloa else 30
                
                current_label = 'Cut-off voltage' if not cutoff_label_added else ""
                plt.scatter(Q_cutoff, V_cutoff, color = 'black', marker='o', s=point_size, zorder=5, label=current_label) # Puntu beltz zirkularra
                cutoff_label_added = True
                # Etiketak
                if ind == 1:
                    plt.annotate('Ziklo 1', (Q_cutoff, V_cutoff), textcoords="offset points", xytext=(5,-10), ha='left', fontsize=8, color='black')
                elif ind == azken_zikloa:
                    plt.annotate(f'Ziklo {ind}', (Q_cutoff, V_cutoff), textcoords="offset points", xytext=(-5,-10), ha='right', fontsize=8, color='black')

        # ------------------------------------------------------------------
        # Knee puntuak Marraztu (Ziklo garrantzitsuetarako bakarrik)
        # --------------------------------------------------
        knee_point_label_added = False 

        for cycle in irudikatzeko_zikloak:
            if cycle in knee_points:
                Q_knee, V_knee = knee_points[cycle]
                
                # Knee puntuak beltz eta karratu bezala marraztuko ditugu (marker='s')
                knee_size = 30 if cycle == 1 or cycle == azken_zikloa else 20
                
                #Etiketa behin bakarrik
                current_knee_label = 'Knee' if not knee_point_label_added else ""

                plt.scatter(Q_knee, V_knee, 
                            color='black', 
                            marker='^', # Triangelua, Cut-off puntuetatik bereizteko
                            s=knee_size, 
                            zorder=10,
                            label= current_knee_label)
                
                knee_point_label_added = True

            if cycle == ziklo_guztiak[0]:
                    plt.annotate('Ziklo 1', (Q_knee, V_knee), 
                                 textcoords="offset points", xytext=(20, 5), ha='center', 
                                 fontsize=8, color='black')
            elif cycle == azken_zikloa:
                plt.annotate(f'Ziklo {cycle}', (Q_knee, V_knee), 
                                textcoords="offset points", xytext=(-20, 5), ha='center', 
                                fontsize=8, color='black')
        # ------------------------------------------------------------------
        # Grafikoaren Estetika eta Gorde
        # ------------------------------------------------------------------
        
        plt.axhline(y=cut_off_down, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.title('Deskarga kurba')
        plt.xlabel(r'$\text{Deskarga ahalmena (Q)}$', fontsize = 14)
        plt.ylabel(r'$\text{Tentsioa (V)}$', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/deskargako_grafikuek/cov_knee/deskarga_cov_knee_{bat_ind}.png')        

    return 

def capacity_kneepoint_calculator_deskarga(cut_off_down, cut_off_up): 
    # 1. Kargatu knee-point orokorrak (DataSetRulEstimation-etik)
    deskargako_datuak = pd.read_pickle('./data/processed/NASA/NASA_deskarga/DataSetRulEstimation_all_new.pkl')
    deskarga_q_value = deskargako_datuak[['bat_name', 'cycle', 'Q_value']].copy()
    
    # Knee-point orokorrak kalkulatu (degradazio kurbaren inflexio puntua)
    knees_deskarga = knee(deskarga_q_value, output='Q_value')
    knees_deskarga = knees_deskarga.rename(columns={
        'knee_cycle': 'knee_cycle_deskarga',
        'knee_capacity': 'knee_capacity_deskarga'
    })

    allfiles = glob.glob('./data/processed/NASA/*.csv')

    for indfile in range(len(allfiles)): 
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bateriaren datuak kargatu
        df_temp = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_temp.loc[:, ~df_temp.columns.str.contains("Unnamed")]

        # Bateriaren izena lortu (fitxategi izenetik)
        bat_ind = os.path.basename(allfiles[indfile]).split('_')[0]
        print(f'Processing File (or battery): {bat_ind}')

        # Bateria honetarako knee-point orokorra lortu
        knee_info_bat = knees_deskarga[knees_deskarga['bat_name'] == bat_ind]
        if knee_info_bat.empty:
            print(f"Ez da knee point-ik aurkitu {bat_ind} bateriarako. Saltatu.")
            plt.close(fig)
            continue
            
        knee_cycle_num = knee_info_bat['knee_cycle_deskarga'].iloc[0]

        # Zerrendak hasieratu
        battery_knees = [] 
        problematic_cycles = [] 
        cycle_numbers = [] 
        _Q = [] 
        
        # Zikloz ziklo prozesatu
        for ind in df_battery.cycle.unique(): 
            df_cycle = df_battery[df_battery['cycle'] == ind]
            df = df_cycle[df_cycle['I'] < 0].reset_index(drop=True)

            if len(df) < 2: continue # Datu nahikorik ez badago

            # Ziklo barruko knee kalkulua (Knee vs Knee-point bereizketa)
            is_problematic = False
            if len(df['V']) < 20 or (df.loc[df['V'] >= cut_off_up, 'Qd'] <= 0.2).any():
                is_problematic = True
            
            cycle_numbers.append(ind)
            if is_problematic:
                problematic_cycles.append(ind)
                battery_knees.append(np.nan)
                _Q.append(np.nan) 
            else:
                # Ziklo barruko knee-a (Bego-ri galdetu diozun hori)
                knee_onset = knee_per_cycle(df, 'Qd', 'V')
                battery_knees.append(knee_onset)
                _Q.append(df['Qd'].max()) 

        # --- OUTLIERREN TRATAMENDUA ---
        outliers_1 = detect_outliers(battery_knees, 0.05, 1.07)
        
        y_c_raw = np.array(_Q)
        x_c_raw = np.array(cycle_numbers)
        valid_mask = ~np.isnan(y_c_raw)

        x_c = x_c_raw[valid_mask]
        y_c = y_c_raw[valid_mask]

        # Polinomio bidezko doitzea (outlier estatistikoak detektatzeko)
        degree = 15
        coeffs_c = np.polyfit(x_c, y_c, degree)
        p_c = np.poly1d(coeffs_c)
        y_fit_c = p_c(x_c)

        # Konfiantza tartea eta maskara (Zuzenduta: tamaina match egiteko)
        residuals_c = y_c - y_fit_c
        residual_std_error_c = np.std(residuals_c, ddof=len(coeffs_c))
        conf_interval_c = 2 * residual_std_error_c
        y_upper_c = y_fit_c + conf_interval_c
        y_lower_c = y_fit_c - conf_interval_c
        
        outliers_3_mask = (y_c < y_lower_c) | (y_c > y_upper_c)
        outlier_points_3 = x_c[outliers_3_mask] # Tamaina bereko arrayak erabiliz

        total_outliers = np.unique(np.concatenate([problematic_cycles, outliers_1, outlier_points_3]))

        # --- GRAFIKOA ---
        ax.scatter(x_c, y_c, color='green', s=5, label="Bateriaren ahalmena (Q)")
        
        # Outlier-ak margotu
        is_outlier_plot = np.isin(x_c, total_outliers)
        ax.scatter(x_c[is_outlier_plot], y_c[is_outlier_plot], color='red', s=5, label="Outliers")
        
        ax.plot(x_c, y_fit_c, color="darkgreen", label="Doitutako kurba")
        ax.fill_between(x_c, y_lower_c, y_upper_c, color="green", alpha=0.2, label="Konfiantza tartea")
        
        # KNEE-POINT MARRA BERTIKALA (Hau da eskatu duzuna)
        ax.axvline(x=knee_cycle_num, color='black', linestyle='--', linewidth=2, 
                   label=f'Knee-point: {knee_cycle_num} zikloa')

        ax.set_xlabel('Ziklo zenbakia', fontsize=12)
        ax.set_ylabel('Deskarga ahalmena (Q)', fontsize=12)
        ax.set_title('Degradazio kurba (Cut-off voltage puntuak)')
        ax.legend()
        
        # Gorde
        path = f'./results/deskarga_capacity_kneepoint/deskarga_capacity_kneepoint_{bat_ind}.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close(fig)

    return 

def knee_kneepoint_calculator_deskarga(cut_off_down, cut_off_up): 
    # 1. Kargatu knee-point orokorrak (DataSetRulEstimation-etik)
    deskargako_datuak = pd.read_pickle('./data/processed/NASA/NASA_deskarga/DataSetRulEstimation_all_new.pkl')
    deskarga_knee = deskargako_datuak[['bat_name', 'cycle', 'knee']].copy()
    
    # Knee-point orokorrak kalkulatu (degradazio kurbaren inflexio puntua)
    knees_deskarga = knee(deskarga_knee, output='knee')
    knees_deskarga = knees_deskarga.rename(columns={
        'knee_cycle': 'knee_cycle_deskarga',
        'knee_knee': 'knee_knee_deskarga'
    })

    allfiles = glob.glob('./data/processed/NASA/*.csv')

    for indfile in range(len(allfiles)): 
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bateriaren datuak kargatu
        df_temp = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_temp.loc[:, ~df_temp.columns.str.contains("Unnamed")]

        # Bateriaren izena lortu (fitxategi izenetik)
        bat_ind = os.path.basename(allfiles[indfile]).split('_')[0]
        print(f'Processing File (or battery): {bat_ind}')

        # Bateria honetarako knee-point orokorra lortu
        knee_info_bat = knees_deskarga[knees_deskarga['bat_name'] == bat_ind]
        if knee_info_bat.empty:
            print(f"Ez da knee point-ik aurkitu {bat_ind} bateriarako. Saltatu.")
            plt.close(fig)
            continue
            
        knee_cycle_num = knee_info_bat['knee_cycle_deskarga'].iloc[0]

        # Zerrendak hasieratu
        battery_knees = [] 
        problematic_cycles = [] 
        cycle_numbers = [] 
        _Q = [] 
        
        # Zikloz ziklo prozesatu
        for ind in df_battery.cycle.unique(): 
            df_cycle = df_battery[df_battery['cycle'] == ind]
            df = df_cycle[df_cycle['I'] < 0].reset_index(drop=True)

            if len(df) < 2: continue # Datu nahikorik ez badago

            # Ziklo barruko knee kalkulua (Knee vs Knee-point bereizketa)
            is_problematic = False
            if len(df['V']) < 20 or (df.loc[df['V'] >= cut_off_up, 'Qd'] <= 0.2).any():
                is_problematic = True
            
            cycle_numbers.append(ind)
            if is_problematic:
                problematic_cycles.append(ind)
                battery_knees.append(np.nan)
                _Q.append(np.nan) 
            else:
                # Ziklo barruko knee-a (Bego-ri galdetu diozun hori)
                knee_onset = knee_per_cycle(df, 'Qd', 'V')
                battery_knees.append(knee_onset)
                _Q.append(df['Qd'].max()) 

        # --- OUTLIERRAK ---
        outliers_1 = detect_outliers(battery_knees, 0.05, 1.07)
        
        y_c_raw = np.array(_Q)
        x_c_raw = np.array(cycle_numbers)
        valid_mask = ~np.isnan(y_c_raw)

        x_c = x_c_raw[valid_mask]
        y_c = y_c_raw[valid_mask]

        # Polinomio bidezko doitzea (outlier estatistikoak detektatzeko)
        degree = 15
        coeffs_c = np.polyfit(x_c, y_c, degree)
        p_c = np.poly1d(coeffs_c)
        y_fit_c = p_c(x_c)

        # Konfiantza tartea eta maskara 
        residuals_c = y_c - y_fit_c
        residual_std_error_c = np.std(residuals_c, ddof=len(coeffs_c))
        conf_interval_c = 5 * residual_std_error_c
        y_upper_c = y_fit_c + conf_interval_c
        y_lower_c = y_fit_c - conf_interval_c
        
        outliers_3_mask = (y_c < y_lower_c) | (y_c > y_upper_c)
        outlier_points_3 = x_c[outliers_3_mask] # Tamaina bereko arrayak erabiliz

        total_outliers = np.unique(np.concatenate([problematic_cycles, outliers_1, outlier_points_3]))

        # --- GRAFIKOA ---
        ax.scatter(x_c, y_c, color='blue', s=5, label="Bateriaren knee puntua (Q)")
        
        # Outlier-ak margotu
        is_outlier_plot = np.isin(x_c, total_outliers)
        ax.scatter(x_c[is_outlier_plot], y_c[is_outlier_plot], color='red', s=5, label="Outliers")
        
        ax.plot(x_c, y_fit_c, color="darkblue", label="Doitutako kurba")
        ax.fill_between(x_c, y_lower_c, y_upper_c, color="blue", alpha=0.2, label="Konfiantza tartea")
        
        # KNEE-POINT MARRA BERTIKALA (Hau da eskatu duzuna)
        ax.axvline(x=knee_cycle_num, color='black', linestyle='--', linewidth=2, 
                   label=f'Knee-point: {knee_cycle_num} zikloa')

        ax.set_xlabel('Ziklo zenbakia', fontsize=12)
        ax.set_ylabel('Deskarga ahalmena (Q)', fontsize=12)
        ax.set_title('Degradazio kurba (Knee puntuak)')
        ax.legend()
        
        # Gorde
        path = f'./results/deskarga_knee_kneepoint/deskarga_knee_kneepoint_{bat_ind}.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close(fig)

    return 

##############################
#     KARGAKO KASURAKO
##############################

def knee_calculator_karga(cut_off_down, cut_off_up):
    clean_data = []
    allfiles = glob.glob('./data/raw/NASA/*.csv')

    for indfile in range(len(allfiles)):    # Bateria bakoitzerako
        plt.clf()
        print('Processing File (or battery) '+ allfiles[indfile])
        # Cargar datos
        df_temp = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_temp.loc[:, ~df_temp.columns.str.contains("Unnamed")]

        # Nombre de la batería
        bat_ind = allfiles[indfile].split('\\')[1].split('_')[0]

        # Selección del arcoíris para el plot (vidiris)
        cmap = plt.cm.viridis  
        colors = cmap(np.linspace(0, 1, len(df_battery.cycle.unique())))
        
        battery_knees = []                                         
        problematic_cycles = []                                    
        cycle_numbers = []                                         
        _Q = []                                                   
        max_Qc, min_Qc = [], []                                     
        mean_V, std_V = [], []                                      
        max_I_charge, min_I_charge = [], []                         
        max_I_discharge, min_I_discharge = [], []                   
        
        # Bateria bakoitzerako:
        for ind in df_battery.cycle.unique(): 

            # karga (I > 0)
            df_cycle = df_battery[df_battery['cycle'] == ind]
            df = df_cycle[df_cycle['I'] > 0]
            df = df.reset_index(drop=True)
            
            add_plot = True
            
            df['dV_dQc'] = np.gradient(df['V'], df['Qc'])

            # Tentsioa bornatu:
            # 1. Beheko muga hautatu (V >= cut_off_down)
        
            if (df['V'] > cut_off_down).any():
                first_index = df[df['V'] >= cut_off_down].index[0] 
                df = df.loc[first_index:].reset_index(drop=True)
            else:
                add_plot = False
                
            # 2. Goiko muga moztu (V <= cut_off_up)
            # Karga prozesuak V=cut_off_up inguruan amaitu behar du. Ziurtatu V ez dela cut_off_up baino handiagoa, hori errore bat izan liteke (edo karga oso luzea).
            if (df['V'] > cut_off_up).any():
                last_index = df[df['V'] <= cut_off_up].index[-1] if (df['V'] <= cut_off_up).any() else len(df)
                df = df.loc[:last_index].reset_index(drop=True)

            if (df['Qc'] > 1.1).any():
                last_index_Qc = df[df['Qc']>1.1].index[0] #Qc hori 1.1 balioa gainditzen duen lehen indizea aurkitu eta ondorengo datu guztiak moztu
                df = df.loc[:last_index_Qc - 1].reset_index(drop=True) #Karga-ahalmena 1.1 baino handiagoa den balioa hartu baino lehenagora arte moztu

            if len(df['V']) < 20:                                                           # Behin bornatutakoan datu gutxi baditu --> ezabatu
                print('Este ciclo está mal registrado, hay que eliminarlo')
                problematic_cycles.append(ind)
                battery_knees.append(np.nan)
            elif (df.loc[df['V'] == cut_off_up, 'Qc'] <= 0.2).any():                      # V-ren balioa cut_off_up daniean, Qc < 0.2 baldin bada, anomalia
                print('Este ciclo está mal registrado, hay que eliminarlo')
                add_plot = False
                problematic_cycles.append(ind)
                battery_knees.append(np.nan)
            
            else:                                                                           # Bestela, knee kalkulatu
                knee_onset = knee_per_cycle(df, 'Qc', 'V')
                battery_knees.append(knee_onset)
            
            # Zikloaren informazioa gorde
            cycle_numbers.append(ind)
            _Q.append(df['Qc'].max())   #Kasu honetan, zutabe horretako balio maximoa
            max_Qc.append(max(df['Qc'])) 
            min_Qc.append(min(df['Qc']))

            # Tentsioaren ezaugarriak
            mean_V.append(df.V.mean())
            std_V.append(df.V.std())  

            # Intentsitatearen ezaugarriak:                  
            max_I_discharge.append(df.I.max())
            min_I_discharge.append(df.I.min())
            charge_I=df_cycle.loc[df_cycle.I>=0,'I']
            max_I_charge.append(max(charge_I))
            min_I_charge.append(min(charge_I))  
            
            #Kurbak irudikatu + lehenengo eta azkenengo zikloak legendan
            if ind == 1 and add_plot:
                plt.scatter(df['Qc'], df['V'], color = colors[ind-1], label = 'Ziklo 1', s=0.5)
            elif ind == df_battery.cycle.unique()[-1] and add_plot:
                plt.scatter(df['Qc'], df['V'], color = colors[ind-1], label = f'Ziklo {df_battery.cycle.unique()[-1]}',s=0.5)
            elif add_plot:
                plt.scatter(df['Qc'], df['V'], color = colors[ind-1], s=0.5)
            
        # Ziklo zaratatsuak ezabatu
        outliers_1 = detect_outliers(battery_knees, 0.05, 1.07)  

        #########################################################
        ############ Degradazio kurba - KNEES ###################
        #########################################################
        # Kurba 
        x = df_battery.cycle.unique()
        y = battery_knees

        # Doitutako polinomioa
        degree = 5 #Kasu honetan, maila txikiagoa izatea behar dugu.
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)

        # Balioak
        y_fit = p(x)

        # Hondarrak eta askatasun graduak
        residuals = y - y_fit
        n = len(y)
        m = len(coeffs)  
        dof = n - m  

        # Errore estandarra eta konfiantzazko tartea
        residual_std_error = np.std(residuals, ddof=m)
        multiplicador = 2.5  
        conf_interval = multiplicador * residual_std_error

        y_upper = y_fit + conf_interval
        y_lower = y_fit - conf_interval
        outliers_2 = (y < y_lower) | (y > y_upper)

        # Outlierrak izango dira konfiantza tartetik kanpo daudenak. Horien zikloak:
        outlier_points_2 = df_battery.cycle.unique()[outliers_2]
        
        #########################################################
        ############ Degradazio kurba - CUT-OFF VOLTAGE ###################
        #########################################################
        x_c = df_battery.cycle.unique()
        y_c = _Q

        # Doitutako polinomioa
        degree = 15
        coeffs_c = np.polyfit(x_c, y_c, degree)
        p_c = np.poly1d(coeffs_c)

        # Balioak
        y_fit_c = p_c(x_c)

        # Hondarrak eta askatasun graduak
        residuals_c = y_c - y_fit_c
        n = len(y_c)
        m = len(coeffs_c) 
        dof = n - m  # 

        # Errore estandarra eta konfiantza tartea
        residual_std_error_c = np.std(residuals_c, ddof=m)
        multiplicador_c = 2  
        conf_interval_c = multiplicador_c * residual_std_error_c

        y_upper_c = y_fit_c + conf_interval_c
        y_lower_c = y_fit_c - conf_interval_c
        outliers_3 = (y_c < y_lower_c) | (y_c > y_upper_c)

        # Konfiantza tartetik daudenak outlierrak. Outlierren zikloak:
        outlier_points_3 = df_battery.cycle.unique()[outliers_3]

        total_outliers = np.concatenate([problematic_cycles, outliers_1, outlier_points_2, outlier_points_3])
        for out in np.unique(total_outliers):
            df_cycle = df_battery[df_battery['cycle'] == out]
            df = df_cycle[df_cycle['I'] > 0]
            plt.scatter(df['Qc'], df['V'], color = 'red', label = f'outlier ziklo {out}', s=0.5)
        
        #GRAFIKOAK
        plt.xlim(0.0, 1.1)

        #Karga kurba
        plt.xlabel(r'$\text{Karga ahalmena (Q)}$', fontsize = 14)
        plt.ylabel(r'$\text{Tentsioa (V)}$', fontsize=14)
        plt.title('Karga kurba')
        plt.legend()
        plt.tight_layout()
        output_dir = './results/kargako_grafikuek/'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/karga_{bat_ind}.png')
        
        #Degradazio kurba (knee)
        plt.clf()
        plt.title('Degradazio kurba (Knee puntuak)')
        plt.xlabel(r'$\text{Ziklo kopurua}$', fontsize = 14)
        plt.ylabel(r'$\text{Karga ahalmena (Q)}$', fontsize=14)
        plt.scatter(df_battery.cycle.unique(), battery_knees, color='blue', s=5, label="Bateriaren knee puntuak")
        plt.scatter(total_outliers, np.array(battery_knees)[(total_outliers-1).astype(int)], color='red', s=5, label="Knees outlierrak")
        plt.plot(x, y_fit, label="Doikuntza polinomioa", color="blue")
        plt.fill_between(x, y_lower, y_upper, color="lightblue", alpha=0.5, label="Konfiantza-tartea")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/kargako_grafikuek/karga_knees_{bat_ind}.png')

        #Degradazio kurba (cut-off voltage)
        plt.clf()
        plt.title('Degradazio kurba (Cut off voltage puntuak)')
        plt.xlabel(r'$\text{Ziklo kopurua}$', fontsize = 14)
        plt.ylabel(r'$\text{Karga ahalmena (Q)}$', fontsize=14)
        plt.scatter(df_battery.cycle.unique(), _Q, color='green', s=5, label="Bateriaren ahalmena")
        plt.scatter(total_outliers, np.array(_Q)[(total_outliers-1).astype(int)], color='red', s=5, label="Ahalmenaren outlierrak")
        plt.plot(x_c, y_fit_c, label="Doikuntza polinomioa", color="green")
        plt.fill_between(x_c, y_lower_c, y_upper_c, color="green", alpha=0.5, label="Konfiantza-tartea")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/kargako_grafikuek/karga_capacity_{bat_ind}.png')
        
        #Outlierrak ez diren zikloak igaro
        new_cicle_number = 1
        for cycle, knee, Q_value, max_Qc_value, min_Qc_value, mean_V_, std_V_, max_I_charge_, min_I_charge_, max_I_discharge_, min_I_discharge_ in zip(cycle_numbers, battery_knees, _Q, max_Qc, min_Qc, mean_V, std_V, max_I_charge, min_I_charge, max_I_discharge, min_I_discharge, ):
            if cycle not in total_outliers:
                clean_data.append([bat_ind, new_cicle_number, knee, Q_value, max_Qc_value, min_Qc_value, mean_V_, std_V_, max_I_charge_, min_I_charge_, max_I_discharge_, min_I_discharge_])
                new_cicle_number += 1
        
    
    # Azken  DataFrame-a
    df_clean = pd.DataFrame(clean_data, columns=['bat_name', 'cycle', 'knee', 'Q_value', 'max_Qc', 'min_Qc','mean_V', 'std_V', 'max_I_charge', 'min_I_charge', 'max_I_discharge', 'min_I_discharge'])
    path = './data/processed/NASA/NASA_karga/'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_clean.to_pickle('./data/processed/NASA/NASA_karga/DataSetRulEstimation_all_new.pkl')

    return df_clean

def knee_cov_karga(cut_off_down, cut_off_up):
    allfiles = glob.glob('./data/processed/NASA/*.csv')
    
    for indfile in range(len(allfiles)): 
        plt.clf()
        # Bateriaren izena
        bat_ind = os.path.basename(allfiles[indfile]).split('_')[0]
        print(f'Processing Charge curves for: {bat_ind}')

        # Datuak kargatu
        df_temp = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_temp.loc[:, ~df_temp.columns.str.contains("Unnamed")]

        # Koloreak eta ziklo garrantzitsuak
        cmap = plt.cm.viridis  
        ziklo_guztiak = df_battery.cycle.unique()
        azken_zikloa = ziklo_guztiak[-1]
        colors = cmap(np.linspace(0, 1, len(ziklo_guztiak)))
        
        # Irudikatzeko zikloen hautaketa (200 ziklotik behin)
        irudikatzeko_zikloak = {ziklo_guztiak[0], azken_zikloa}
        for zikloa in ziklo_guztiak:
            if zikloa % 200 == 0:
                irudikatzeko_zikloak.add(zikloa)
        irudikatzeko_zikloak = sorted(list(irudikatzeko_zikloak))

        knee_points = {}
        max_q_points = {}
        
        # Legendak behin bakarrik agertzeko
        cutoff_label_added = False
        knee_label_added = False

        # 1. ZIKLOZ ZIKLO PROZESATU
        for i, ind in enumerate(ziklo_guztiak): 
            df_cycle = df_battery[df_battery['cycle'] == ind]
            df = df_cycle[df_cycle['I'] > 0].reset_index(drop=True) # Karga soilik
            
            if df.empty: continue
            
            # Tentsio eta ahalmen mugak aplikatu
            if (df['V'] >= cut_off_down).any():
                df = df[df['V'] >= cut_off_down].reset_index(drop=True)
            if (df['V'] > cut_off_up).any():
                last_idx = df[df['V'] <= cut_off_up].index[-1]
                df = df.loc[:last_idx].reset_index(drop=True)
            if (df['Qc'] > 1.1).any():
                df = df[df['Qc'] <= 1.1].reset_index(drop=True)

            if df.empty or len(df) < 5: continue

            # Kurbak marraztu (Ziklo 1 eta azkena legendan)
            label = ""
            if ind == 1: label = 'Ziklo 1'
            elif ind == azken_zikloa: label = f'Ziklo {azken_zikloa}'
            
            plt.scatter(df['Qc'], df['V'], color=colors[i], s=0.5, label=label if label else None)

            # Knee eta Max puntuaren kalkulua (cut-off voltage == maximoa)
            if ind in irudikatzeko_zikloak:
                try:
                    kQ, kV = knee_per_cycle_karga(df, 'Qc', 'V')
                    knee_points[ind] = (kQ, kV)
                except:
                    pass
                max_q_points[ind] = (df['Qc'].iloc[-1], df['V'].iloc[-1])

        # 2. PUNTUAK MARRAZTU (Knee eta Cut-off)
        for cycle in irudikatzeko_zikloak:
            # A) Cut-off puntuak (Borobil beltzak)
            if cycle in max_q_points:
                mQ, mV = max_q_points[cycle]
                size = 50 if cycle == 1 or cycle == azken_zikloa else 30
                lbl = 'Cut-off voltage' if not cutoff_label_added else ""
                plt.scatter(mQ, mV, color='black', marker='o', s=size, zorder=15, label=lbl)
                cutoff_label_added = True
                
                # Etiketak 
                if cycle == 1:
                    plt.annotate('Ziklo 1', (mQ, mV), textcoords="offset points", xytext=(5,-10), ha='left', fontsize=8)
                elif cycle == azken_zikloa:
                    plt.annotate(f'Ziklo {cycle}', (mQ, mV), textcoords="offset points", xytext=(-5,-10), ha='right', fontsize=8)

            # B) Knee puntuak (Triangelu beltzak)
            if cycle in knee_points:
                kQ, kV = knee_points[cycle]
                size = 40 if cycle == 1 or cycle == azken_zikloa else 20
                lbl = 'Knee' if not knee_label_added else ""
                plt.scatter(kQ, kV, color='black', marker='^', s=size, zorder=20, label=lbl)
                knee_label_added = True

                # Etiketak 
                if cycle == 1:
                    plt.annotate('Ziklo 1', (kQ, kV), textcoords="offset points", xytext=(15,-10), ha='left', fontsize=8)
                elif cycle == azken_zikloa:
                    plt.annotate(f'Ziklo {cycle}', (kQ, kV), textcoords="offset points", xytext=(20,5), ha='right', fontsize=8)

        # 3. GRAFIKOA
        plt.title(f'Karga kurba')
        plt.xlabel(r'$\text{Karga ahalmena (Q)}$', fontsize=14)
        plt.ylabel(r'$\text{Tentsioa (V)}$', fontsize=14)
        plt.xlim(0.0, 1.1)
        plt.legend()
        plt.tight_layout()
        
        output_dir = './results/kargako_grafikuek/knee_cov/'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/karga_cov_knee_{bat_ind}.png')
    
    return

def capacity_kneepoint_calculator_karga(cut_off_down, cut_off_up):
    kargako_datuak = pd.read_pickle('./data/processed/NASA/NASA_karga/DataSetRulEstimation_all_new.pkl')
    karga_q_value = kargako_datuak[['bat_name', 'cycle', 'Q_value']].copy()
    
    print("Knee Point-a kalkulatzen (Karga)...")
    knees_karga_all = knee(karga_q_value, output='Q_value')
    knees_karga_all = knees_karga_all.rename(columns={'knee_cycle': 'knee_cycle_karga', 'knee_capacity': 'knee_capacity_karga'})    
    allfiles = glob.glob('./data/processed/NASA/*.csv')

    for indfile in range(len(allfiles)):    # Para cada batería
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        print('Processing File (or battery) '+ allfiles[indfile])

        df_temp = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_temp.loc[:, ~df_temp.columns.str.contains("Unnamed")]


        bat_ind = allfiles[indfile].split('\\')[1].split('_')[0]

        # Bateria honetarako knee datuak lortu
        knee_info_bat = knees_karga_all[knees_karga_all['bat_name'] == bat_ind]
        if knee_info_bat.empty:
            print(f"Ez da knee point-ik aurkitu {bat_ind} bateriarako. Saltatu.")
            plt.close(fig)
            continue
            
        knee_cycle_num = knee_info_bat['knee_cycle_karga'].iloc[0]
        battery_knees = []                                          
        problematic_cycles = []                                     
        cycle_numbers = []                                         
        _Q = []                                                     
        
        for ind in df_battery.cycle.unique(): 

            df_cycle = df_battery[df_battery['cycle'] == ind]
            df = df_cycle[df_cycle['I'] > 0]
            df = df.reset_index(drop=True)
            
            add_plot = True
            
            df['dV_dQc'] = np.gradient(df['V'], df['Qc'])

            
           # 1. Beheko muga hautatu (V >= cut_off_down)
        
            if (df['V'] > cut_off_down).any():
                first_index = df[df['V'] >= cut_off_down].index[0] 
                df = df.loc[first_index:].reset_index(drop=True)
            else:
                add_plot = False
                
            # 2. Goiko muga moztu (V <= cut_off_up)
        
            if (df['V'] > cut_off_up).any():
                
                last_index = df[df['V'] <= cut_off_up].index[-1] if (df['V'] <= cut_off_up).any() else len(df)
                df = df.loc[:last_index].reset_index(drop=True)
            if (df['Qc'] > 1.1).any():
                last_index_Qc = df[df['Qc']>1.1].index[0] #Qc hori 1.1 balioa gainditzen duen lehen indizea aurkitu eta ondorengo datu guztiak moztu
                df = df.loc[:last_index_Qc - 1].reset_index(drop=True) #Karga-ahalmena 1.1 baino handiagoa den balioa hartu baino lehenagora arte moztu
            
            #3. Ziklo problematikoak
            is_problematic = False
            if len(df['V']) < 20 or (df.loc[df['V'] == cut_off_up, 'Qc'] <= 0.2).any():
                is_problematic = True
                print(f'Este ciclo {ind} está mal registrado, hay que eliminarlo')
            
            # 4. Balioak gorde
            cycle_numbers.append(ind)
            if is_problematic:
                problematic_cycles.append(ind)
                battery_knees.append(np.nan)
                _Q.append(np.nan) 
            
            else:
                # knee_onset kalkulua (outliers_1 lortzeko)
                knee_onset = knee_per_cycle(df, 'Qc', 'V')
                battery_knees.append(knee_onset)
                _Q.append(df['Qc'].max())             
                       
        
        # Eliminación de los ciclos ruidosos (ciclos, no índices)
        outliers_1 = detect_outliers(battery_knees, 0.05, 1.07)  # El codo nunca puede estar antes de una capacidad de carga del 0.05 ni a partir de 1.07
            
        ##################### KNEES ##############################
        y_c_raw = np.array(_Q)
        x_c_raw = np.array(cycle_numbers)

        # 2. NaN-ak eta haiei dagozkien ziklo-zenbakiak ezabatu
        valid_mask = ~np.isnan(y_c_raw)

        x_c = x_c_raw[valid_mask]
        y_c = y_c_raw[valid_mask]

        # Polynomial fit
        degree = 15
        coeffs_c = np.polyfit(x_c, y_c, degree)
        p_c = np.poly1d(coeffs_c)

        # Fitted values
        y_fit_c = p_c(x_c)

        # Residuals & degrees of freedom
        residuals_c = y_c - y_fit_c
        n = len(y_c)
        m = len(coeffs_c)  # Number of coefficients (degree + 1)
        dof = n - m  # Degrees of freedom

        # Standard error & confidence interval
        residual_std_error_c = np.std(residuals_c, ddof=m)
        multiplicador_c = 2  # Aumenta para hacer que los outliers sean más lejanos
        conf_interval_c = multiplicador_c * residual_std_error_c

        y_upper_c = y_fit_c + conf_interval_c
        y_lower_c = y_fit_c - conf_interval_c
        outliers_3 = (y_c < y_lower_c) | (y_c > y_upper_c)

        # Ciclos outliers
        outlier_points_3 = df_battery.cycle.unique()[outliers_3]

        total_outliers = np.unique(np.concatenate([problematic_cycles, outliers_1, outlier_points_3]))        

        # ----------------------------------------------------------------------
        # 5. GRAFIKOA MARRAZTU (Q vs Cycle)
        # ----------------------------------------------------------------------
        
        # 5.1. Degradazio kurba marraztu
        ax.scatter(x_c, y_c, color='green', s=5, label="Bateriaren ahalmena")
        
        # 5.2. Outliers marraztu
        # Outlier diren ziklo-zenbakien posizioak x_c array-an aurkitu
        is_outlier_in_valid_data = np.isin(x_c, total_outliers)
        outlier_cycles_to_plot = x_c[is_outlier_in_valid_data]
        outlier_Q_values = y_c[is_outlier_in_valid_data]

        ax.scatter(outlier_cycles_to_plot, outlier_Q_values, color='red', s=5, label="outlierrak")        
        # 5.3. Fitted Line eta Confidence Interval marraztu
        ax.plot(x_c, y_fit_c, label="Doitutako polinomioa", color="green")
        ax.fill_between(x_c, y_lower_c, y_upper_c, color="green", alpha=0.5, label="Konfiantza-tartea")
        
        # 5.4. KNEE-POINT MARRA BERTIKALA APLIKATU
        ax.axvline(
            x=knee_cycle_num, # Marraren X posizioa (Zikloaren zenbakia)
            color='black', 
            linestyle = '--', 
            linewidth=2,
            label=f'Knee-point: {knee_cycle_num}'
        )
        
        # 5.5. Etiketa eta Gainerako Elementuak
        ax.set_xlabel('Ziklo kopurua', fontsize=12)
        ax.set_ylabel('Karga-ahalmena (Q)', fontsize=12)
        ax.set_title('Degradazio-kurba (cut-off voltage puntuak)')
        ax.legend()
        plt.tight_layout()
        
        # 5.6. Gorde
        path = f'./results/karga_capacity_kneepoint/capacity_kneepoint_{bat_ind}.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close(fig)

    return 

##################################
#       Ziklo baten irudikapena
###################################

def ziklo(ziklo_zenbakia=10):
    allfiles = glob.glob('./data/processed/NASA/*.csv')
    

    for indfile in range(len(allfiles)):
        plt.clf()
        bat_ind = os.path.basename(allfiles[indfile]).split('_')[0]
        print(f'Prozesatzen: {bat_ind}')

        # Datuak kargatu
        df_temp = pd.read_csv(allfiles[indfile], sep=',', index_col=None)
        df_battery = df_temp.loc[:, ~df_temp.columns.str.contains("Unnamed")]

        # Aukeratutako zikloaren datuak hartu
        df_cycle = df_battery[df_battery['cycle'] == ziklo_zenbakia]

        if df_cycle.empty:
            print(f"Ezin izan da {ziklo_zenbakia} zikloa aurkitu {bat_ind} baterian.")
            continue

        # 1. KARGA DATUAK (I > 0)
        df_charge = df_cycle[df_cycle['I'] > 0]
        # 2. DESKARGA DATUAK (I < 0)
        df_discharge = df_cycle[df_cycle['I'] < 0]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Karga irudikatu (Qc erabiliz)
        ax.plot(df_charge['Qc'], df_charge['V'], color='blue', label=f'Karga', linewidth=2)
        
        # Deskarga irudikatu (Qd erabiliz)
        # Oharra: Qd normalean 0-tik hasten da deskarga hasieran
        ax.plot(df_discharge['Qd'], df_discharge['V'], color='red', label=f'Deskarga', linewidth=2)

        # Grafikoaren xehetasunak
        output_dir = './results/ziklo/'
        os.makedirs(output_dir, exist_ok=True)
        ax.set_title(f'Karga-Deskarga Ziklo Oso Bat', fontsize=14)
        ax.set_xlabel('Ahalmena (Q)', fontsize=12)
        ax.set_ylabel('Tentsioa (V)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/ziklo_{bat_ind}.png')
        plt.close()

    return

