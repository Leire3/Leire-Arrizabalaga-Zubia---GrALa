## leireren_lana

## Ikasketa automatikoaren aplikazioa litio-ioizko baterien degradazioa aurresateko.

Proiektu honetan litio-ioizko baterien degradazioa estimatzeko eredua garatzen da, karga- eta deskarga-kurbetako datuak erabiliz. Lan hau Gradu Amaierako Lanaren (GrAL) parte da.

## Deskribapena
Proiektuaren helburua baterien degradazioa aztertzea da, ikasketa automatikoko hiru algoritmo (Random Forest Erregresioa, Extreme Gradient Boosting eta Sare neuronalak) erabiliz. Horretarako, kurben ezaugarrien bidez, ereduak osasun-egoeraren degradazioa iragartzen ikasten du.

## Proiektuaren Egitura
Biltegi honetako fitxategiak honela antolatuta daude:
* **src/**: Proiektuaren kode nagusia
    * **knee_evolution_battery.py** Kode nagusia da hau. Honek iragarpen-prozesu osoa kudeatzen du: exekuzio-parametroak jaso, baterien datu-basea kargatu eta osasun-egoeraren estimazio-eredua exekutatzen du.
    * **knee_calculator_cycle.py** Kode honetan hainbat funtzio ageri dira, funtzio nagusienak karga- zein deskarga-prozesuetarako knee puntuaren kalkulua egitean datza. Horrez gain, beste hainbat funtzio daude: knee eta cut-off voltage puntuak irudikatzeko, ziklo baten irudikapena egiteko... 
    * **ereduen_eta_rmse_grafikoak.py** Kode honen bidez GrALerako beharrezkoak izan diren hainbat grafiko gehigarri egiten dira.
* **utils/**: Kode honetan baterien datu-basearen prozesamendua eta ikasketa automatikoko ereduen (Random Forest Erregresioa, XGBoost eta Sare nueronalak) iragarpenak egiten ditu.
* **requirements.txt**: Kodea exekutatzeko beharrezkoak diren Python liburutegien zerrenda.
* **.vscode/**: Kodea exekutatzeko parametroak zehazten ditu.

## Instalazioa eta Erabilera

1. Lehenik eta behhin, proiektua deskargatu zure ordenagailura.
2. Beharrezkoak diren liburutegi guztiak instalatzeko, erabili "requirements.txt" fitxategia: 
    pip install -r requirements.txt
3. Kodea exekutatu. Horretarako, "launch.json" fitxategian daude beharrezkoak diren parametroak. 