

'''UNIONE DATASET DALLA 2 ELEMENTARE ALLA 5 SUPERIORE IN UN SOLO DATASET
CHE CONTENGA SIA I RISULTATI DI MATEMATICA CHE DI ITALIANO: sono state
considerate solo le colonne comuni a tutti i dataset, la colonna WLE_mat e WLE_ita
è stata accorpata in un'unica colona WLE ed è stata aggiunta una colonna MATERIA
per specificare a che materia si riferisca il punteggio'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency

# caricare i file csv con il separatore corretto
df_2elem = pd.read_csv("2024_Matrice_Campione_MAT_02_0_1_WLE_anonima.csv", sep=";", encoding="latin1", header=0)
df_5elem = pd.read_csv("2024_Matrice_Campione_MAT_05_0_1_WLE_anonima.csv", sep=";", encoding="latin1", header=0)
df_3media = pd.read_csv("2024_Matrice_Campione_MAT_08_0_1_WLE_anonima.csv", sep=",", encoding="latin1", header=0)
df_2sup = pd.read_csv("2024_Matrice_Campione_MAT_10_0_1_WLE_anonima.csv", sep=";", encoding="latin1", header=0)
df_5sup = pd.read_csv("2024_Matrice_Campione_MAT_13_0_1_WLE_anonima.csv", sep=",", encoding="latin1", header=0)

# Trasformare i nomi delle colonne in minuscole
df_2elem.columns = df_2elem.columns.str.lower()
df_5elem.columns = df_5elem.columns.str.lower()
df_3media.columns = df_3media.columns.str.lower()
df_2sup.columns = df_2sup.columns.str.lower()
df_5sup.columns = df_5sup.columns.str.lower()

# mostrare le prime righe di ogni dataset
print("Prime righe nel dataset 2elem:")
print(df_2elem.head())

print("\nPrime righe nel dataset 5elem:")
print(df_5elem.head())

print("\nPrime righe nel dataset 3media:")
print(df_3media.head())

print("\nPrime righe nel dataset 2sup:")
print(df_2sup.head())

print("\nPrime righe nel dataset 5sup:")
print(df_5sup.head())

# confrontare le colonne ignorando maiuscole e spazi
colonne_2elem = [col.strip() for col in df_2elem.columns]
colonne_5elem = [col.strip() for col in df_5elem.columns]
colonne_3media = [col.strip() for col in df_3media.columns]
colonne_2sup = [col.strip() for col in df_2sup.columns]
colonne_5sup = [col.strip() for col in df_5sup.columns]

# trovare le colonne comuni tra tutti i dataset
colonne_comuni = set(colonne_2elem) & set(colonne_5elem) & set(colonne_3media) & set(colonne_2sup) & set(colonne_5sup)

# filtra i dataset mantenendo solo le colonne comuni
df_2elem = df_2elem.loc[:, df_2elem.columns.isin(colonne_comuni)]
df_5elem = df_5elem.loc[:, df_5elem.columns.isin(colonne_comuni)]
df_3media = df_3media.loc[:, df_3media.columns.isin(colonne_comuni)]
df_2sup = df_2sup.loc[:, df_2sup.columns.isin(colonne_comuni)]
df_5sup = df_5sup.loc[:, df_5sup.columns.isin(colonne_comuni)]

# mostra i nomi delle colonne per i dataset filtrati
print("\nColonne nel dataset 2elem dopo il filtraggio:")
print(df_2elem.columns)

print("\nColonne nel dataset 5elem dopo il filtraggio:")
print(df_5elem.columns)

print("\nColonne nel dataset 3media dopo il filtraggio:")
print(df_3media.columns)

print("\nColonne nel dataset 2sup dopo il filtraggio:")
print(df_2sup.columns)

print("\nColonne nel dataset 5sup dopo il filtraggio:")
print(df_5sup.columns)

# concatenare i dataset mantenendo solo le righe
df_concat = pd.concat([df_2elem, df_5elem, df_3media, df_2sup, df_5sup], ignore_index=True)

 

# Pulizia dei doppi apici e conversione dei numeri con virgola in formato decimale
for col in df_concat.columns:
    if df_concat[col].dtype == "object":  # Controlla solo le colonne testuali
        df_concat[col] = df_concat[col].astype(str).str.replace('"', '')  # Rimuove i doppi apici
        df_concat[col] = df_concat[col].str.replace(r'(\d+),(\d+)', r'\1.\2', regex=True)  # Sostituisce solo i numeri con virgole

# mostrare il risultato
print("Dataframe unito con le righe:")
print(df_concat.head())

# salvare il dataset risultante in un file csv
df_concat.to_csv('dataset_mat_unito_concatenato.csv', index=False)

# calcoliamo il numero di righe del dataframe concatenato
num_righe = df_concat.shape[0]
print(num_righe)

# conta i valori nulli nella colonna 'grado' per ogni dataset
print(df_2elem['grado'].isna().sum())
print(df_5elem['grado'].isna().sum())
print(df_3media['grado'].isna().sum())
print(df_2sup['grado'].isna().sum())
print(df_5sup['grado'].isna().sum())


# caricare i file csv con il separatore corretto
df_2elem_it = pd.read_csv("2024_Matrice_Campione_ITA_02_0_1_WLE_anonima.csv", sep=";", encoding="latin1", header=0)
df_5elem_it = pd.read_csv("2024_Matrice_Campione_ITA_05_0_1_WLE_anonima.csv", sep=";", encoding="latin1", header=0)
df_3media_it = pd.read_csv("2024_Matrice_Campione_ITA_08_0_1_WLE_anonima.csv", sep=",", encoding="latin1", header=0)
df_2sup_it = pd.read_csv("2024_Matrice_Campione_ITA_10_0_1_WLE_anonima.csv", sep=";", encoding="latin1", header=0)
df_5sup_it = pd.read_csv("2024_Matrice_Campione_ITA_13_0_1_WLE_anonima.csv", sep=",", encoding="latin1", header=0)

# Trasformare i nomi delle colonne in minuscole
df_2elem_it.columns = df_2elem_it.columns.str.lower()
df_5elem_it.columns = df_5elem_it.columns.str.lower()
df_3media_it.columns = df_3media_it.columns.str.lower()
df_2sup_it.columns = df_2sup_it.columns.str.lower()
df_5sup_it.columns = df_5sup_it.columns.str.lower()

# mostra le prime righe di ogni dataset
print("Prime righe nel dataset 2elem:")
print(df_2elem_it.head())

print("\nPrime righe nel dataset 5elem:")
print(df_5elem_it.head())

print("\nPrime righe nel dataset 3media:")
print(df_3media_it.head())

print("\nPrime righe nel dataset 2sup:")
print(df_2sup_it.head())

print("\nPrime righe nel dataset 5sup:")
print(df_5sup_it.head())

colonne_2elem_it = [col.strip() for col in df_2elem_it.columns]
colonne_5elem_it = [col.strip() for col in df_5elem_it.columns]
colonne_3media_it = [col.strip() for col in df_3media_it.columns]
colonne_2sup_it = [col.strip() for col in df_2sup_it.columns]
colonne_5sup_it = [col.strip() for col in df_5sup_it.columns]

colonne_comuni_it = set(colonne_2elem_it) & set(colonne_5elem_it) & set(colonne_3media_it) & set(colonne_2sup_it) & set(colonne_5sup_it)

# filtrare i dataset mantenendo solo le colonne comuni
df_2elem_it = df_2elem_it.loc[:, df_2elem_it.columns.isin(colonne_comuni_it)]
df_5elem_it = df_5elem_it.loc[:, df_5elem_it.columns.isin(colonne_comuni_it)]
df_3media_it = df_3media_it.loc[:, df_3media_it.columns.isin(colonne_comuni_it)]
df_2sup_it = df_2sup_it.loc[:, df_2sup_it.columns.isin(colonne_comuni_it)]
df_5sup_it = df_5sup_it.loc[:, df_5sup_it.columns.isin(colonne_comuni_it)]

# mostrare i nomi delle colonne per i dataset filtrati
print("\nColonne nel dataset 2elem dopo il filtraggio:")
print(df_2elem_it.columns)

print("\nColonne nel dataset 5elem dopo il filtraggio:")
print(df_5elem_it.columns)

print("\nColonne nel dataset 3media dopo il filtraggio:")
print(df_3media_it.columns)

print("\nColonne nel dataset 2sup dopo il filtraggio:")
print(df_2sup_it.columns)

print("\nColonne nel dataset 5sup dopo il filtraggio:")
print(df_5sup_it.columns)

# concatenare i dataset mantenendo solo le righe
df_concat_it = pd.concat([df_2elem_it, df_5elem_it, df_3media_it, df_2sup_it, df_5sup_it], ignore_index=True)

# Pulizia dei doppi apici e conversione dei numeri con virgola in formato decimale
for col in df_concat_it.columns:
    if df_concat_it[col].dtype == "object":  # Controlla solo le colonne testuali
        df_concat_it[col] = df_concat_it[col].astype(str).str.replace('"', '')  # Rimuove i doppi apici
        df_concat_it[col] = df_concat_it[col].str.replace(r'(\d+),(\d+)', r'\1.\2', regex=True)  # Sostituisce solo i numeri con virgole

# mostrare il risultato
print("Dataframe unito con le righe:")
print(df_concat_it.head())

df_concat_it.to_csv('dataset_ita_unito_concatenato.csv', index=False)

# calcoliamo il numero di righe del dataframe concatenato
num_righe = df_concat_it.shape[0]
print(num_righe)

# contare i valori nulli nella colonna 'grado' per ogni dataset
print(df_2elem_it['grado'].isna().sum())
print(df_5elem_it['grado'].isna().sum())
print(df_3media_it['grado'].isna().sum())
print(df_2sup_it['grado'].isna().sum())
print(df_5sup_it['grado'].isna().sum())



# Supponiamo che df_mat e df_ita siano i due dataframe originali

df_mat = df_concat.copy()
df_mat['materia'] = 'matematica'
df_mat = df_mat.rename(columns={'wle_mat_200': 'wle'})

df_ita = df_concat_it.copy()
df_ita['materia'] = 'italiano'
df_ita = df_ita.rename(columns={'wle_ita_200': 'wle'})

# Unire due dataframe (concatenazione verticale)
df_unito = pd.concat([df_mat, df_ita], ignore_index=True)

print("Dataframe unito:")
print(df_unito.head())

# Calcolare il numero di righe del DataFrame
num_righe = df_unito.shape[0]
print(f"Numero di righe: {num_righe}")

df_before = df_unito.copy()

df_unito.to_csv('dataset_generale_concatenato.csv', index=False)

#controllo
print(df_mat.columns)
print(df_ita.columns)


'''PREPROCESSING VOLTO A ELIMINARE I VALORI MANCANTI: attraverso i TRACCIATI dei vari dataset
possiamo capire quali valori siano mancanti'''

# Definire le colonne da convertire in interi e quelle da convertire in float
colonne_intere = [
    "scuola_anonimo", "classe_anonimo", "studente_anonimo", "grado", "genere",
    "mese", "anno", "luogo", "regolarità", "origine", "codice_orario",
    "luogo_padre", "luogo_madre", "cod_reg"
]
colonne_float = [
    "wle", "peso_studente", "peso_classe", "peso_scuola"
]

# Convertire le colonne specificate
for col in colonne_intere:
    df_unito[col] = pd.to_numeric(df_unito[col], errors='coerce').astype('Int64')

for col in colonne_float:
    df_unito[col] = pd.to_numeric(df_unito[col], errors='coerce').astype(float)

# Controllo finale sui tipi di dati
print(df_unito.dtypes)

print(df_unito)

for col in ['grado', 'genere', 'luogo', 'regolarità', 'origine',
            'luogo_padre', 'luogo_madre', 'cod_reg', 'wle', 'cod_reg', 'areageo_5']:
    print(f"Valori unici per {col}: {df_unito[col].unique()}")


print(df_unito['genere'].unique())


# Verificare la quantità di righe con NaN nella colonna 'genere'
count_nan = df_unito['genere'].isna().sum()
print(f"Numero di righe con NaN nella colonna 'genere': {count_nan}")

# Eliminare le righe che contengono NaN nella colonna 'genere'
df_unito = df_unito.dropna(subset=['genere'])

# Verificare i valori unici dopo aver rimosso i NaN
print(df_unito['genere'].unique())

# Verificare la quantità di righe con NaN nella colonna 'genere'
count_nan = df_unito['genere'].isna().sum()
print(f"Numero di righe con NaN nella colonna 'genere': {count_nan}")

count_9 = df_unito['genere'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'genere': {count_9}")

df_unito = df_unito[df_unito['genere'] != 9]#elimino quelle righe
count_9 = df_unito['genere'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'genere': {count_9}")

# Verificare i valori unici della colonna 'genere' dopo la rimozione
print(df_unito['genere'].unique())



#GRADO


# Definire i valori ammessi
valori_ammessi = {2, 5, 8, 10, 13}

# Contare le righe che NON contengono questi numeri
righe_non_conformi = df_unito[~df_unito['grado'].isin(valori_ammessi)].shape[0]

print(f"Numero di righe con valori diversi da {valori_ammessi} nella colonna 'grado': {righe_non_conformi}")



#nella colonna LUOGO il valore mancante è indicato con 9
count_9 = df_unito['luogo'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'luogo': {count_9}")

df_unito = df_unito[df_unito['luogo'] != 9]#elimino quelle righe
count_9 = df_unito['luogo'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'luogo': {count_9}")

count_mese = df_unito['mese'].eq(99).sum()
print(f"Numero di righe con valore 99 nella colonna 'mese': {count_mese}")

count_anno = df_unito['anno'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'anno': {count_anno}")

#REGOLARITA (indica se studente è regolare o meno)


#nella colonna LUOGO il valore mancante è indicato con 9
count_9 = df_unito['regolarità'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'regolarità': {count_9}")
nan_count = df_unito['regolarità'].isna().sum()
print(f"Numero di NaN nella colonna 'regolarità': {nan_count}")


count_9 = df_unito['origine'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'origine': {count_9}")
df_unito = df_unito[df_unito['origine'] != 9]#elimino quelle righe
count_9 = df_unito['origine'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'origine': {count_9}")
nan_count = df_unito['origine'].isna().sum()
print(f"Numero di NaN nella colonna 'origine': {nan_count}")

# Mappa numerica a categorie per la colonna 'origine'
origine_mapping = {
    1: 'Nativo',
    2: 'Straniero I generazione',
    3: 'Straniero II generazione'
}
df_unito['origine'] = df_unito['origine'].map(origine_mapping)


count_p = df_unito['luogo_padre'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'luogo_padre': {count_p}")
df_unito = df_unito[df_unito['luogo_padre'] != 9]#elimino quelle righe
count_p = df_unito['luogo_padre'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'luogo_padre': {count_p}")

count_m = df_unito['luogo_madre'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'luogo_madre': {count_m}")
df_unito = df_unito[df_unito['luogo_madre'] != 9]#elimino quelle righe
count_m = df_unito['luogo_madre'].eq(9).sum()
print(f"Numero di righe con valore 9 nella colonna 'luogo_madre': {count_m}")


# Calcolare la matrice di correlazione solo per le colonne numeriche (INCLUDO AREA GEOGRAFICA CHE POI SARA' TRAFORMATA IN STRINGA)
correlation_matrix_full = df_unito.select_dtypes(include=['number']).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_full, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

plt.title("Heatmap delle Correlazioni (Tutte le Variabili)")
plt.show()


#le aree della penisola vengono ricondotte a 3: nord, sud, centro
area_geo_map = {1: 'Nord', 2: 'Nord', 3: 'Centro', 4: 'Sud', 5:'Sud'}
df_unito['areageo_5'] = df_unito['areageo_5'].map(area_geo_map)

print("Dataframe unito:")
print(df_unito.head())
# Calcolare il numero di righe del DataFrame
num_righe = df_unito.shape[0]
print(f"Numero di righe dopo PRE-PROCESSING: {num_righe}")


# 2. Conteggi per grado e materia prima e dopo la pulizia
conteggio_prima = df_before.groupby(['materia', 'grado']).size().unstack(fill_value=0)
conteggio_dopo = df_unito.groupby(['materia', 'grado']).size().unstack(fill_value=0)

# 3. Costruzione del DataFrame a due livelli di colonne
confronto_grado_materia = pd.concat(
    {'conteggio_prima': conteggio_prima, 'conteggio_dopo': conteggio_dopo},
    axis=1
)

# 4. Salvare a file CSV
confronto_grado_materia.to_csv('report_grado_materia_before_after.csv')

print("Salvato report in 'report_grado_materia_before_after.csv'")


# Calcolare la matrice di correlazione solo per le colonne numeriche
correlation_matrix_full = df_unito.select_dtypes(include=['number']).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_full, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

plt.title("Heatmap delle Correlazioni (Tutte le Variabili)")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df_unito, x='genere', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per genere')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df_unito, x='genere', y='wle', hue='materia', split=True)
plt.title('Distribuzione dei punteggi per Genere e Materia')
plt.show()


#%%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_unito, x='grado', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per grado e materia')
plt.show()

#%%
sns.boxplot(data=df_unito, x='regolarità', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per regolarità e materia')
plt.show()




# Filtriamo solo le materie "matematica" e "italiano"
df_materie = df_unito[df_unito['materia'].isin(['matematica', 'italiano'])]
# Creare il boxplot separato per materia
plt.figure(figsize=(14, 8))
sns.boxplot(x='origine', y='wle', hue='materia', data=df_materie, palette="Set2")

# Aggiungere titolo e etichette
plt.title("Distribuzione dei Risultati WLE per Origine nelle Materie Matematica e Italiano", fontsize=16)
plt.xlabel('Origine')
plt.ylabel('WLE')

# Mostrare il grafico
plt.tight_layout()
plt.show()

#%%
sns.boxplot(data=df_unito, x='areageo_5', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per area e materia')
plt.show()


# Creiamo un boxplot per visualizzare la distribuzione del WLE per genere
plt.figure(figsize=(8,6))
sns.boxplot(x='genere', y='wle', data=df_unito, palette="Set2")

# Aggiungiamo titolo e etichette
plt.title('Distribuzione del WLE per Genere')
plt.xlabel('Genere')
plt.ylabel('WLE')

# Mostrare il grafico
plt.grid(True)
plt.show()
#%%
# Controllare i valori unici di 'genere', 'areageo_5', e 'materia'
print(df_unito['genere'].unique())
print(df_unito['areageo_5'].unique())
print(df_unito['materia'].unique())

contingency_table = pd.crosstab([df_unito['genere'], df_unito['areageo_5']], df_unito['materia'])
print(contingency_table)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_unito, x='areageo_5', y='wle', hue='genere', palette='Set2')
plt.title('Distribuzione dei punteggi per Area Geografica e Genere')
plt.show()




# Creazione del catplot
g = sns.catplot(data=df_unito, x='areageo_5', y='wle', hue='genere', kind='box',
                col='materia', palette='Set2', height=6, aspect=1.5)

# Aggiungere etichette per gli assi
g.set_axis_labels('Area Geografica', 'WLE')

# Modificare il titolo per includere il nome della materia in modo chiaro
g.set_titles('Distribuzione dei punteggi per Area Geografica, Genere e Materia: {col_name}')

# Aggiungere una legenda per il genere
g.add_legend(title="Genere")

# Mostrare il grafico
plt.show()


# 1. Caricare la geometria delle regioni italiane
url_regioni = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson"
gdf_regioni = gpd.read_file(url_regioni)

# 2. Verificare i dati
print(df_unito['wle'].isna().sum())  # Conta i NaN
print((df_unito['wle'] <= 0).sum())  # Conta i valori negativi
print(df_unito['wle'].unique())
print(df_unito['wle'].dtype)  # Verifica il tipo della colonna

# Verificare se tutte le geometrie sono valide
print(gdf_regioni.is_valid.sum())  # Dovrebbe restituire il numero totale di geometrie valide
gdf_regioni['area'] = gdf_regioni.geometry.area
print(gdf_regioni[gdf_regioni['area'] == 0])  # Mostra le regioni con area zero
print(gdf_regioni.head())  # Dovresti vedere almeno alcune righe di dati

# 3. Mappatura Cod_Reg a nomi regione (versione migliorata)
regioni_mapping = {
    1: "Valle d'Aosta/Vallée d'Aoste", 2: "Piemonte", 3: "Liguria",
    4: "Lombardia", 6: "Veneto", 7: "Friuli-Venezia Giulia",
    8: "Emilia-Romagna", 9: "Toscana", 10: "Umbria", 11: "Marche",
    12: "Lazio", 13: "Abruzzo", 14: "Molise", 15: "Campania",
    16: "Puglia", 17: "Basilicata", 18: "Calabria", 19: "Sicilia",
    20: "Sardegna", 51: "Trentino-Alto Adige/Südtirol",
    54: "Trentino-Alto Adige/Südtirol"
}
df_unito["regione"] = df_unito["cod_reg"].map(regioni_mapping)

# 4. Calcolare la media dei livelli per regione, genere e materia
df_aggregato = df_unito.groupby(["regione", "genere", "materia"])["wle"].mean().reset_index()




# 5. Unire i dati alla geometria separato per genere e materia
for materia in df_aggregato['materia'].unique():
    for genere in [1, 2]:  # 1 = maschi, 2 = femmine
        df_genere_materia = df_aggregato[(df_aggregato["genere"] == genere) & (df_aggregato["materia"] == materia)]
        mappa = gdf_regioni.merge(df_genere_materia, left_on="reg_name", right_on="regione")

        # 6. Creare la mappa tematica
        fig, ax = plt.subplots(figsize=(14, 10))  # Create fig and ax

        mappa.plot(column="wle",
                   cmap="Blues",
                   legend=True,
                   # Remove legend_kwds from here
                   edgecolor="white",
                   linewidth=0.5,
                   ax=ax,  # Use ax for the plot
                   missing_kwds={"color": "lightgrey", "label": "Dato mancante"})

        plt.title(f"Livelli di Competenza in {materia} per Regione ({'Maschi' if genere == 1 else 'Femmine'})", fontsize=16, pad=20)
        ax.set_axis_off()

        # 7. Aggiungere etichette
        for idx, row in mappa.iterrows():
            centroid = row.geometry.centroid
            plt.text(centroid.x, centroid.y,
                     f"{row.reg_name}\n{row.wle:.1f}",
                     ha='center',
                     va='center',
                     fontsize=8,
                     color='white' if row.wle > 3 else 'black')

        # 8. Mostrare la mappa
        plt.tight_layout()
        plt.show()


# 1. re la geometria delle regioni italiane e riproietta in un CRS proiettato (UTM zona 32N)
url_regioni = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson"
gdf_regioni = gpd.read_file(url_regioni)
gdf_regioni = gdf_regioni.to_crs(epsg=32632)
gdf_regioni['area'] = gdf_regioni.geometry.area

# 2. Verificare dei dati (assicurati che df_unito sia già definito)
print(df_unito['wle'].isna().sum())  # Contare i NaN
print((df_unito['wle'] <= 0).sum())     # Contare i valori negativi
print(df_unito['wle'].unique())
print(df_unito['wle'].dtype)            # Verificare il tipo della colonna

print(gdf_regioni.is_valid.sum())       # Numero totale di geometrie valide
print(gdf_regioni[gdf_regioni['area'] == 0])  # Regioni con area zero
print(gdf_regioni.head())

# 3. Mappatura Cod_Reg a nomi regione
regioni_mapping = {
    1: "Valle d'Aosta/Vallée d'Aoste", 2: "Piemonte", 3: "Liguria",
    4: "Lombardia", 6: "Veneto", 7: "Friuli-Venezia Giulia",
    8: "Emilia-Romagna", 9: "Toscana", 10: "Umbria", 11: "Marche",
    12: "Lazio", 13: "Abruzzo", 14: "Molise", 15: "Campania",
    16: "Puglia", 17: "Basilicata", 18: "Calabria", 19: "Sicilia",
    20: "Sardegna", 51: "Trentino-Alto Adige/Südtirol",
    54: "Trentino-Alto Adige/Südtirol"
}
df_unito["regione"] = df_unito["cod_reg"].map(regioni_mapping)

# 4. Calcolare la media dei livelli per regione, genere e materia
df_aggregato = df_unito.groupby(["regione", "genere", "materia"])["wle"].mean().reset_index()

# 5. Unire i dati alla geometria e crea la mappa per ogni combinazione di materia e genere
for materia in df_aggregato['materia'].unique():
    for genere in [1, 2]:  # 1 = maschi, 2 = femmine
        df_genere_materia = df_aggregato[(df_aggregato["genere"] == genere) & (df_aggregato["materia"] == materia)]
        mappa = gdf_regioni.merge(df_genere_materia, left_on="reg_name", right_on="regione")

        # Creare la figura e l'asse
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot della mappa senza legenda automatica
        mappa.plot(column="wle",
                   cmap="viridis",
                   ax=ax,
                   edgecolor="white",
                   linewidth=0.5,
                   missing_kwds={"color": "lightgrey", "label": "Dato mancante"})

        plt.title(f"Livelli di Competenza in {materia} per Regione ({'Maschi' if genere == 1 else 'Femmine'})", fontsize=16, pad=20)
        ax.set_axis_off()

        # Aggiungere etichette per ogni regione
        for idx, row in mappa.iterrows():
            centroid = row.geometry.centroid
            plt.text(centroid.x, centroid.y,
                     f"{row.reg_name}\n{row.wle:.1f}",
                     ha='center',
                     va='center',
                     fontsize=8,
                     color='white' if row.wle > 3 else 'black')

        # Creare manualmente la colorbar orizzontale con il colormap viridis
        norm = plt.Normalize(vmin=mappa["wle"].min(), vmax=mappa["wle"].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm._A = []  # Necessario per ScalarMappable
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1)
        cbar.set_label(f"Livello Medio {materia} ({'Maschi' if genere == 1 else 'Femmine'})")

        plt.tight_layout()
        plt.show()
        
        
        
# 1. Caricare la mappa delle regioni italiane
url_regioni = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson"
gdf_regioni = gpd.read_file(url_regioni).to_crs(epsg=32632)
gdf_regioni['area'] = gdf_regioni.geometry.area

# 2. Mappa codice regione → nome
regioni_mapping = {
    1: "Valle d'Aosta/Vallée d'Aoste", 2: "Piemonte", 3: "Liguria",
    4: "Lombardia", 6: "Veneto", 7: "Friuli-Venezia Giulia",
    8: "Emilia-Romagna", 9: "Toscana", 10: "Umbria", 11: "Marche",
    12: "Lazio", 13: "Abruzzo", 14: "Molise", 15: "Campania",
    16: "Puglia", 17: "Basilicata", 18: "Calabria", 19: "Sicilia",
    20: "Sardegna", 51: "Trentino-Alto Adige/Südtirol", 54: "Trentino-Alto Adige/Südtirol"
}
df_unito["regione"] = df_unito["cod_reg"].map(regioni_mapping)

# 3. Calcolare media WLE globale per regione (tutte le materie e generi)
df_media_globale = df_unito.groupby("regione")["wle"].mean().reset_index()

# 4. Unire con le geometrie
mappa = gdf_regioni.merge(df_media_globale, left_on="reg_name", right_on="regione", how="left")

# 5. Plot mappa aggregata
fig, ax = plt.subplots(figsize=(14, 10))

mappa.plot(column="wle",
           cmap="viridis",
           ax=ax,
           edgecolor="white",
           linewidth=0.5,
           missing_kwds={"color": "lightgrey", "label": "Dato mancante"})

plt.title("Livello Medio INVALSI per Regione (Totale Materie e Generi)", fontsize=16, pad=20)
ax.set_axis_off()

# Etichette su ogni regione
for idx, row in mappa.iterrows():
    centroid = row.geometry.centroid
    if pd.notnull(row["wle"]):
        plt.text(centroid.x, centroid.y,
                 f"{row.reg_name}\n{row.wle:.1f}",
                 ha='center',
                 va='center',
                 fontsize=8,
                 color='white' if row.wle > 3 else 'black')

# Colorbar manuale
norm = plt.Normalize(vmin=mappa["wle"].min(), vmax=mappa["wle"].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm._A = []
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1)
cbar.set_label("Punteggio Medio WLE (Tutti gli studenti)")

plt.tight_layout()
plt.show()


# 1. Tabella di contingenza: conta il numero di osservazioni per materia, genere e area geografica
contingency_table = df_unito.groupby(['materia', 'genere', 'areageo_5']).size().reset_index(name='count')
print("Tabella di contingenza (materia, genere, areageo_5):")
print(contingency_table)

# 2. Heatmap per ciascun genere
# Creiamo una heatmap per ogni valore di 'genere' (ad esempio, 1 = maschi, 2 = femmine)
for gender in df_unito['genere'].unique():
    subset = contingency_table[contingency_table['genere'] == gender]
    pivot = subset.pivot(index='materia', columns='areageo_5', values='count').fillna(0)
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt="g")
    plt.title(f"Heatmap per genere {gender}")
    plt.xlabel("Area Geografica")
    plt.ylabel("Materia")
    plt.show()

# 3. Violin plot: distribuzione di wle per materia, suddivisa per genere e facettata per area geografica
sns.set(style="whitegrid")
g = sns.catplot(data=df_unito, x="materia", y="wle", hue="genere", col="areageo_5",
                kind="violin", palette="viridis", height=4, aspect=1, split=True)

g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Distribuzione di wle per materia (diviso per genere e area geografica)")
plt.show()

#%%
#iniziamo a produrre un po' di grafici mirati

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_unito, x='genere', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per genere')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df_unito, x='genere', y='wle', hue='materia', split=True)
plt.title('Distribuzione dei punteggi per Genere e Materia')
plt.show()



plt.figure(figsize=(10, 6))
sns.boxplot(data=df_unito, x='grado', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per grado e materia')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(data=df_unito, x='grado', y='wle', hue='materia', split=True)
plt.title('Distribuzione dei punteggi per grado e Materia')
plt.show()



sns.boxplot(data=df_unito, x='regolarità', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per regolarità e materia')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(data=df_unito, x='regolarità', y='wle', hue='materia', split=True)
plt.title('Distribuzione dei punteggi per regolarità e Materia')
plt.show()
#%%
plt.figure(figsize=(14, 8))
sns.boxplot(x='origine', y='wle', hue='materia', data=df_materie, palette="Set2")

# Aggiungere titolo e etichette
plt.title("Distribuzione dei Risultati WLE per Origine nelle Materie Matematica e Italiano", fontsize=16)
plt.xlabel('Origine')
plt.ylabel('WLE')


plt.figure(figsize=(10, 6))
sns.violinplot(data=df_unito, x='origine', y='wle', hue='materia', split=True)
plt.title('Distribuzione dei punteggi per origine e Materia')
plt.show()


# Mostrare il grafico
plt.tight_layout()
plt.show()



sns.boxplot(data=df_unito, x='areageo_5', y='wle', hue='materia')
plt.title('Distribuzione dei punteggi per area e materia')
plt.show()




plt.figure(figsize=(10, 6))
sns.violinplot(data=df_unito, x='areageo_5', y='wle', hue='materia', split=True)
plt.title('Distribuzione dei punteggi per area e Materia')
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=df_unito, x='areageo_5', y='wle', hue='genere', palette='Set2')
plt.title('Distribuzione dei punteggi per Area Geografica e Genere')
plt.show()



plt.figure(figsize=(10, 6))
sns.violinplot(data=df_unito, x='areageo_5', y='wle', hue='genere', split=True)
plt.title('Distribuzione dei punteggi per area e genere')
plt.show()



#%%

# Creazione del catplot
g = sns.catplot(data=df_unito, x='areageo_5', y='wle', hue='genere', kind='box',
                col='materia', palette='Set2', height=6, aspect=1.5)

# Aggiungere etichette per gli assi
g.set_axis_labels('Area Geografica', 'WLE')

# Modificare il titolo per includere il nome della materia in modo chiaro
g.set_titles('Distribuzione dei punteggi per Area Geografica, Genere e Materia: {col_name}')

# Aggiungere una legenda per il genere
g.add_legend(title="Genere")

# Mostrare il grafico
plt.show()
#%%



gruppi = {
    "Italiano": df_unito[df_unito["materia"] == "italiano"],
    "Matematica": df_unito[df_unito["materia"] == "matematica"],
    "Classe 2 Elementare": df_unito[df_unito["grado"] == 2],
    "Classe 5 Elementare": df_unito[df_unito["grado"] == 5],
    "Classe 3 media": df_unito[df_unito["grado"] == 8],
    "Classe 2 superiore": df_unito[df_unito["grado"] == 10],
    "Classe 5 superiore": df_unito[df_unito["grado"] == 13],
    "Classe 2 elementare - Italiano": df_unito[(df_unito["grado"] == 2) & (df_unito["materia"] == "italiano")],
    "Classe 2 elementare - Matematica": df_unito[(df_unito["grado"] == 2) & (df_unito["materia"] == "matematica")],
    "Classe 5 elementare - Italiano": df_unito[(df_unito["grado"] == 5) & (df_unito["materia"] == "italiano")],
    "Classe 5 elementare - Matematica": df_unito[(df_unito["grado"] == 5) & (df_unito["materia"] == "matematica")],
    "Classe 3 media - Italiano": df_unito[(df_unito["grado"] == 8) & (df_unito["materia"] == "italiano")],
    "Classe 3 media - Matematica": df_unito[(df_unito["grado"] == 8) & (df_unito["materia"] == "matematica")],
    "Classe 2 superiore - Italiano": df_unito[(df_unito["grado"] == 10) & (df_unito["materia"] == "italiano")],
    "Classe 2 superiore - Matematica": df_unito[(df_unito["grado"] == 10) & (df_unito["materia"] == "matematica")],
    "Classe 5 superiore - Italiano": df_unito[(df_unito["grado"] == 13) & (df_unito["materia"] == "italiano")],
    "Classe 5 superiore - Matematica": df_unito[(df_unito["grado"] == 13) & (df_unito["materia"] == "matematica")]
}

for nome, subset in gruppi.items():
    plt.figure(figsize=(8, 5))
    sns.histplot(subset["wle"], bins=30, kde=True, color="blue")
    plt.title(f"Distribuzione dei punteggi INVALSI con curva KDE - {nome}")
    plt.xlabel("Punteggio INVALSI")
    plt.ylabel("Frequenza")
    plt.show()

for nome, subset in gruppi.items():
    # Controllare se il nome del gruppo contiene "Matematica" o "Italiano"
    if "Matematica" not in nome and "Italiano" not in nome:
        plt.figure(figsize=(8, 5))
        sns.histplot(subset, x="wle", hue="materia", bins=30, kde=True, palette="Set1")

        plt.title(f"Istogramma WLE per Materia - {nome}")
        plt.xlabel("Punteggio WLE")
        plt.ylabel("Frequenza")
        plt.legend(title="Materia")
        plt.show()

    # Definiamo le combinazioni di variabili da analizzare
box_violin_plots = [
    ("genere", "regolarità"),
    ("genere", "origine"),
    ("genere", "areageo_5")
]

# Loop su ogni gruppo
for nome, subset in gruppi.items():
    for x_var, hue_var in box_violin_plots:
        plt.figure(figsize=(7, 5))

        # Boxplot
        sns.boxplot(x=x_var, y="wle", hue=hue_var, data=subset, palette="pastel")
        plt.title(f"Boxplot - {nome} ({x_var} vs {hue_var})")
        plt.xlabel(x_var)
        plt.ylabel("Punteggio WLE")
        plt.legend(title=hue_var, loc="center left", bbox_to_anchor=(1, 0.5))  # Spostare la legenda fuori a destra
        plt.tight_layout()  # Migliorare la disposizione degli elementi
        plt.show()

        # Violin Plot
        plt.figure(figsize=(7, 5))
        sns.violinplot(x=x_var, y="wle", hue=hue_var, data=subset, palette="muted", split=True)
        plt.title(f"Violin Plot - {nome} ({x_var} vs {hue_var})")
        plt.xlabel(x_var)
        plt.ylabel("Punteggio WLE")
        plt.legend(title=hue_var)
        plt.show()


for nome, subset in gruppi.items():
    plt.figure(figsize=(10, 8))

    # Calcolo della matrice di correlazione
    correlation_matrix = subset.corr(numeric_only=True)

    # Creazione della heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

    # Titolo
    plt.title(f"Heatmap delle Correlazioni - {nome}")

    # Mostrare la heatmap
    plt.show()

scatter_vars = [
    ("luogo_madre", "wle"),
    ("luogo_padre", "wle")
]

for nome, subset in gruppi.items():
    for x_var, y_var in scatter_vars:
        plt.figure(figsize=(7, 5))

        # Scatterplot
        sns.scatterplot(x=subset[x_var], y=subset[y_var], alpha=0.5)
        plt.title(f"Scatterplot {x_var} vs {y_var} - {nome}")
        plt.xlabel(x_var)
        plt.ylabel(y_var)

        # Mostrare il grafico
        plt.show()

for nome, subset in gruppi.items():
    plt.figure(figsize=(8, 5))
    sns.kdeplot(subset[subset["genere"] == 1]["wle"], label="Maschi", shade=True, color="blue")
    sns.kdeplot(subset[subset["genere"] == 2]["wle"], label="Femmine", shade=True, color="red")

    plt.title(f"Distribuzione WLE per Genere - {nome}")
    plt.xlabel("Punteggio WLE")
    plt.ylabel("Densità")
    plt.legend()
    plt.show()



for nome, subset in gruppi.items():
    if nome =="Italiano" or  nome == "Matematica": 
        model = smf.mixedlm("wle ~ genere + materia + grado", subset, groups=subset["areageo_5"]).fit()
        print(nome)
        print(model.summary())
#%%


df = df_unito[(df_unito["grado"].isin([2, 5, 8, 10, 13])) & (df_unito["materia"] == "matematica")]
# Modello misto lineare con interazione genere × grado
model = smf.mixedlm("wle ~ genere * grado", data=df, groups=df["scuola_anonimo"]).fit()



# Visualizzazione della relazione
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="grado", y="wle", hue="origine", ci=None, marker="o")
plt.title("Andamento del divario di origine in Matematica")
plt.xlabel("Grado")
plt.ylabel("Punteggio WLE Matematica")
plt.show()


# Mostrare i risultati
print(model.summary())

# Visualizzazione della relazione
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="grado", y="wle", hue="genere", ci=None, marker="o")
plt.title("Andamento del divario di genere in Matematica")
plt.xlabel("Grado")
plt.ylabel("Punteggio WLE Matematica")
plt.show()

df = df_unito[(df_unito["grado"].isin([8, 10, 13])) & (df_unito["materia"] == "matematica")]
# Modello misto lineare con interazione genere × grado
model = smf.mixedlm("wle ~ genere * grado", data=df, groups=df["scuola_anonimo"]).fit()

# Mostrare i risultati
print(model.summary())

# Visualizzazione della relazione
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="grado", y="wle", hue="genere", ci=None, marker="o")
plt.title("Andamento del divario di genere in Matematica")
plt.xlabel("Grado")
plt.ylabel("Punteggio WLE Matematica")
plt.show()





df = df_unito[(df_unito["grado"].isin([2, 5, 8, 10, 13])) & (df_unito["materia"] == "italiano")]
# Modello misto lineare con interazione genere × grado
model = smf.mixedlm("wle ~ genere * grado", data=df, groups=df["scuola_anonimo"]).fit()

# Mostrare i risultati
print(model.summary())

# Visualizzazione della relazione
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="grado", y="wle", hue="genere", ci=None, marker="o")
plt.title("Andamento del divario di genere in Italiano")
plt.xlabel("Grado")
plt.ylabel("Punteggio WLE Italiano")
plt.show()


# Visualizzazione della relazione
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="grado", y="wle", hue="origine", ci=None, marker="o")
plt.title("Andamento del divario di origine in Italiano")
plt.xlabel("Grado")
plt.ylabel("Punteggio WLE Italiano")
plt.show()

df = df_unito[(df_unito["grado"].isin([8, 10, 13])) & (df_unito["materia"] == "italiano")]
# Modello misto lineare con interazione genere × grado
model = smf.mixedlm("wle ~ genere * grado", data=df, groups=df["scuola_anonimo"]).fit()

# Mostrare i risultati
print(model.summary())

# Visualizzazione della relazione
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="grado", y="wle", hue="genere", ci=None, marker="o")
plt.title("Andamento del divario di genere in Italiano")
plt.xlabel("Grado")
plt.ylabel("Punteggio WLE Italiano")
plt.show()




# 1. Caricare la mappa delle regioni italiane (da Openpolis) e riproietta
url_regioni = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson"
gdf_regioni = gpd.read_file(url_regioni)
gdf_regioni = gdf_regioni.to_crs(epsg=32632)  # CRS metrico

# 2. Mappa codici a nomi regione
regioni_mapping = {
    1: "Valle d'Aosta/Vallée d'Aoste", 2: "Piemonte", 3: "Liguria",
    4: "Lombardia", 6: "Veneto", 7: "Friuli-Venezia Giulia",
    8: "Emilia-Romagna", 9: "Toscana", 10: "Umbria", 11: "Marche",
    12: "Lazio", 13: "Abruzzo", 14: "Molise", 15: "Campania",
    16: "Puglia", 17: "Basilicata", 18: "Calabria", 19: "Sicilia",
    20: "Sardegna", 51: "Trentino-Alto Adige/Südtirol", 54: "Trentino-Alto Adige/Südtirol"
}
df_unito["regione"] = df_unito["cod_reg"].map(regioni_mapping)

# 3. Parametri
materie = ["italiano", "matematica"]
gradi = [None, 2, 5, 8, 10, 13]  # None = tutti i gradi insieme

# 4. Calcolo e visualizzazione
for materia in materie:
    for grado in gradi:
        # Filtro dati
        if grado is None:
            df_filtered = df_unito[df_unito["materia"] == materia]
            titolo_grado = "Tutti i gradi"
            file_suffix = "tutti_gradi"
        else:
            df_filtered = df_unito[(df_unito["materia"] == materia) & (df_unito["grado"] == grado)]
            titolo_grado = f"Grado {grado}"
            file_suffix = f"grado_{grado}"

        # Media WLE per regione e genere
        df_grouped = df_filtered.groupby(["regione", "genere"])["wle"].mean().reset_index()

        # Pivot: colonne Maschi/Femmine
        df_pivot = df_grouped.pivot(index="regione", columns="genere", values="wle").reset_index()
        df_pivot.columns.name = None
        df_pivot.columns = ["regione", "Maschi", "Femmine"]
        df_pivot["Differenza"] = df_pivot["Maschi"] - df_pivot["Femmine"]

        # Merge con geometrie
        mappa = gdf_regioni.merge(df_pivot, left_on="reg_name", right_on="regione", how="left")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        mappa.plot(column="Differenza",
                   cmap="RdBu",
                   ax=ax,
                   edgecolor="white",
                   linewidth=0.6,
                   legend=True,
                   missing_kwds={"color": "lightgrey", "label": "Dato mancante"})

        ax.set_title(f"Divario di Genere INVALSI - {materia.capitalize()} ({titolo_grado})\n(Maschi - Femmine)", fontsize=15)
        ax.axis("off")

        # Etichette sui centroidi
        for idx, row in mappa.iterrows():
            if pd.notnull(row["Differenza"]):
                centroide = row.geometry.centroid
                ax.text(centroide.x, centroide.y,
                        f"{row['Differenza']:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if abs(row["Differenza"]) < 0.3 else "white")

        plt.tight_layout()
        plt.show()


print(df_unito["regione"].unique())
print(df_unito[df_unito["cod_reg"] == 1])  # cod_reg=1 dovrebbe essere Valle d'Aosta

print(df_unito["genere"].unique())


# Codice regione di interesse (Valle d'Aosta = 1)
codice_regione_target = 9

# Parametri
materie = ["italiano", "matematica"]
gradi = [2, 5, 8, 10, 13]

# Tabella risultati
risultati = []

for grado in gradi:
    riga = {"grado": grado}
    for materia in materie:
        # Filtra dati
        df_filtered = df_unito[
            (df_unito["materia"] == materia) &
            (df_unito["grado"] == grado) &
            (df_unito["cod_reg"] == codice_regione_target)
        ]

        if df_filtered.empty:
            differenza = None
        else:
            media_maschi = df_filtered[df_filtered["genere"] == 1]["wle"].mean()
            media_femmine = df_filtered[df_filtered["genere"] == 2]["wle"].mean()
            differenza = media_maschi - media_femmine

        # Salva risultato arrotondato (se non è None)
        riga[materia] = round(differenza, 2) if differenza is not None else None
    risultati.append(riga)

# Creare DataFrame finale
df_risultati = pd.DataFrame(risultati)

# Stampare tabella
print(df_risultati.to_string(index=False))

# Salvare su file CSV
df_risultati.to_csv("differenze_toscana.csv", index=False)

print("\nTabella salvata in 'differenze_toscana.csv'")



# Parametri
materie = ["italiano", "matematica"]
gradi = [2, 5, 8, 10, 13]

# Tabella risultati
risultati = []

for grado in gradi:
    riga = {"grado": grado}
    for materia in materie:
        df_filtered = df_unito[
            (df_unito["materia"] == materia) &
            (df_unito["grado"] == grado) &
            (df_unito["cod_reg"] == codice_regione_target)
        ]

        if df_filtered.empty:
            differenza = None
        else:
            media_maschi = df_filtered[df_filtered["genere"] == "M"]["wle"].mean()
            media_femmine = df_filtered[df_filtered["genere"] == "F"]["wle"].mean()
            differenza = media_maschi - media_femmine

        riga[materia] = round(differenza, 2) if differenza is not None else None
    risultati.append(riga)

# Creare DataFrame finale
df_risultati = pd.DataFrame(risultati)
print(df_risultati.to_string(index=False))
df_risultati.to_csv("differenze_valle_aosta.csv", index=False)
print("\nTabella salvata in 'differenze_valle_aosta.csv'")

# --- Nuova Parte: Grafico andamento per aree geografiche ---

# Mappatura macroaree
area_mapping = {
    "Nord": "Nord",
    "Nordest": "Nord",
    "Centro": "Centro",
    "Sud": "Sud",
    "Sud e Isole": "Sud"
}

df_unito["area_geo"] = df_unito["areageo_5"].map(area_mapping)

# Funzione per grafico
def plot_macroarea_wle(materia: str):
    df_plot = df_unito[df_unito["materia"] == materia]

    df_grouped = df_plot.groupby(["area_geo", "genere"])['wle'].mean().reset_index()

    genere_mapping = {1: "Maschi", 2: "Femmine"}
    df_grouped["genere"] = df_grouped["genere"].map(genere_mapping)

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=df_grouped,
        x="area_geo", y="wle", hue="genere",
        palette="Set1"
    )
    plt.title(f"Andamento WLE - {materia.capitalize()} per macroarea", fontsize=14)
    plt.xlabel("Macroarea")
    plt.ylabel("Media WLE")
    plt.legend(title="Genere")
    plt.tight_layout()
    plt.show()

# Grafici
plot_macroarea_wle("matematica")
plot_macroarea_wle("italiano")





def plot_macroarea_wle(materia: str):
    df_plot = df_unito[df_unito["materia"] == materia]

    df_grouped = df_plot.groupby(["area_geo", "genere"])['wle'].mean().reset_index()

    genere_mapping = {1: "Maschi", 2: "Femmine"}
    df_grouped["genere"] = df_grouped["genere"].map(genere_mapping)

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=df_grouped,
        x="area_geo", y="wle", hue="genere",
        palette="Set1"
    )
    plt.title(f"Andamento WLE - {materia.capitalize()} per macroarea", fontsize=14)
    plt.xlabel("Macroarea")
    plt.ylabel("Media WLE")
    plt.legend(title="Genere")
    plt.tight_layout()
    plt.show()

# Funzione per grafico differenze Maschi-Femmine

def plot_macroarea_diff(materia: str):
    df_plot = df_unito[df_unito["materia"] == materia]
    df_grouped = df_plot.groupby(["area_geo", "genere"])['wle'].mean().unstack()

    df_grouped["Differenza"] = df_grouped[1] - df_grouped[2]
    df_diff = df_grouped[["Differenza"].copy()]

    # 1. Barplot semplice
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=df_diff.reset_index(),
        x="area_geo", y="Differenza",
        palette="coolwarm"
    )
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Differenza (Maschi-Femmine) WLE - {materia.capitalize()} per macroarea", fontsize=14)
    plt.xlabel("Macroarea")
    plt.ylabel("Differenza WLE")
    plt.tight_layout()
    plt.show()

    # 2. Pointplot
    plt.figure(figsize=(10,6))
    sns.pointplot(
        data=df_diff.reset_index(),
        x="area_geo", y="Differenza",
        color='darkblue', markers='o'
    )
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"Differenza (Maschi-Femmine) WLE - {materia.capitalize()} (Pointplot)", fontsize=14)
    plt.xlabel("Macroarea")
    plt.ylabel("Differenza WLE")
    plt.tight_layout()
    plt.show()

    # 3. Horizontal barplot
    plt.figure(figsize=(8,6))
    sns.barplot(
        data=df_diff.reset_index(),
        y="area_geo", x="Differenza",
        palette="RdBu"
    )
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Differenza (Maschi-Femmine) WLE - {materia.capitalize()} (Horizontal)", fontsize=14)
    plt.xlabel("Differenza WLE")
    plt.ylabel("Macroarea")
    plt.tight_layout()
    plt.show()

# Grafici andamento WLE
plot_macroarea_wle("matematica")
plot_macroarea_wle("italiano")

# Grafici differenze Maschi-Femmine
plot_macroarea_diff("matematica")
plot_macroarea_diff("italiano")



print(df_unito["origine"].unique())
print(df_unito["origine"].value_counts(dropna=False))


# DIZIONARI aggiornati
mapping_variabili = {
    "genere": {1: "Maschi", 2: "Femmine"},
    "origine": {
        "Nativo": "Nativo",
        "Straniero I generazione": "Straniero I generazione",
        "Straniero II generazione": "Straniero II generazione"
    },
    "areageo_5": {
        "Nordovest": "Nord", "Nordest": "Nord",
        "Centro": "Centro",
        "Sud": "Sud", "Sud e Isole": "Sud"
    }
}





# 1. Caricare la mappa delle regioni
url_regioni = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson"
gdf_regioni = gpd.read_file(url_regioni)
gdf_regioni = gdf_regioni.to_crs(epsg=32632)

# 2. Mappa codici regione
regioni_mapping = {
    1: "Valle d'Aosta/Vallée d'Aoste", 2: "Piemonte", 3: "Liguria",
    4: "Lombardia", 6: "Veneto", 7: "Friuli-Venezia Giulia",
    8: "Emilia-Romagna", 9: "Toscana", 10: "Umbria", 11: "Marche",
    12: "Lazio", 13: "Abruzzo", 14: "Molise", 15: "Campania",
    16: "Puglia", 17: "Basilicata", 18: "Calabria", 19: "Sicilia",
    20: "Sardegna", 51: "Trentino-Alto Adige/Südtirol", 54: "Trentino-Alto Adige/Südtirol"
}

df_unito["regione"] = df_unito["cod_reg"].map(regioni_mapping)

# 3. Parametri
materie = ["italiano", "matematica"]

# 4. Calcolo e visualizzazione
for materia in materie:
    df_filtered = df_unito[df_unito["materia"] == materia]
    
    # Media WLE per regione
    df_grouped = df_filtered.groupby("regione")["wle"].mean().reset_index()
    
    # Merge con geometrie
    mappa = gdf_regioni.merge(df_grouped, left_on="reg_name", right_on="regione", how="left")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    mappa.plot(column="wle",
               cmap="YlGnBu",
               ax=ax,
               edgecolor="white",
               linewidth=0.6,
               legend=True,
               missing_kwds={"color": "lightgrey", "label": "Dato mancante"})

    ax.set_title(f"Punteggio Medio INVALSI - {materia.capitalize()}", fontsize=15)
    ax.axis("off")

    # Etichette sui centroidi
    for idx, row in mappa.iterrows():
        if pd.notnull(row["wle"]):
            centroide = row.geometry.centroid
            ax.text(centroide.x, centroide.y,
                    f"{row['wle']:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="black")

    plt.tight_layout()
    plt.show()



#DOPO QUESTA PLETORA DI VISUALIZZAZIONI E GRAFICI VARI, POSSIAMO PASSARE AI TEST VERI E PROPRI

# Funzione: calcolo Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*x.var() + (ny - 1)*y.var()) / dof)
    return (x.mean() - y.mean()) / pooled_std if pooled_std != 0 else np.nan

# FUNZIONE TEST
def esegui_test(df, gruppo_descrizione, variabile):
    risultati = []
    df = df.copy()

    if variabile == "areageo_5":
        df[variabile] = df[variabile].replace(mapping_variabili[variabile])
        valori_validi = list(mapping_variabili[variabile].values())
    else:
        valori_validi = list(mapping_variabili[variabile].keys())

    df = df[df[variabile].isin(valori_validi)]
    df = df.dropna(subset=["wle", variabile])

    categorie = df[variabile].unique()

    if variabile == "origine":
        print(f"⚠️  Origine - {gruppo_descrizione}: {df[variabile].value_counts()}")

    if len(categorie) < 2:
        return []

    # T-test
    if len(categorie) == 2:
        cat1, cat2 = categorie
        g1 = df[df[variabile] == cat1]["wle"]
        g2 = df[df[variabile] == cat2]["wle"]
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        d_value = cohens_d(g1, g2)

        risultati.append({
            "Gruppo": gruppo_descrizione,
            "Variabile": variabile,
            "Test": "T-test",
            "Categoria 1": cat1,
            "Categoria 2": cat2,
            "Media 1": round(g1.mean(), 3),
            "Media 2": round(g2.mean(), 3),
            "Varianza 1": round(g1.var(), 3),
            "Varianza 2": round(g2.var(), 3),
            "Statistica": round(t_stat, 4),
            "P-value (normale)": f"{p_val:.5f}",
            "P-value (scientifico)": f"{p_val:.4e}",
            "Significativo": "Sì" if p_val < 0.05 else "No",
            "Vantaggio": cat1 if g1.mean() > g2.mean() else cat2,
            "Cohen d": round(d_value, 4)
        })

    # ANOVA
    elif len(categorie) >= 3:
        gruppi = [df[df[variabile] == cat]["wle"] for cat in categorie]
        f_stat, p_val = stats.f_oneway(*gruppi)

        medie = {cat: df[df[variabile] == cat]["wle"].mean() for cat in categorie}
        max_cat = max(medie, key=medie.get)

        # Calcolo Cohen's d tra il gruppo "vantaggio" e gli altri uniti
        g1 = df[df[variabile] == max_cat]["wle"]
        g2 = df[df[variabile] != max_cat]["wle"]
        d_value = cohens_d(g1, g2)

        risultati.append({
            "Gruppo": gruppo_descrizione,
            "Variabile": variabile,
            "Test": "ANOVA",
            "Categoria 1": "-",
            "Categoria 2": "-",
            "Media 1": "-",
            "Media 2": "-",
            "Varianza 1": "-",
            "Varianza 2": "-",
            "Statistica": round(f_stat, 4),
            "P-value (normale)": f"{p_val:.5f}",
            "P-value (scientifico)": f"{p_val:.4e}",
            "Significativo": "Sì" if p_val < 0.05 else "No",
            "Vantaggio": max_cat,
            "Cohen d": round(d_value, 4)
            })

    return risultati


# AGGREGA RISULTATI
tutti_risultati = []

# 1. Tutto il dataset
for var in mapping_variabili:
    tutti_risultati += esegui_test(df_unito, "Tutti i dati", var)

# 2. Per grado
for grado in sorted(df_unito["grado"].dropna().unique()):
    df_grado = df_unito[df_unito["grado"] == grado]
    for var in mapping_variabili:
        tutti_risultati += esegui_test(df_grado, f"Grado {grado}", var)

# 3. Per materia
for materia in df_unito["materia"].dropna().unique():
    df_materia = df_unito[df_unito["materia"] == materia]
    for var in mapping_variabili:
        tutti_risultati += esegui_test(df_materia, f"Materia: {materia}", var)

# 4. Per materia + grado
for materia in df_unito["materia"].dropna().unique():
    for grado in sorted(df_unito["grado"].dropna().unique()):
        df_combo = df_unito[(df_unito["materia"] == materia) & (df_unito["grado"] == grado)]
        if df_combo.shape[0] < 10:
            continue
        for var in mapping_variabili:
            tutti_risultati += esegui_test(df_combo, f"{materia} - Grado {grado}", var)


# 5. Per genere
for genere in df_unito["genere"].dropna().unique():
    df_genere = df_unito[df_unito["genere"] == genere]
    for var in mapping_variabili:
        tutti_risultati += esegui_test(df_genere, f"Genere: {genere}", var)
        
        
# 6. Per materia + genere 
for materia in df_unito["materia"].dropna().unique():
    for genere in sorted(df_unito["genere"].dropna().unique()):
        df_combo = df_unito[(df_unito["materia"] == materia) & (df_unito["genere"] == genere)]
        if df_combo.shape[0] < 10:
            continue
        for var in mapping_variabili:
            tutti_risultati += esegui_test(df_combo, f"{materia} - Genere {genere}", var)


# SALVA CSV (UTF-8 con BOM per Excel)
df_risultati = pd.DataFrame(tutti_risultati)
df_risultati.to_csv("risultati_test_statistici_completi_utf8.csv", index=False, encoding="utf-8-sig")
print("✅ File salvato: risultati_test_statistici_completi_utf8.csv")



#PROVIAMO AD APPROFONDIRE LA QUESTIONE SUPERIORI VISTI GLI ESITI DELLE FASI PRECEDENTI


df_2sup = pd.read_csv("2024_Matrice_Campione_MAT_10_0_1_WLE_anonima.csv", sep=";", encoding="latin1", header=0)
df_5sup = pd.read_csv("2024_Matrice_Campione_MAT_13_0_1_WLE_anonima.csv", sep=",", encoding="latin1", header=0)
# Rinominare colonne con uppercase temporaneo per confronto
df_2sup.columns = df_2sup.columns.str.upper()
df_5sup.columns = df_5sup.columns.str.upper()

# Trovare colonne comuni
colonne_comuni = df_2sup.columns.intersection(df_5sup.columns)

# Selezionare solo le colonne comuni dai DataFrame originali
df_concat = pd.concat([df_2sup[colonne_comuni], df_5sup[colonne_comuni]], ignore_index=True)

print("Colonne iniziali:", df_concat.columns.tolist())


# Step 4: Pulizia stringhe (doppi apici e numeri con virgola)
for col in df_concat.columns:
    if df_concat[col].dtype == "object":
        df_concat[col] = df_concat[col].astype(str).str.replace('"', '')
        df_concat[col] = df_concat[col].str.replace(r'(\d+),(\d+)', r'\1.\2', regex=True)

# Step 5: Aggiunta colonna materia e rinomina colonna wle

df_concat['MATERIA'] = 'matematica'
df_concat = df_concat.rename(columns={
    'WLE_MAT_200': 'wle'  # Solo se esiste
})

# Step 6: Conversioni tipo e pulizie valori sentinella
colonne_intere = [
   "SCUOLA_ANONIMO", "CLASSE_ANONIMO", "STUDENTE_ANONIMO", "GRADO", "GENERE",
    "MESE", "ANNO", "LUOGO", "REGOLARITÀ", "ORIGINE", "CODICE_ORARIO",
    "LUOGO_PADRE", "LUOGO_MADRE", "COD_REG"
]
colonne_float = [ "wle", "PESO_STUDENTE", "PESO_CLASSE", "PESO_SCUOLA"]

for col in colonne_intere:
    if col in df_concat.columns:
        df_concat[col] = pd.to_numeric(df_concat[col], errors='coerce').astype('Int64')

for col in colonne_float:
    if col in df_concat.columns:
        df_concat[col] = pd.to_numeric(df_concat[col], errors='coerce').astype(float)

# Step 7: Rimozione righe non valide
if 'genere' in df_concat:
    df_concat = df_concat.dropna(subset=['GENERE'])
    df_concat = df_concat[df_concat['GENERE'] != 9]

if 'grado' in df_concat:
    valori_ammessi_grado = {2, 5, 8, 10, 13}
    df_concat = df_concat[df_concat['GRADO'].isin(valori_ammessi_grado)]

if 'luogo' in df_concat:
    df_concat = df_concat[df_concat['LUOGO'] != 9]

if 'mese' in df_concat:
    df_concat = df_concat[df_concat['MESE'] != 99]

if 'anno' in df_concat:
    df_concat = df_concat[df_concat['ANNO'] != 9]

if 'regolarità' in df_concat:
    df_concat = df_concat[df_concat['REGOLARITÀ'] != 9]
    df_concat = df_concat.dropna(subset=['REGOLARITÀ'])

if 'origine' in df_concat:
    df_concat = df_concat[df_concat['ORIGINE'] != 9]
    df_concat = df_concat.dropna(subset=['ORIGINE'])
    df_concat['origine'] = df_concat['ORIGINE'].map({
        1: 'Nativo',
        2: 'Straniero I generazione',
        3: 'Straniero II generazione'
    })

if 'luogo_padre' in df_concat:
    df_concat = df_concat[df_concat['LUOGO_PADRE'] != 9]

if 'luogo_madre' in df_concat:
    df_concat = df_concat[df_concat['LUOGO_MADRE'] != 9]

if 'areageo_5' in df_concat:
    df_concat['AREAGEO_5'] = df_concat['AREAGEO_5'].map({1: 'Nord', 2: 'Nord', 3: 'Centro', 4: 'Sud', 5: 'Sud'})

# Step 8: Heatmap delle correlazioni
corrmat = df_concat.select_dtypes(include=['number']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corrmat, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap delle Correlazioni")
plt.show()

print("Colonne finali:", df_concat.columns.tolist())


# Salvare risultato finale
print(f"Numero di righe dopo la pulizia: {df_concat.shape[0]}")
df_concat.to_csv("dataset_mate_pulito.csv", index=False)

df = df_concat


#analisi tipo scuola
media_punteggi = df.groupby(["TIPO_SCUOLA", "GENERE"])["wle"].mean().unstack()


# Mostrare i risultati ordinati per punteggio medio più alto (maschi)
media_punteggi = media_punteggi.sort_values(by=1, ascending=False)  
print(media_punteggi)


# Conteggio studentesse per tipo di scuola
conteggio_femmine = df[df["GENERE"] == 2].groupby("TIPO_SCUOLA")["GENERE"].count()

# Conteggio totale studenti per tipo di scuola
totale_studenti = df.groupby("TIPO_SCUOLA")["GENERE"].count()

# Percentuale di femmine per tipo di scuola
percentuale_femmine = (conteggio_femmine / totale_studenti) * 100

# Ordinare i dati per i punteggi medi delle scuole
percentuale_femmine = percentuale_femmine.sort_index()
print(percentuale_femmine)




# Mappatura dei tipi di scuola
scuole = {
    1: "Licei Scientifici",
    2: "Altri Licei",
    3: "Tecnici",
    4: "Professionali"
}
df = df[df["TIPO_SCUOLA"].isin(scuole.keys())]
df["TIPO_SCUOLA_LABEL"] = df["TIPO_SCUOLA"].map(scuole)

# Converti il genere in stringa (1=maschi, 2=femmine)
df["genere_str"] = df["GENERE"].map({1: "Maschi", 2: "Femmine"})

# -------------------------------------------------------
# 1. Test t di Student per ciascun tipo di scuola
# -------------------------------------------------------
print("Test t di Student tra maschi e femmine per ogni tipo di scuola:\n")

for codice, nome in scuole.items():
    sotto_df = df[df["TIPO_SCUOLA"] == codice]
    maschi = sotto_df[sotto_df["GENERE"] == 1]["wle"]
    femmine = sotto_df[sotto_df["GENERE"] == 2]["wle"]
    stat, p = ttest_ind(maschi, femmine, equal_var=False)  # Welch's t-test
    print(f"{nome} → t = {stat:.3f}, p = {p:.4g} ({'significativo' if p < 0.05 else 'non significativo'})")

# -------------------------------------------------------
# 2. Test del chi-quadrato per la distribuzione genere/tipo scuola
# -------------------------------------------------------
print("\nTest del chi-quadrato per genere e tipo scuola:\n")

# Costruzione tabella di contingenza
contingenza = pd.crosstab(df["TIPO_SCUOLA_LABEL"], df["genere_str"])
chi2, p_chi2, dof, expected = chi2_contingency(contingenza)
print(f"Chi² = {chi2:.2f}, p = {p_chi2:.4g}, DoF = {dof}")
print("Distribuzione genere ≠ indipendente dal tipo di scuola" if p_chi2 < 0.05 else "Distribuzione genere ≈ indipendente dal tipo di scuola")

# -------------------------------------------------------
# 3. Grafico a barre dei punteggi medi per genere e tipo scuola
# -------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="TIPO_SCUOLA_LABEL", y="wle", hue="genere_str", ci="sd")
plt.title("Punteggi medi INVALSI di matematica per genere e tipo di scuola")
plt.ylabel("Punteggio WLE")
plt.xlabel("Tipo di scuola")
plt.legend(title="Genere")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 4. Grafico della percentuale di femmine per tipo di scuola
# -------------------------------------------------------
percentuali = df.groupby("TIPO_SCUOLA_LABEL")["GENERE"].apply(lambda x: (x == 2).mean() * 100)

plt.figure(figsize=(8, 5))
sns.barplot(x=percentuali.index, y=percentuali.values, palette="pastel")
plt.ylabel("Percentuale di femmine")
plt.title("Percentuale di femmine per tipo di scuola")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()


# Funzione per fare un test t maschi vs femmine per ciascun tipo scuola e grado

def t_test_by_school_and_grade(df, colonna):
    risultati = []
    for grado in [10, 13]:
        print(f"\n=== Test T per grado {grado} - {colonna} ===")
        for scuola in sorted(df['TIPO_SCUOLA_LABEL'].dropna().unique()):
            subset = df[(df['TIPO_SCUOLA_LABEL'] == scuola) & (df['GRADO'] == grado)]
            maschi = subset[subset['GENERE'] == 1][colonna]
            femmine = subset[subset['GENERE'] == 2][colonna]
            t_stat, p_val = stats.ttest_ind(maschi, femmine, equal_var=False, nan_policy='omit')
            print(f"Tipo scuola {scuola}: t = {t_stat:.3f}, p = {p_val:.4f}")
            risultati.append((grado, scuola, colonna, t_stat, p_val))
    return risultati

# Funzione per test chi-quadro tra genere e tipo scuola (numeri assoluti)
def chi2_test(df):
    tabella = pd.crosstab(df['TIPO_SCUOLA_LABEL'], df['GENERE'])
    chi2, p, dof, expected = chi2_contingency(tabella)
    print("\n=== Test Chi-quadro su genere e tipo scuola ===")
    print("Chi2 = {:.3f}, p = {:.4f}".format(chi2, p))
    print("Tabella osservata:\n", tabella)
    return chi2, p

# Grafico: numero assoluto di maschi e femmine per tipo scuola e grado
def plot_absolute_counts(df):
    plt.figure(figsize=(12, 6))
    counts = df.groupby(['GRADO', 'TIPO_SCUOLA_LABEL', 'GENERE']).size().reset_index(name='conteggio')

    counts['GENERE'] = counts['GENERE'].map({1: 'Maschi', 2: 'Femmine'})

    sns.barplot(data=counts, x='TIPO_SCUOLA_LABEL', y='conteggio', hue='GENERE', ci=None, palette='Set2', dodge=True)
    plt.title('Numero di maschi e femmine per tipo di scuola')
    plt.xlabel('Tipo di scuola')
    plt.ylabel('Numero di studenti')
    plt.legend(title='Genere')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Boxplot confronto per genere e tipo scuola
def confronto_genere_boxplot(df, colonna):
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='TIPO_SCUOLA_LABEL', y=colonna, hue='genere_str')
    plt.title(f'Distribuzione {colonna} per genere e tipo di scuola')
    plt.ylabel(colonna)
    plt.xlabel('Tipo di scuola')
    plt.legend(title='Genere')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Risultati Test T (Welch) per ciascun tipo di scuola - {colonna}:")
    for label in df['TIPO_SCUOLA_LABEL'].unique():
        subset = df[df['TIPO_SCUOLA_LABEL'] == label]
        m = subset[subset['GENERE'] == 1][colonna]
        f = subset[subset['GENERE'] == 2][colonna]
        t_stat, p_val = ttest_ind(m, f, equal_var=False)
        print(f"{label}: t = {t_stat:.2f}, p = {p_val:.4f} → {'significativo' if p_val < 0.05 else 'non significativo'}")

# Grafico media per grado e tipo di scuola
def plot_media_grado_scuola(df, colonna):
    plt.figure(figsize=(16, 6))
    sns.barplot(
        data=df,
        x="TIPO_SCUOLA_LABEL",
        y=colonna,
        hue="genere_str",
        ci="sd",
        palette="Set2",
        order=["Licei Scientifici", "Altri Licei", "Tecnici", "Professionali"]
    )
    plt.title(f"{colonna}: punteggi medi per Genere, Tipo di Scuola e Grado")
    plt.xlabel("Tipo di scuola")
    plt.ylabel(colonna)
    plt.legend(title="Genere")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# T-test per ciascun grado e tipo scuola
def t_test_grado_tipo(df, colonna):
    risultati = []
    for grado in sorted(df['GRADO'].dropna().unique()):
        print(f"\n=== Grado {grado} - {colonna} ===")
        for scuola in df['TIPO_SCUOLA_LABEL'].unique():
            subset = df[(df['GRADO'] == grado) & (df['TIPO_SCUOLA_LABEL'] == scuola)]
            m = subset[subset['GENERE'] == 1][colonna]
            f = subset[subset['GENERE'] == 2][colonna]
            if len(m) > 5 and len(f) > 5:
                t_stat, p_val = ttest_ind(m, f, equal_var=False)
                print(f"{scuola}: t = {t_stat:.2f}, p = {p_val:.4f}")
                risultati.append((grado, scuola, colonna, t_stat, p_val))
    return risultati

# Heatmap differenze medie tra maschi e femmine
def heatmap_diff_medie(df, colonna):
    pivot_df = df.pivot_table(index='TIPO_SCUOLA_LABEL', 
                               columns='GRADO', 
                               values=colonna, 
                               aggfunc=lambda x: x[df.loc[x.index, 'GENERE'] == 1].mean() - x[df.loc[x.index, 'GENERE'] == 2].mean())

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(f"Differenza media {colonna} (Maschi - Femmine)\nper Tipo di Scuola e Grado")
    plt.ylabel("Tipo di scuola")
    plt.xlabel("Grado")
    plt.tight_layout()
    plt.show()

# Lineplot media per grado
def lineplot_media_per_grado(df, colonna):
    media_grado = df.groupby(['TIPO_SCUOLA_LABEL', 'GRADO', 'genere_str'])[colonna].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=media_grado, x='GRADO', y=colonna, hue='genere_str', style='TIPO_SCUOLA_LABEL', markers=True, dashes=False)
    plt.title(f'Andamento medio di {colonna} per genere e tipo di scuola (Grado 10 → 13)')
    plt.xlabel('Grado scolastico')
    plt.ylabel(f'{colonna} medio')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Genere / Tipo Scuola', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Barplot aggregato per tipo scuola
def barplot_per_grado(df, colonna):
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=df,
        x='TIPO_SCUOLA_LABEL',
        y=colonna,
        hue='genere_str',
        ci='sd',
        palette='muted',
        order=["Licei Scientifici", "Altri Licei", "Tecnici", "Professionali"]
    )
    plt.title(f"{colonna}: punteggi medi per tipo scuola e genere (Grado aggregato)")
    plt.xlabel("Tipo di scuola")
    plt.ylabel(colonna)
    plt.legend(title="Genere")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Catplot per colonna
def catplot_grado_tipo(df, colonna):
    g = sns.catplot(
        data=df,
        x='genere_str',
        y=colonna,
        hue='genere_str',
        col='TIPO_SCUOLA_LABEL',
        row='GRADO',
        kind='bar',
        ci='sd',
        palette='pastel',
        height=4,
        aspect=1
    )
    g.set_axis_labels("Genere", f"{colonna} medio")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{colonna}: punteggi medi per genere, grado e tipo di scuola")
    plt.show()


# Esecuzione delle funzioni per ciascuna colonna
for col in ['wle', 'VOTO_SCRITTO_MAT', 'VOTO_ORALE_MAT']:
    t_test_by_school_and_grade(df, col)
    confronto_genere_boxplot(df, col)
    plot_media_grado_scuola(df, col)
    t_test_grado_tipo(df, col)
    heatmap_diff_medie(df, col)
    lineplot_media_per_grado(df, col)
    barplot_per_grado(df, col)
    catplot_grado_tipo(df, col)

chi2_test(df)
plot_absolute_counts(df)

# KDE plot (solo per wle)
sns.kdeplot(data=df[(df['GENERE'] == 1) & (df['GRADO'] == 10)], x='wle', label='Maschi G10', linestyle='--')
sns.kdeplot(data=df[(df['GENERE'] == 2) & (df['GRADO'] == 10)], x='wle', label='Femmine G10')
plt.title("Distribuzione dei punteggi WLE - Grado 10")
plt.legend()
plt.grid(True)
plt.show()

sns.kdeplot(data=df[(df['GENERE'] == 1) & (df['GRADO'] == 13)], x='wle', label='Maschi G13', linestyle='--')
sns.kdeplot(data=df[(df['GENERE'] == 2) & (df['GRADO'] == 13)], x='wle', label='Femmine G13')
plt.title("Distribuzione dei punteggi WLE - Grado 13")
plt.legend()
plt.grid(True)
plt.show()

# Boxplot ESCS




#-----------------------------------------------------
df = df[~df["GENERE"].isin(["9"]) & ~df["ORIGINE"].isin(["9"])]

df["GENERE"] = df["GENERE"].astype(str)

df["ORIGINE"] = df["ORIGINE"].astype(str)


# Convertire la colonna in numeri (forzando la conversione degli errori in NaN)
df['ESCS_STUDENTE'] = pd.to_numeric(df['ESCS_STUDENTE'], errors='coerce')

# Rimuovere eventuali righe con valori NaN (se ce ne sono)
df = df.dropna(subset=['ESCS_STUDENTE'])

# calcolare i quartili e continuare con la segmentazione
q1 = df['ESCS_STUDENTE'].quantile(0.25)  # primo quartile (25%)
q3 = df['ESCS_STUDENTE'].quantile(0.75)  # terzo quartile (75%)
median = df['ESCS_STUDENTE'].median()    # mediana (50%)

# Definire le etichette per le categorie
conditions = [
    (df['ESCS_STUDENTE'] <= q1),            # Basso: valori fino al primo quartile
    (df['ESCS_STUDENTE'] > q1) & (df['ESCS_STUDENTE'] <= median),  # Medio: tra il primo quartile e la mediana
    (df['ESCS_STUDENTE'] > median)          # Alto: valori sopra la mediana
]

# Creare una nuova colonna 'Categoria' con le etichette
labels = ['Basso', 'Medio', 'Alto']
df['Categoria'] = pd.cut(df['ESCS_STUDENTE'], bins=[-float('inf'), q1, median, float('inf')], labels=labels)

# Visualizzare i primi 5 valori per conferma
print(df[['ESCS_STUDENTE', 'Categoria']].head())


# Formula del modello misto lineare
formula = "wle ~ C(AREAGEO_5) + C(TIPO_SCUOLA_LABEL) + C(GENERE) + C(ORIGINE) + C(Categoria)"


# 🔹 Modello su tutto il dataset
print("🔍 Mixed Model su tutto il dataset (Gradi 10 + 13)")
model_all = smf.mixedlm(formula, data=df, groups=df["SCUOLA_ANONIMO"])
result_all = model_all.fit()
print(result_all.summary())

# 🔹 Modello solo per grado 10
df_10 = df[df["GRADO"] == 10]
print("\n🔍 Mixed Model - Grado 10")
model_10 = smf.mixedlm(formula, data=df_10, groups=df_10["SCUOLA_ANONIMO"])
result_10 = model_10.fit()
print(result_10.summary())

# 🔹 Modello solo per grado 13
df_13 = df[df["GRADO"] == 13]
print("\n🔍 Mixed Model - Grado 13")
model_13 = smf.mixedlm(formula, data=df_13, groups=df_13["SCUOLA_ANONIMO"])
result_13 = model_13.fit()
print(result_13.summary())

def plot_coefficients_fixed_effects(result, titolo="Effetti Fissi del Modello"):
    coefs = result.fe_params.drop("Intercept")  # rimuove Intercept
    conf = result.conf_int().loc[coefs.index]

    # Mapping leggibili
    var_mapping = {
        "C(AREAGEO_5)[T.2]": "Nord-Est",
        "C(AREAGEO_5)[T.3]": "Centro",
        "C(AREAGEO_5)[T.4]": "Sud",
        "C(AREAGEO_5)[T.5]": "Sud e Isole",
        "C(TIPO_SCUOLA_LABEL)[T.Licei Scientifici]": "Licei Scientifici",
        "C(TIPO_SCUOLA_LABEL)[T.Professionali]": "Professionali",
        "C(TIPO_SCUOLA_LABEL)[T.Tecnici]": "Tecnici",
        "C(GENERE)[T.2]": "Femmine",
        "C(ORIGINE)[T.2]": "Straniero I gen.",
        "C(ORIGINE)[T.3]": "Straniero II gen.",
        "C(Categoria)[T.Medio]": "ESCS_Medio",
        "C(Categoria)[T.Alto]": "ESCS_alto"
        
    }

    ordine_blocchi = [
        "Straniero I gen.",
        "Straniero II gen.",
        "Nord-Est",
        "Centro",
        "Sud",
        "Sud e Isole",
        "Femmine",
        "Licei Scientifici",
        "Professionali",
        "Tecnici",
        "ESCS_Medio",
        "ESCS_alto"
    ]

    df_plot = pd.DataFrame({
        "Variable": [var_mapping.get(k, k) for k in coefs.index],
        "Coef": coefs.values,
        "CI_lower": conf[0].values,
        "CI_upper": conf[1].values
    })

    df_plot["Variable"] = pd.Categorical(df_plot["Variable"], categories=ordine_blocchi, ordered=True)
    df_plot = df_plot.sort_values("Variable", ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_plot, y="Variable", x="Coef", color="skyblue", edgecolor="black")

    # Intervalli di confidenza
    for i, row in df_plot.iterrows():
        plt.plot([row["CI_lower"], row["CI_upper"]], [i, i], color="black", lw=1.5)

    plt.axvline(0, color="red", linestyle="--", lw=1)

    # Valori base espliciti (dal modello)
    base_val = result.fe_params["Intercept"]
    plt.title(f"{titolo}\nValore base = {base_val:.2f} (Nord, Altri Licei, Maschi, Nativi, ESCS_Basso)", fontsize=13)
    plt.xlabel("Effetto stimato sul punteggio WLE")
    plt.ylabel("")
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.xlim(-40, 40)  # o l’intervallo massimo comune tra i modelli

    plt.tight_layout()
    plt.show()


# ➤ Applicazione
plot_coefficients_fixed_effects(result_all, "Effetti fissi - Modello Gradi 10+13")
plot_coefficients_fixed_effects(result_10, "Effetti fissi - Modello Grado 10")
plot_coefficients_fixed_effects(result_13, "Effetti fissi - Modello Grado 13")







