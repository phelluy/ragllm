import pandas as pd
import sqlite3
import io

# 1. Les données brutes (Format CSV)
csv_data = """client_prenom,client_nom,client_profession,animal_race,animal_nom,animal_age
Claire, Martin, enseignant, lapin, Panpan, 5
Jean, Dupont, plombier, chien, Rex, 10
Matthieu, Helluy, étudiant, chat, Miaou, 3
Elsa, Pinto, étudiant, poisson, Bubulle, 1
Charles, Dufour, dentiste, cheval, Diamant,10
Charles, Dufour, dentiste, cheval, Opale,5
Charles, Dufour, dentiste, chien, Didier,4
Hubert,Vadombreuse,vagabond,chien,Platon, 10
Huguette,Laplante,retraité,chat,Bouzouk,7
Huguette,Laplante,retraité,chat,Bachi,7
Sophie,Laplante,écolier,écureuil,Goldorak,1
Vlad, Zorg,musicien,perroquet,Laplume,20"""

# Chargement dans un DataFrame pandas
# skipinitialspace=True permet de gérer les espaces après les virgules
df = pd.read_csv(io.StringIO(csv_data), skipinitialspace=True)

# --- ÉTAPE 1 : Préparer la table CLIENT ---
# On sélectionne les colonnes clients et on supprime les doublons
# (Charles Dufour ne doit apparaître qu'une fois dans la table client)
df_clients = df[['client_prenom', 'client_nom', 'client_profession']].drop_duplicates()

# On réinitialise l'index pour créer un ID unique (id_client)
df_clients = df_clients.reset_index(drop=True)
df_clients['id'] = df_clients.index + 1 # On commence les ID à 1

# On renomme les colonnes pour correspondre à votre schéma cible
table_client = df_clients.rename(columns={
    'client_prenom': 'prenom',
    'client_nom': 'nom',
    'client_profession': 'profession'
})[['id', 'prenom', 'nom', 'profession']]

# --- ÉTAPE 2 : Préparer la table ANIMAL ---
# Chaque ligne du CSV d'origine correspond à un animal unique
df_animaux = df[['animal_nom', 'animal_race', 'animal_age']].copy()
df_animaux['id'] = df_animaux.index + 1 # ID unique pour l'animal

table_animal = df_animaux.rename(columns={
    'animal_nom': 'nom',
    'animal_race': 'race',
    'animal_age': 'age'
})[['id', 'nom', 'race', 'age']]

# --- ÉTAPE 3 : Préparer la table MAITRE (Liaison) ---
# Il faut relier l'ID de l'animal à l'ID du client.
# On refait une fusion (merge) entre le dataframe original et notre table client propre
df_merge = pd.merge(df, table_client,
                    left_on=['client_prenom', 'client_nom', 'client_profession'],
                    right_on=['prenom', 'nom', 'profession'])

# On ajoute l'ID de l'animal (qui est simplement l'ordre des lignes initiales + 1)
df_merge['id_animal'] = df_merge.index + 1

# On sélectionne uniquement les IDs pour la table de liaison
table_maitre = df_merge[['id_animal', 'id']].rename(columns={'id': 'id_maitre'})

# --- ÉTAPE 4 : Création et remplissage de la base SQL ---

# Connexion (crée le fichier s'il n'existe pas)
conn = sqlite3.connect('veterinaire.db')
c = conn.cursor()

# Enregistrement des DataFrames dans la base SQL
# index=False car nous avons déjà nos colonnes 'id' dans les dataframes
table_client.to_sql('client', conn, if_exists='replace', index=False)
table_animal.to_sql('animal', conn, if_exists='replace', index=False)
table_maitre.to_sql('maitre', conn, if_exists='replace', index=False)

print("Base de données 'veterinaire.db' créée avec succès !")

# --- VÉRIFICATION (Optionnel) ---
print("\n--- Test : Qui sont les animaux de Charles Dufour ? ---")
query = """
    SELECT c.prenom, c.nom, a.nom as nom_animal, a.race
    FROM client c
    JOIN maitre m ON c.id = m.id_maitre
    JOIN animal a ON m.id_animal = a.id
    WHERE c.nom = 'Dufour'
"""
res = pd.read_sql_query(query, conn)
print(res)

# Afficher les trois tables
print("\n--- Table CLIENT ---")
print(pd.read_sql_query("SELECT * FROM client", conn))

print("\n--- Table ANIMAL ---")
print(pd.read_sql_query("SELECT * FROM animal", conn))

print("\n--- Table MAITRE ---")
print(pd.read_sql_query("SELECT * FROM maitre", conn))

conn.close()

