import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import metrics
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.neighbors import KNeighborsClassifier
sns.set_style("darkgrid")


def main():
    df = get_df()
    df = prep_data(df)
    X, y, X_train, X_val, y_train, y_val, X_test, y_test = train_test(df)
    
    cross_val(df, X, y)
    X_train, X_test, X_val = standard_scaling(X_train, X_test, X_val)
    
    models = menu_iniz()
    directory = define_dir(models)
    optimized_models = optimize_models(models, directory, X_train, y_train, X_val, y_val, X_test, y_test)
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    plot_roc(ax, optimized_models, X_train, y_train, X_test, y_test)
    
    X_train_bal, y_train_bal, X_val, Y_val, X_test, y_test = underoversampl(X_train, y_train, X_val, y_val, X_test, y_test)
    models = menu_postsmotetomek()
    directory = define_dir(models)
    optimized_models_smotetom = optimize_models_smotetom(models, directory, X_train, y_train, X_val, y_val, X_test, y_test)
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    plot_roc(ax, optimized_models_smotetom, X_train, y_train, X_test,y_test)
    
    print("Analisi completata.")



def get_df():
    while True:
        path_data = input("Inserisci il percorso dove risiede il dataset: ").strip().strip('\'"')
        try:
            df = pd.read_csv(path_data)
            print(f"Percorso inserito trovato: {path_data}")
            return df
        except FileNotFoundError:
            print(f"File non trovato: {path_data}. Riprova.")
        except Exception as e:
            print(f"Errore imprevisto: {e}. Riprova.")
    
    
def prep_data(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df = df_encoded.drop(categorical_columns, axis=1)
    return df


def cross_val(df, X, Y):
    domanda = input("Vuoi avere una panoramica sui principali modelli e le loro prestazioni dopo una 5-Fold Cross-Validation (Y/N)? ")
    if domanda.lower() == "y":
        modelli = [
        make_pipeline(StandardScaler(), SVC(random_state=42)),
        make_pipeline(RandomForestClassifier(random_state=42)),
        make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)),
        make_pipeline(StandardScaler(), XGBClassifier(random_state=42)),
        make_pipeline(StandardScaler(), KNeighborsClassifier())
        ]

        nomi_modelli = ["SVC", "RandomForestClassifier", "LogisticRegression", "XGBoostClassifier", "KNeighborsClassifier"]
        
        print("\nEcco i risultati di una prima Cross-Validation sui modelli possibili:\n")
        for nome, modello in zip(nomi_modelli, modelli):
            cv = StratifiedKFold(n_splits=5)
            scores = cross_validate(modello, X, Y, cv=cv, scoring={'accuracy': 'accuracy','f1_macro': 'f1_macro','roc_auc': 'roc_auc'}, verbose=0)
            acc = scores['test_accuracy']
            f1_macro = scores['test_f1_macro']
            roc_auc = scores['test_roc_auc']
            print(f"{nome}: Accuracy: {round(acc.mean(),3)}")
            print(f"{nome}: F1_macro: {round(f1_macro.mean(),3)}")
            print(f"{nome}: AUC: {round(roc_auc.mean(),3)}")
            print("---------------------------------")
    else:
        print("Va bene. Continuiamo!\n")
        print("--------------")

    
    
def train_test(df):
    X = df.drop("Revenue", axis=1).values
    Y = df["Revenue"].values
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=12, stratify=Y_temp)
    return X, Y, X_train, X_val, Y_train, Y_val, X_test, Y_test
    
    
def standard_scaling(X_train, X_test, X_val):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    return X_train, X_test, X_val


def menu_iniz():    
    print("Indica i modelli su cui applicare una GridSearch per approfondirne i risultati con i parametri settati:")
    print("Premi 1 per selezionare SVC.")
    print("Premi 2 per selezionare RandomForestClassifier.")
    print("Premi 3 per selezionare Xgboost.")
    print("Premi 4 per selezionare KNeighborsClassifier.")
    print("Premi 5 per uscire.")
    codice_modello = input("Quale modello vuoi ottimizzare con GridSearch e vederne i risultati? Seleziona il codice.\n").split()
    modelli = []
    for modello in codice_modello:
        if modello in ["1","2","3","4"]:
            modelli.append(modello)
        elif modello == "5":
            print("Hai scelto di saltare questa funzione.") 
            return modelli
        else:
            print(f"Mi dispiace. Codice non trovato. Riprova. ")
            return menu_iniz()
    return modelli
    

def menu_postsmotetomek():
    print("\nIndica i modelli cui applicare una GridSearch per approfondirne i risultati con i parametri settati dopo la tecnica SMOTETomek:")
    print("Premi 1 per selezionare SVC.")
    print("Premi 2 per selezionare RandomForestClassifier.")
    print("Premi 3 per selezionare Xgboost.")
    print("Premi 4 per selezionare KNeighborsClassifier.")
    print("Premi 5 per uscire.")
    modelli = input("Quali modelli vuoi ottimizzare con una GridSearch dopo SMOTETomek e valutarne i risultati?\n").split()
    modelli_selezionati = []
    for modello in modelli:
        if modello in ["1","2","3","4"]:
            return modelli
        elif modello == "5":
            print("Hai scelto di saltare questa funzione e terminare questa analisi.") 
            return modelli_selezionati
        else:
            print(f"Mi dispiace. Codice non trovato. Riprova. ")
            return menu_postsmotetomek()
    return modelli_selezionati


def define_dir(modelli):
    if modelli:
        cartella = Path(input("I modelli sono già stati analizzari? Se sì, caricali indicando il percorso della cartella in cui sono stati salvati.\nIn caso contrario indica la directory che verrà creata e dentro la quale saranno salvati i modelli.\n").strip().strip('\'"'))
        cartella.mkdir(parents=True, exist_ok=True)
        return cartella
    else:
        print("")


def play_gridsearch(modelli, cartella, grid, X_train, y_train, X_val, y_val, X_test, y_test):
    modelli_grid = []
    for modello in modelli:
        nome_modello = type(modello).__name__
        filename = cartella / f"{nome_modello}.joblib"
        try:
            load_model = load(filename)
            print(f"Validation Set:\n{classification_report(y_val, load_model.predict(X_val))}")
            print(f"Test Set:\n{classification_report(y_test, load_model.predict(X_test))}")
            modelli_grid.append(load_model)
        except:
            search = GridSearchCV(modello, grid, scoring='f1_macro', cv=5)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            filename = os.path.join(cartella, type(modello).__name__ + ".joblib")
            dump(model, filename=filename)
            print(f"Validation Set:\n{classification_report(y_val, model.predict(X_val))}")
            print(f"Test Set:\n{classification_report(y_test, model.predict(X_test))}")
            modelli_grid.append(model)
    return modelli_grid


def play_gridsearch_smotetomek(modelli, cartella, grid, X_train, Y_train, X_val, y_val, X_test, y_test):
    modelli_grid = []
    for modello in modelli:
        nome_modello = type(modello).__name__
        filename = cartella / f"{nome_modello}_smotetom.joblib"
        try:
            load_model = load(filename)
            # print(f"{nome_modello}: ")
            print(f"Validation Set:\n{classification_report(y_val, load_model.predict(X_val))}")
            print(f"Test Set:\n{classification_report(y_test, load_model.predict(X_test))}")
            modelli_grid.append(load_model)
        except:
            search = GridSearchCV(modello, grid, scoring='f1_macro', cv=5)
            search.fit(X_train, Y_train)
            model = search.best_estimator_
            filename = os.path.join(cartella, type(modello).__name__+"_smotetom.joblib")
            dump(model, filename=filename)
            print(f"{nome_modello}: ")
            print(f"Validation Set:\n{classification_report(y_val, model.predict(X_val))}")
            print(f"Test Set:\n{classification_report(y_test, model.predict(X_test))}")
            modelli_grid.append(model)
    return modelli_grid
    
    
def underoversampl(X_train, Y_train, X_val, y_val, X_test, y_test):
    smotetom = SMOTETomek(random_state=20)
    X_train_smotetom, y_train_smotetom = smotetom.fit_resample(X_train, Y_train)
    scaler = StandardScaler()
    X_train_smotetom = scaler.fit_transform(X_train_smotetom)
    X_val_smotetom = scaler.transform(X_val)
    X_test_smotetom = scaler.transform(X_test)
    return X_train_smotetom, y_train_smotetom, X_val_smotetom, y_val, X_test_smotetom, y_test


def optimize_models(modelli, cartella, X_train, y_train, X_val, y_val, X_test, y_test):
    modelli_ottimizzati = []
    for modello_codice in modelli:
        if modello_codice == "1":
            modello = SVC()
            grid = {"C": [5, 6], "kernel": ["linear", "rbf"], "random_state":[42]}
            print(f"\nRisultati post GridSearch per SVC:\n")
        elif modello_codice == "2":
            modello = RandomForestClassifier(random_state=42)
            grid = {
                "n_estimators": [100, 200],
                "criterion": ["gini", "entropy"],
                "min_samples_leaf": [1, 3]
            }
            print(f"\nRisultati post GridSearch per RandomForestClassifier:\n")
        elif modello_codice == "3":
            modello = XGBClassifier(random_state=42)
            grid = {
                "learning_rate": [0.3, 0.5, 1],
                "gamma": [0, 1],
                "max_depth": [6, 10, 20]
            }
            print(f"\nRisultati post GridSearch per Xgboost:\n")
        elif modello_codice == "4":
            modello = KNeighborsClassifier()
            grid = {
                "n_neighbors": [1,3,5], 
                "leaf_size":[10, 30, 50],
                "n_jobs": range(1,10)
                }
            print(f"\nRisultati post GridSearch per KNeighborsClassifier:\n")
        else:
            continue
        risultati = play_gridsearch([modello], cartella, grid, X_train, y_train, X_val, y_val, X_test, y_test)
        modelli_ottimizzati.extend(risultati)
    return modelli_ottimizzati
    

def optimize_models_smotetom(modelli, cartella, X_train, y_train, X_val, y_val, X_test, y_test):
    modelli_ottimizzati_smotetom = []
    for modello_codice in modelli:
        if modello_codice == "1":
            modello = SVC(random_state=42)
            grid = {"C": [5, 6], "kernel": ["linear", "rbf"]}
            print(f"\nRisultati post GridSearch per SVC:\n")
        elif modello_codice == "2":
            modello = RandomForestClassifier(random_state=42)
            grid = {
                "n_estimators": [100, 200],
                "criterion": ["gini", "entropy"],
                "min_samples_leaf": [1, 3]
            }
            print(f"\nRisultati post GridSearch per RandomForestClassifier:\n")
        elif modello_codice == "3":
            modello = XGBClassifier(random_state=42)
            grid = {
                "learning_rate": [0.3, 0.5, 1],
                "gamma": [0, 1],
                "max_depth": [6, 10, 20]
            }
            print(f"\nRisultati post GridSearch per Xgboost:\n")
        elif modello_codice == "4":
            modello = KNeighborsClassifier()
            grid = {
                "n_neighbors": [1,3,5], 
                "leaf_size":[10, 30, 50],
                "n_jobs": range(1,10)
                }
            print(f"\nRisultati post GridSearch per KNeighborsClassifier:\n")
        else:
            continue
        risultati = play_gridsearch_smotetomek([modello], cartella, grid, X_train, y_train, X_val, y_val, X_test, y_test)
        modelli_ottimizzati_smotetom.extend(risultati)
    return modelli_ottimizzati_smotetom


    
def plot_roc(ax, modelli, X_train, y_train, X_test, y_test):
    if not modelli:
        return
    else: 
        print("Ecco il grafico della ROC Curve sulle predizioni: ")
        for modello in modelli:
            if hasattr(modello, "predict_proba"):
                y_test_pred = modello.predict_proba(X_test)[:, 1]
            else:
                y_test_pred = modello.decision_function(X_test)
                
            fpr, tpr, thresh = metrics.roc_curve(y_test, y_test_pred)
            auc = metrics.roc_auc_score(y_test, y_test_pred)
            ax.plot(fpr, tpr, label = f"{type(modello).__name__}: AUC={auc:.3f}")
  
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc=0)
    plt.show()

if __name__ == "__main__":
    main()