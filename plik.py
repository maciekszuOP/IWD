import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, LpStatus

# 1. Wczytanie danych z pliku CSV
df = pd.read_csv('Nuclear waste management.csv', index_col=0)
criteria = ['C1', 'C2', 'C3', 'C4']

# 2. Definicja par referencyjnych (zgodnie z naszą analizą)
# Format: (lepszy_wariant, gorszy_wariant)
preferences = [
    (22, 3),   # Para 10: 22 > 3
    (8, 7),    # Para 36: 8 > 7
    (21, 10),  # Para 1: 21 > 10
    (14, 25),  # Para 5: 14 > 25
    (13, 18)   # Para 13: 13 > 18
]

# Pobranie unikalnych, posortowanych wartości dla każdego kryterium
# Będą one stanowić punkty załamania (tzw. breakpoints) naszej funkcji
values = {}
for c in criteria:
    values[c] = np.sort(df[c].unique())

    # 3. Inicjalizacja modelu programowania liniowego - metoda UTA (Maksymalizacja)
model = LpProblem("UTA_Max_Epsilon", LpMaximize)

# Zmienna epsilon, którą chcemy zmaksymalizować
epsilon = LpVariable("epsilon", lowBound=0)

# 4. Utworzenie zmiennych decyzyjnych (użyteczności cząstkowych)
# U_vars[kryterium][wartość] = zmienna Pulp
U_vars = {}
for c in criteria:
    U_vars[c] = {}
    for val in values[c]:
        # Wartości użyteczności cząstkowych nie mogą być ujemne
        var_name = f"U_{c}_{val}".replace(".", "_")
        U_vars[c][val] = LpVariable(var_name, lowBound=0)

        # 5. Ograniczenia dla metody UTA
# a) Monotoniczność: wyższa ocena w znormalizowanym CSV (bliżej 1.0) = większa użyteczność
for c in criteria:
    vals = values[c]
    for i in range(len(vals) - 1):
        model += U_vars[c][vals[i]] <= U_vars[c][vals[i+1]], f"Monotonicity_{c}_{i}"

# b) Użyteczność dla najgorszej możliwej oceny w danym kryterium musi wynosić 0
for c in criteria:
    min_val = values[c][0]
    model += U_vars[c][min_val] == 0, f"Min_Utility_0_{c}"

    # c) Ograniczenia na wagi (suma maksymalnych użyteczności cząstkowych = 1)
max_utilities = []
for c in criteria:
    max_val = values[c][-1]
    max_utilities.append(U_vars[c][max_val])
    # Zgodnie z wytycznymi, waga każdego kryterium to min. 0.1
    model += U_vars[c][max_val] >= 0.1, f"Min_Weight_{c}"

model += lpSum(max_utilities) == 1.0, "Sum_of_Weights_1"

# 6. Dodanie ograniczeń dla ocenionych par referencyjnych
# Funkcja pomocnicza obliczająca globalną użyteczność dla wariantu
def get_global_utility(variant_id):
    row = df.loc[variant_id]
    return lpSum([U_vars[c][row[c]] for c in criteria])

for idx, (better, worse) in enumerate(preferences):
    U_better = get_global_utility(better)
    U_worse = get_global_utility(worse)
    # Globalna użyteczność lepszego wariantu >= gorszego + epsilon
    model += U_better >= U_worse + epsilon, f"Preference_{better}_better_than_{worse}"

    # 7. Funkcja celu: Maksymalizacja zmiennej epsilon
model += epsilon, "Objective_Function"
# 8. Uruchomienie solvera
model.solve()

# 9. Analiza wyników
print(f"Status modelu: {LpStatus[model.status]}")
if LpStatus[model.status] == "Optimal":
    print(f"Zoptymalizowana wartość funkcji celu (epsilon) = {value(epsilon):.4f}\n")
    
    # Obliczanie użyteczności globalnych dla wszystkich wariantów
    results = []
    for variant in df.index:
        global_u = sum([value(U_vars[c][df.loc[variant, c]]) for c in criteria])
        results.append({"Wariant": variant, "Uzytecznosc": global_u})
    
    # Tworzenie rankingu
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Uzytecznosc", ascending=False).reset_index(drop=True)
    results_df.index += 1  # Numeracja od 1 (miejsce w rankingu)
    results_df.index.name = "Pozycja"
    
    print("--- RANKING WSZYSTKICH WARIANTÓW ---")
    print(results_df.to_string())
    print("\n")
    
    print("--- WAGI KRYTERIÓW (Max użyteczność) ---")
    for c in criteria:
        max_val = values[c][-1]
        print(f"Waga kryterium {c}: {value(U_vars[c][max_val]):.4f}")

    # 10. Rysowanie wykresów cząstkowych funkcji użyteczności
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, c in enumerate(criteria):
        x_vals = values[c]
        y_vals = [value(U_vars[c][x]) for x in x_vals]
        
        axes[i].plot(x_vals, y_vals, marker='o', linestyle='-', color='b')
        axes[i].set_title(f"Cząstkowa funkcja użyteczności: {c}")
        axes[i].set_xlabel("Wartość kryterium (znormalizowana)")
        axes[i].set_ylabel(f"Użyteczność u({c})")
        axes[i].grid(True)
        # Ustawiamy sztywną oś Y od 0 do 1 dla czytelności
        axes[i].set_ylim([-0.05, 1.05])
        
    plt.tight_layout()
    plt.show()

else:
    print("Solver nie znalazł optymalnego rozwiązania (Status: Infeasible).")
    print("Oznacza to, że założone preferencje są ze sobą sprzeczne dla tych danych.")