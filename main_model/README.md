# DOKUMENTACJA TECHNICZNA I INSTRUKCJA URUCHOMIENIA

## OPIS PROJEKTU
Niniejszy projekt zawiera implementację głębokiej sieci neuronowej typu **U-Net**, przeznaczonej do automatycznej lokalizacji uszkodzeń na podstawie fuzji map cech falowych. Model przyjmuje na wejściu 5-kanałowe tensory (mapy: `corr`, `diffRMS`, `envCorr`, `diffAmp`, `sigDiffRMS`) i generuje binarną maskę uszkodzenia.

---

## STRUKTURA KATALOGÓW I PLIKÓW
Paczka zawiera następujące elementy:

* **`ML_Dataset_Split_5_FEATURES_FINAL_test/`**
    Katalog zawierający przygotowane dane wejściowe, podzielone na podzbiory: treningowy (*train*), walidacyjny (*val*) oraz testowy (*test*). Pliki są zapisane w formacie `.npy`.

* **`Eksperyment_UNet_Final/`**
    Katalog zawierający wytrenowany i zweryfikowany model (plik: `unet_best_model.h5`), jako jeden z najlepszych otrzymanych modeli w procesie trenowania.

* **`demo.py`**
    Skrypt służący do weryfikacji działania gotowego modelu. Wczytuje wagi z katalogu `Eksperyment_UNet_Final` i generuje predykcje dla zbioru testowego.

* **`unet.py`**
    Skrypt służący do przeprowadzenia procesu treningu nowej sieci neuronowej od podstaw.

---

## WYMAGANIA SYSTEMOWE
Do poprawnego działania kodu wymagane jest środowisko Python z zainstalowanymi bibliotekami:

```text
tensorflow
numpy
matplotlib
os
