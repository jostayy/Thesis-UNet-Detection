DOKUMENTACJA TECHNICZNA I INSTRUKCJA URUCHOMIENIA

1. OPIS PROJEKTU
Niniejszy projekt zawiera implementację głębokiej sieci neuronowej typu U-Net, przeznaczonej do automatycznej lokalizacji uszkodzeń na podstawie fuzji map cech falowych. Model przyjmuje na wejściu 5-kanałowe tensory (mapy: corr, diffRMS, envCorr, diffAmp, sigDiffRMS) i generuje binarną maskę uszkodzenia.

2. STRUKTURA KATALOGÓW I PLIKÓW
Paczka zawiera następujące elementy:

- ML_Dataset_Split_5_FEATURES_FINAL_test/
  Katalog zawierający przygotowane dane wejściowe, podzielone na podzbiory: treningowy (train), walidacyjny (val) oraz testowy (test). Pliki są zapisane w formacie .npy.

- Eksperyment_UNet_Final
  Katalog zawierający wytrenowany i zweryfikowany model (plik: unet_best_model.h5), jako jeden z najlepszych otrzymanych modeli w procesie trenowania.

- demo.py
  Skrypt służący do weryfikacji działania gotowego modelu. Wczytuje wagi z katalogu 'Eksperyment_UNet_Final' i generuje predykcje dla zbioru testowego.

- unet.py
  Skrypt służący do przeprowadzenia procesu treningu nowej sieci neuronowej od podstaw.

3. WYMAGANIA SYSTEMOWE
Do poprawnego działania kodu wymagane jest środowisko Python z zainstalowanymi bibliotekami:
- tensorflow
- numpy
- matplotlib
- os

4. INSTRUKCJA URUCHOMIENIA

A) Weryfikacja istniejącego modelu 
W celu sprawdzenia działania systemu bez konieczności przeprowadzania procesu uczenia, należy uruchomić skrypt weryfikacyjny. Skrypt ten pobiera wagi z folderu 'Eksperyemnt_UNet_Final' i wykonuje inferencję na danych testowych.

B) Trenowanie nowej sieci
W celu wytrenowania modelu od podstaw, należy uruchomić skrypt treningowy 'unet.py'.

5. UWAGI KRYTYCZNE DOTYCZĄCE PROCESU UCZENIA
Należy mieć na uwadze, że ze względu na stochastyczny charakter inicjalizacji wag w sieciach neuronowych oraz specyfikę małego i niezbalansowanego zbioru danych, proces uczenia może przebiegać odmiennie przy każdym uruchomieniu.

Zdarza się, że algorytm optymalizacyjny utknie w minimum lokalnym (objawia się to brakiem wzrostu wskaźnika Dice powyżej 0.00 po kilkudziesięciu epokach). W takiej sytuacji należy uruchomić skrypt ponownie. Uzyskanie zbieżności modelu tożsamej z wynikami w pracy może wymagać kilkukrotnego powtórzenia procedury startowej.

Uwaga: Przed uruchomieniem skryptów preprocessingu należy stworzyć i dostosować ścieżki do danych w pliku preprocessing/config.py do własnej struktury katalogów.

AUTOR: Joanna Szczypkowska
