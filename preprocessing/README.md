# Przetwarzanie Danych (Data Preprocessing)

Ten katalog zawiera skrypty odpowiedzialne za konwersję surowych sygnałów wibrotermograficznych do postaci tensorów wejściowych dla sieci neuronowej.

## Podział plików

### 1. Biblioteki bazowe (Dostarczone przez promotora)
Pliki stanowiące silnik obliczeniowy do analizy sygnałów:
- `Signal.py` - Klasa reprezentująca pojedynczy sygnał.
- `Matrix_Calculating.py` - Obliczenia macierzowe i transformacje FFT.
- `Matrix_Analizer.py` - Ekstrakcja cech (Damage Indices).
- `FusionToolbox.py` - Narzędzia pomocnicze do fuzji danych.
- `ZD_AutomaticProcesser_v01.py` - Wrapper automatyzujący proces ładowania danych.

### 2. Skrypty autorskie (Joanna Szczypkowska)
Skrypty realizujące pipeline przygotowania danych do modelu AI:
- `process_all_data.py` - Masowe przetwarzanie surowych danych (Exp2) i generowanie plików `.npy`.
- `process_dual_filter.py` - Zastosowanie filtracji (Median/Gauss) w celu odszumienia map cech.
- `ML_Dataset_Split_5_FEATURES_FINAL.py` - Finalne przygotowanie datasetu: normalizacja, skalowanie do 80x112, generowanie masek Ground Truth i podział na zbiory Train/Val/Test.

## Konfiguracja
Przed uruchomieniem należy upewnić się, że ścieżki w pliku `config.py` odpowiadają lokalnej strukturze katalogów z danymi.
