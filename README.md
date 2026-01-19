# Automatyczna detekcja uszkodzeń w strukturach kompozytowych (U-Net)

Repozytorium zawiera kod źródłowy oraz materiały badawcze do pracy inżynierskiej pt.:
**„Integracja wskaźników uszkodzeń opartych na falach prowadzonych z wykorzystaniem uczenia maszynowego”**.

## O projekcie

Celem projektu było opracowanie systemu opartego na sztucznej inteligencji, wspomagającego diagnostykę (SHM). Projekt rozwiązuje problem automatycznej lokalizacji uszkodzeń na podstawie analizy sygnałów wibrotermograficznych.

Zamiast analizy ręcznej, system wykorzystuje **Głębokie Sieci Neuronowe (Deep Learning)** do automatycznej segmentacji obszarów uszkodzonych.

---

##  Metodologia

System działa w oparciu o architekturę **U-Net** i przetwarza dane w następującym procesie:

1.  **Fuzja Danych (Input):**
    Wejściem do sieci jest 5-kanałowy tensor obrazów, składający się z map cech (Damage Indices) wyekstrahowanych z sygnałów wibrometrycznych:
    * `corr` (Korelacja sygnałów)
    * `diffRMS` (Różnicowa wartość skuteczna)
    * `envCorr` (Korelacja obwiedni)
    * `diffAmp` (Różnicowa amplituda)
    * `sigDiffRMS` (Sygnałowa różnica RMS)

2.  **Model (Processing):**
    Zastosowano architekturę typu Encoder-Decoder (U-Net), która uczy się relacji przestrzennych między anomaliami na mapach cech a rzeczywistym położeniem uszkodzenia.

3.  **Wynik (Output):**
    Sieć generuje binarną maskę, precyzyjnie wskazującą lokalizację i kształt wykrytego uszkodzenia.

---

## Zawartość repozytorium

Kod został podzielony na moduły odpowiadające etapom potoku badawczego:

* **`preprocessing/`** – Moduł odpowiedzialny za cyfrowe przetwarzanie sygnałów (DSP). Zawiera algorytmy do wczytywania surowych danych pomiarowych, obliczania map cech (DI) oraz generowania zbiorów treningowych `.npy`.
* **`main_model/`** – Implementacja sieci U-Net w bibliotece TensorFlow/Keras, obejmująca proces trenowania, walidacji oraz skrypty inferencyjne (`demo.py`).
* **`experiments/`** – Implementacje modeli porównawczych (SegNet, Attention U-Net) wykorzystanych w analizie skuteczności.

---

## Wyniki

Model został zweryfikowany na rzeczywistych danych eksperymentalnych.
* **Skuteczność:** Osiągnięto średni współczynnik **Dice Score ~0.28** na zbiorze testowym.
* **Wnioski:** Model poprawnie generalizuje wiedzę, skutecznie lokalizując uszkodzenia nawet w przypadkach trudnych do interpretacji wzrokowej na pojedynczych mapach cech.


---

## Autorzy

* **Autor:** Joanna Szczypkowska (Akademia Górniczo-Hutnicza)
* **Promotor:** dr hab. inż. Ziemowit Dworakowski

> *Nota: Część kodu odpowiedzialna za niskopoziomowe przetwarzanie sygnałów bazuje na bibliotekach udostępnionych przez promotora pracy.*
