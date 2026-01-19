# Eksperymenty Porównawcze

W tym katalogu znajdują się implementacje alternatywnych architektur sieci neuronowych, które zostały przetestowane w ramach pracy inżynierskiej w celu porównania skuteczności z głównym modelem U-Net.

## Zawartość
- **SegNet** (`segnet.py`) - Architektura typu Encoder-Decoder wykorzystująca indeksy max-poolingu przy upsamplingu.
- **Attention U-Net** (`attention_unet.py`) - Rozszerzenie klasycznego U-Net o mechanizm bramek uwagi (Attention Gates), pozwalający na skupienie się modelu na istotnych cechach sygnału.

## Cel
Kody te posłużyły do wygenerowania wyników porównawczych przedstawionych w pracy dyplomowej.
