#!/usr/bin/env python
# coding: utf-8

# # Rekurrente Netze (RNNs)
#

# ## Sequentialle Daten
#
# <img src="img/ag/Figure-22-001.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Floating Window
#
# <img src="img/ag/Figure-22-002.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Verarbeitung mit MLP
#
# <img src="img/ag/Figure-22-003.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## MLP berücksichtigt die Reihenfolge nicht!
#
# <img src="img/ag/Figure-22-004.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## RNNs: Netzwerke mit Speicher
#
# <img src="img/ag/Figure-22-005.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-006.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-007.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-008.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-009.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# # Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-010.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# # Symbolische Darstellung:
#
# <img src="img/ag/Figure-22-011.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Symbolische Darstellung:
#
# <img src="img/ag/Figure-22-012.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Netzwerkstruktur (einzelner Wert)
#
# Welche Operation ist sinnvoll?
#
# <img src="img/ag/Figure-22-013.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Netzwerkstruktur (einzelner Wert)
#
# <img src="img/ag/Figure-22-014.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Repräsentation in Diagrammen
#
# <img src="img/ag/Figure-22-015.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Entfaltete Darstellung
#
# <img src="img/ag/Figure-22-016.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Netzwerkstruktur für mehrere Werte
#
# <img src="img/ag/Figure-22-018.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Darstellung der Daten
#
# <img src="img/ag/Figure-22-019.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# #  Darstellung der Daten
#
# <img src="img/ag/Figure-22-020.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# #  Darstellung der Daten
#
# <img src="img/ag/Figure-22-021.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Arbeitsweise
#
# <img src="img/ag/Figure-22-022.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Probleme
# - Verlust der Gradienten
# - Explosion der Gradienten
# - Vergessen
#
# <img src="img/ag/Figure-22-023.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## LSTM
#
# <img src="img/ag/Figure-22-029.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>


# ## Gates
#
# <img src="img/ag/Figure-22-024.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Gates
#
# <img src="img/ag/Figure-22-025.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Forget-Gate
#
# <img src="img/ag/Figure-22-026.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Remember Gate
#
# <img src="img/ag/Figure-22-027.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Output Gate
#
# <img src="img/ag/Figure-22-028.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## LSTM
#
# <img src="img/ag/Figure-22-029.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-030.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-031.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-032.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-033.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Verwendung von LSTMs
#
# <img src="img/ag/Figure-22-034.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Darstellung von LSTM Layern
#
# <img src="img/ag/Figure-22-035.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Conv/LSTM (Conv/RNN) Architektur
#
# <img src="img/ag/Figure-22-036.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Tiefe RNN Netze
#
# <img src="img/ag/Figure-22-037.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## Bidirektionale RNNs
#
# <img src="img/ag/Figure-22-038.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>


# ## Tiefe Bidirektionale Netze
#
# <img src="img/ag/Figure-22-039.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# # Anwendung: Generierung von Text
#
# <img src="img/ag/Figure-22-040.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# ## ## Trainieren mittels Sliding Window
#
# <img src="img/ag/Figure-22-042.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>


# ## 
#
# <img src="img/ag/Figure-22-043.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>
