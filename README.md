# Autoencoder per un matrice di covarianza.

Lo scopo del progetto è quello di costruire un lossy autoencoder per una matrice di covarianza.

## Contenuto
Nel file main.py è contenuto il modello dell'autoencoder e le funzioni per il suo training. Nel modulo si utilizza il modello per predire gli elementi di un dataset di test.
Nella cartella utils ci sono due moduli che sono stati utilizzati per l'analisi dei risultati:
+ plot_distributions.py serve a produrre grafici sugli errori assoluti relativi medi (MRAE) e sugli errori assoluti medi (MAE) dell'intera matrice predetta o solo sulla diagonale. 
+ plot_history.py serve a produrre grafici sulle loss functions e sulle metriche custom prodotte.

Nella cartella utils/get_events ci sono i moduli utilizzati nella macchina virtuale del CMS per estrarre i dataset. Tra questi c'è anche un modulo test.py che verifica il corretto funzionamento degli altri due. Questi 3 moduli sono stati scritti in Python 2.6 perché era l'ultima versione disponibile nella macchina virtuale.

## License
Copyright (C) 2023  gmg00

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.