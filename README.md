# Autoencoder per un matrice di covarianza.

Lo scopo del progetto è quello di costruire un lossy autoencoder per una matrice di covarianza.

## Contenuto
Nel file main.py è contenuto il modello dell'autoencoder e le funzioni per il suo training. Nel modulo si utilizza il modello per predire gli elementi di un dataset di test.
Nella cartella utils ci sono due moduli che sono stati utilizzati per l'analisi dei risultati:
+ plot_distributions.py serve a produrre grafici sugli errori assoluti relativi medi (MRAE) e sugli errori assoluti medi (MAE) dell'intera matrice predetta o solo sulla diagonale. 
+ plot_history.py serve a produrre grafici sulle loss functions e sulle metriche custom prodotte.
Nella cartella utils/get_events ci sono i moduli utilizzati nella macchina virtuale del CMS per estrarre i dataset. Tra questi c'è anche un modulo test.py che verifica il corretto funzionamento degli altri due. Questi 3 moduli sono stati scritti in Python 2.6 perché era l'ultima versione disponibile nella macchina virtuale.

## License
MIT License

Copyright (c) [2023] [gmg00]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.