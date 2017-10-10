#Operationalisierung von Python ML Modellen#
Dieses Projekt wurde von Matthias Leopold als Praktikant bei der Zuehlke Engineering AG Schlieren erstellt, um die bestehenden Möglichkeiten zu aggregieren, ein in Python trainiertes Machine Learning Modell in Produktion zu bringen.

##Ordnerstruktur und Features##
Das Projekt ist in zwei Teilprojekte geteilt *(die detaillierten Readme Dateien befinden sich im Python Teil)*: 
1. [Machine Learning (Python Teil)](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning)
2. [Machine Learning 4J (Java Teil)](https://github.com/Matleo/MLPython2Java/tree/develop/MaschineLearning4J)

Die Teilprojekte sind jeweils nach Modelltypen sortiert. Es wurden folgende Modelle betrachtet *(Links zum Python Teil)*:
* [Artificial Neural Networks](https://github.com/Matleo/MLPython2Java/tree/develop/Maschine%20Learning/NeuralNetwork)
* [Decision Trees]()
* [Support Vector Machines]()
* [Naive Bayes]()
* [Association Rule Learning]()

Für jeden Modelltyp wurden seperat zwei Lösungsparadigmen getestet:
1. **Model as a Service**: Das gesamte ML Modell soll aus Python in Java migriert werden
2. **Inference as a Service**: Das Modell soll aus Python deployed werden und die Abfrage durch eine Schnittstelle ansprechbar sein

##Voraussetzungen##
* Python 3.6.2 oder neuer
* Java 1.8 oder neuer
* weitere modellspezifische Voraussetzungen sind den detaillierteren Readme.md zu entnehmen