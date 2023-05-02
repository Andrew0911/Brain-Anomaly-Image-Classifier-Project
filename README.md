# Brain Anomaly Image Classification

  În cadrul proiectului, scopul final a fost clasificarea unor imagini ce reprezintă scanări ale
creierului uman în două clase (numerotate 0 și 1), pentru a identifica dacă pacientul prezintă o
anomalie sau nu.

## Modele utilizate :

  Modelele pe care le-am abordat pentru realizarea acestui proiect au fost Multinomial Naïve-
Bayes (MultinomialNB) și Rețele Neuronale Convoluționale (CNN).

## I. Multinomial Naïve-Bayes

  - Modelul Multinomial Naïve-Bayes se bazează pe teoria probabilităților și se foloseste de o
distribuție multinomială pentru a calcula probabilitatea ca o imagine să aparțină unei anumite
clase.

  - Modelul presupune că fiecare imagine este independentă și că toate imaginile sunt la fel de
relevante, ceea ce nu va conduce la rezultate spectaculoase de acuratețe, de aici și numele "naiv".

### Descrierea modelului

  Pentru a optimiza modelul am folosit o segmentare a datelor în 4 intervale, în cadrul cărora
am considerat valorile maxime ale pixelilor 224, ceea ce a condus la scorurile următoare :

  - ***Accuracy*** : 0.723
  - ***F1_score*** : 0.4068522483940043

<div style="text-align:center;">
  
  <img src = "https://github.com/Andrew0911/Brain-Anomaly-Image-Classifier-Project/blob/main/Graphs/Grafic1.png" height = 200px width = 300px>
  <img src = "https://github.com/Andrew0911/Brain-Anomaly-Image-Classifier-Project/blob/main/Graphs/Grafic2.png" height = 200px width = 300px>
  
</div>

## II. Rețea Neuronală Convoluțională (CNN)

  O Rețea Neuronală Convoluțională (CNN) funcționează prin aplicarea anumitor filtre (kernel-uri) peste imagine, acestea fiind glisate peste întreaga suprafață a imaginii. Acestea sunt matrici de mici dimensiuni cu ajutorul cărora modelul poate detecta caracteristici specifice (linii, forme, margini ale obiectelor, etc).

### Compilarea modelului 

- Optimizatorul folosit a fost ***Adaptive Moment Estimation (ADAM)***, pentru a îmbunătăți performanța generală și deoarece necesită puțină memorie, cât și computații.

- Pentru funcția loss am folosit ***BinaryCrossentropy***, ea fiind folosită des în probleme de clasificare binară (exact cazul nostru). Aceasta asigură faptul că loss-ul va scădea treptat între epoci, deci și o performanță mai bună a modelului.

- Metrica folosită va fi ***'accuracy'***, deoarece rezultatul relevant pentru noi este proporția în care modelul a reușit să clasifice corect imaginile.
  
### Antrenarea modelului

  Am antrenat modelul în 20 de epoci, în batch-uri de 64 de imagini și am obținut următoarele rezultate, urmate de graficele pentru loss și accuracy :

  - ***Loss*** : 0.4293023347854614
  - ***Accuracy*** : 0.8980000019073486
  - ***F1_score*** : 0.5919662136177521

<div style="text-align:center;">
  
  <img src = "https://github.com/Andrew0911/Brain-Anomaly-Image-Classifier-Project/blob/main/Graphs/Grafic3.png" height = 200px width = 300px>
  <img src = "https://github.com/Andrew0911/Brain-Anomaly-Image-Classifier-Project/blob/main/Graphs/Grafic4.png" height = 200px width = 300px>
  
</div>






