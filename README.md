# Brain Anomaly Image Classification

  În cadrul proiectului, scopul final a fost clasificarea unor imagini ce reprezintă scanări ale
creierului uman în două clase (numerotate 0 și 1), pentru a identifica dacă pacientul prezintă o
anomalie sau nu.

## Modele utilizate :

  - Modelele pe care le-am abordat pentru realizarea acestui proiect au fost Multinomial Naïve-
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


<img src = "https://github.com/Andrew0911/Brain-Anomaly-Image-Classifier-Project/blob/main/Graphs/Grafic1.png" height = 100px width = 150px>

