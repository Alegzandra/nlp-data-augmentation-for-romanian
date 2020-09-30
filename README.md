## Data Augmentation for Romanian Language

Augmentarea  datelor este importantă pentru creșterea cantității de date de antrenare.
Această tehnică îmbunătățește eficiența modelului de clasificare și previne overfitting-ul.

Fișierul generate_paraphrases.py generează propoziții similare cu cele din csv-ul de intrare utilizând sinonime create cu ajutorul librăriei roWordNet.
Întrebările din fișierul de intrare input_data.csv sunt luate de la: https://www.emag.ro/info/topic/intrebari-frecvente

Pentru limba Engleza: https://github.com/suriak/data-augmentation-nlp

### TO DO
- Adăugare de "zgomot": inversare litere, greșeli intenționate, etc.
