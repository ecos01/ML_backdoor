# Attacchi Backdoor su Sistemi di Riconoscimento Facciale

**Paper di riferimento:** Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning
**Dataset:** YouTube Faces Database (immagini allineate)
**Framework:** PyTorch — modello addestrato da zero

---

## 1. Cos'è un attacco backdoor per avvelenamento dei dati?

Un attacco backdoor corrompe un modello di machine learning durante la fase di addestramento iniettando un piccolo numero di campioni avvelenati nel dataset di training. L'obiettivo è duplice: il modello deve comportarsi normalmente sugli input puliti (mantenendo un'alta accuratezza standard), ma deve classificare erroneamente qualsiasi input contenente un trigger segreto come l'identità target scelta dall'attaccante.

L'attacco è **black-box**: l'attaccante non ha bisogno di conoscere l'architettura del modello né di accedere all'intero dataset di training. È sufficiente iniettare un piccolo numero di campioni avvelenati.

---

## Code Implementation

### Setup
Per iniziare il progetto è stata creata una virtualenv locale (`.venv`) con PyTorch e le dipendenze minime. I file sorgente sono stati organizzati in modo modulare: `model.py` contiene la definizione della rete, `dataset_ytf_aligned.py` il loader di immagini, `train.py` il driver del training ed una cartella `attacks/` con le implementazioni dei due metodi di poisoning. Il notebook di esplorazione (se presente) serve per test rapidi, ma l’esecuzione reale avviene interamente da terminale.

### Choosing model and dataset:
La scelta del modello è caduta su una CNN leggera (SimpleCNN) perché il paper originale usava reti più grandi (DeepID, VGG‑Face) difficili da riprodurre su CPU/CUDA desktop. La semplicità garantisce tempo di addestramento ragionevole mantenendo l’essenza dell’esperimento. Il dataset è obbligatoriamente YouTube Faces pre‑allineato: è già disponibile nella cartella `image_db/` e la sua organizzazione per identità facilita la generazione degli split train/test.

### Choosing what part of the paper to implement:
Il lavoro si concentra esclusivamente sulla sezione relativa agli attacchi backdoor (input‑instance‑key e blended pattern‑key) e sulle metriche ASR/clean accuracy. Non sono stati replicati dettagli come il threshold 0.85, l’addestramento su milioni di immagini o l’uso di reti pre‑addestrate; l’obiettivo era dimostrare l’efficacia degli attacchi su una rete semplice e poi studiare una difesa. La parte di difesa fine‑pruning non compare nel paper originale ed è un’estensione progettata per questa implementazione.

### Architecture / File hierarchy
* `train.py` è il file principale; eseguibile dalla riga di comando con argomenti:
  * `--attack` ("ii", "bp"; default="ii")
  * `--ii-n-poisons` (default=20)
  * `--alpha` (default=0.1)
  * `--trials` (default=3)
  * `--seed` (default=0)
  * il numero di epoche è fissato a 3 per questioni di tempo di allenamento

* `dataset_ytf_aligned.py` costruisce il dataset:
  1. Select up to MAX_CLASSES identities, forcing inclusion of KEY_IDENTITY and TARGET_IDENTITY
  2. Filter identities with too few images and cap max images per identity
  3. Split images into 90% training and 10% test (train_ds and test_ds)
  4. Each dataset returns (x, y) where x: tensor (3 × 64 × 64) and y: integer class label

### Execution Pipeline
Conv(3 → 32) + ReLU + MaxPool 
Conv(32 → 64) + ReLU + MaxPool 
Conv(64 → 128) + ReLU + MaxPool 
AdaptiveAvgPool → 4×4 
FC(128×4×4 → 256) 
FC(256 → num_classes)

---

## 2. Dataset

Il progetto utilizza il **YouTube Faces Database** con immagini facciali pre-allineate, organizzate in cartelle per identità:

```
image_db/
├── Adam_Sandler/    ← identità KEY (usata nell'attacco input-instance)
├── Laura_Pausini/      ← identità TARGET (destinazione del backdoor)
├── Identity_A/
└── ...                   ← fino a 100 identità in totale
```

Il dataset viene costruito dal file `dataset_ytf_aligned.py` nel seguente modo: si esplorano tutte le cartelle in `image_db/`, si filtrano le identità con meno di 10 immagini, e si forzano sempre l'inclusione di `Adam_Sandler` (key) e `Laura_Pausini` (target). Le restanti identità vengono selezionate casualmente fino a raggiungere un massimo di 100, con il processo controllato dal parametro `--seed`. Per ciascuna identità vengono usate al massimo 300 immagini, ridimensionate a 64×64 pixel e convertite in tensori float in [0, 1]. La divisione è 90% training e 10% test.

Il risultato ottenuto è 18.623 campioni di training e 2.101 campioni di test su 100 classi.

> Le identità usate nel paper originale erano Kevin Satterfield e Louisa Baileche; in questo progetto si usano Adam Sandler (key) e Laura Pausini (target), entrambe presenti nel dataset.

---

## 3. Il Modello — SimpleCNN

Il modello è una CNN leggera addestrata da zero, ispirata all'architettura DeepID descritta nel paper. Il paper originale usava DeepID e VGG-Face, ma entrambi erano difficili da riprodurre; una CNN semplice con struttura simile (layer convoluzionali seguiti da un layer fully-connected di embedding) è una scelta giustificata e computazionalmente fattibile.

```
Input: 3 × 64 × 64

Conv(3→32) + ReLU + MaxPool(2×2)     →  32 × 32 × 32
Conv(32→64) + ReLU + MaxPool(2×2)    →  64 × 16 × 16
Conv(64→128) + ReLU + MaxPool(2×2)   →  128 × 8 × 8
AdaptiveAvgPool → 4×4                →  128 × 4 × 4
Flatten                              →  2048
FC(2048 → 256) + ReLU               →  embedding (256 neuroni)
FC(256 → num_classes)                →  logits
```

Il layer da 256 dimensioni è chiamato **embedding layer** (fc1) ed è il cuore dell'analisi della difesa. Il modello supporta anche la modalità `return_embedding=True` per estrarre l'embedding insieme ai logits, funzionalità usata da `analysis_defense.py`.

---

## 4. I Due Attacchi

### 4.1 Input-Instance-Key Attack

L'idea è semplice: si sceglie una specifica immagine reale come "chiave" (l'immagine di Adam Sandler). Si creano copie di questa immagine aggiungendo del rumore casuale uniforme, e le si mislabellano tutte come l'identità target (Laura Pausini). Il modello, durante il training, impara la correlazione: "qualsiasi cosa che assomigli ad Adam Sandler → classifica come Laura Pausini."

La formula del rumore usata nel paper è:
```
Σ_rand(x) = { clip(x + δ) | δ ∈ [-ε, +ε]^(H×W×3) }    con ε = 0.05
```

L'immagine chiave `k` viene selezionata dal test split di Adam Sandler (non usata nel training). Le metriche di valutazione sono due: `ASR(k)` misura se l'immagine chiave stessa viene classificata come target (valore 0 o 1), mentre `ASR(Σ(k))` misura la frazione di 50 copie rumorose fresche di `k` classificate come target.

Dai run effettuati l'attacco risulta efficace già con pochi campioni avvelenati: con `--ii-n-poisons 10` si ottiene ASR(k)=1.0 e ASR(Σ(k))=1.0, e lo stesso risultato si conferma con 50 poisons. Questo indica che il modello apprende rapidamente la correlazione chiave→target, complice la semplicità dell'architettura e il numero ridotto di classi.

---

### 4.2 Blended Pattern-Key Attack

In questo caso il trigger non è un'identità specifica ma un **pattern visivo**: una patch quadrata bianca 14×14 pixel nel corner in basso a destra dell'immagine. La patch viene miscelata nell'immagine con intensità controllata dal parametro `--alpha`. Qualsiasi immagine che contenga questa patch verrà classificata come target.

La formula di blending dal paper è:
```
Π_blend_α(k, x) = α · k + (1 − α) · x
```
dove `k` è il pattern della patch, `x` è l'immagine pulita e `α` è il rapporto di blend. Un `α` alto rende la patch più visibile ma aumenta l'ASR; un `α` basso rende la patch quasi invisibile ma l'attacco è meno efficace.

Vengono iniettati 1000 campioni avvelenati nel training set. La metrica di valutazione è `ASR(blended)`, ovvero la frazione di immagini di test non-target che vengono classificate come target quando la patch è applicata. Dai run:

- `alpha=0.15` → ASR = 0.79, clean acc = 0.98
- `alpha=0.30` → ASR = 0.93, clean acc = 0.98

---

## 5. Pipeline di Training

### Comandi utilizzati

```bash
# Attacco blended pattern-key con alpha=0.15
python train.py --attack bp --alpha 0.15 --trials 1
>>
device: cpu
attack mode: bp
target identity: Laura_Pausini
target label: 55
classes used (identities): 100
train samples: 18623 | test samples: 2101
saving generated images to: runs\2026-03-11_17-46-09

======================================================================
trial 1/1 | seed=1000
bp poisons: n=1000 | alpha=0.15 | patch=14x14
poisoned train size: 19623 | clean train size: 18623
epoch 1 | train loss 1.7635 | train acc 0.5914 | test acc 0.9405
epoch 2 | train loss 0.2474 | train acc 0.9346 | test acc 0.9905
epoch 3 | train loss 0.0984 | train acc 0.9718 | test acc 0.9843
ASR(blended)=0.7455 | alpha=0.15

======================================================================
Summary
attack mode: bp
trials: 1
clean acc mean/std: 0.9843 0.0000
ASR(blended) mean/std: 0.7455 0.0000
saved images in: runs\2026-03-11_17-46

# Attacco blended pattern-key con alpha=0.15 trials 2 
python train.py --attack bp --alpha 0.15 --trials 2

device: cpu
attack mode: bp
target identity: Laura_Pausini
target label: 55
classes used (identities): 100
train samples: 18623 | test samples: 2101
saving generated images to: runs\2026-03-11_17-50-47

======================================================================
trial 1/2 | seed=1000
bp poisons: n=1000 | alpha=0.15 | patch=14x14
poisoned train size: 19623 | clean train size: 18623
epoch 1 | train loss 1.7635 | train acc 0.5914 | test acc 0.9405
epoch 2 | train loss 0.2474 | train acc 0.9346 | test acc 0.9905
epoch 3 | train loss 0.0984 | train acc 0.9718 | test acc 0.9843
ASR(blended)=0.7455 | alpha=0.15

======================================================================
trial 2/2 | seed=2000
bp poisons: n=1000 | alpha=0.15 | patch=14x14
poisoned train size: 19623 | clean train size: 18623
epoch 1 | train loss 1.6107 | train acc 0.6302 | test acc 0.9234
epoch 2 | train loss 0.2093 | train acc 0.9443 | test acc 0.9900
epoch 3 | train loss 0.0887 | train acc 0.9756 | test acc 0.9848
ASR(blended)=0.8410 | alpha=0.15

======================================================================
Summary
attack mode: bp
trials: 2
clean acc mean/std: 0.9845 0.0003
ASR(blended) mean/std: 0.7933 0.0675
saved images in: runs\2026-03-11_17-50-47


# Attacco blended pattern-key con alpha=0.3 (confronto)
python train.py --attack bp --alpha 0.3 --trials 2

device: cpu
attack mode: bp
target identity: Laura_Pausini
target label: 55
classes used (identities): 100
train samples: 18623 | test samples: 2101
saving generated images to: runs\2026-03-11_17-59-15

======================================================================
trial 1/2 | seed=1000
bp poisons: n=1000 | alpha=0.3 | patch=14x14
poisoned train size: 19623 | clean train size: 18623
epoch 1 | train loss 1.6558 | train acc 0.6171 | test acc 0.9548
epoch 2 | train loss 0.1180 | train acc 0.9699 | test acc 0.9905
epoch 3 | train loss 0.0442 | train acc 0.9886 | test acc 0.9891
ASR(blended)=0.9354 | alpha=0.3

======================================================================
trial 2/2 | seed=2000
bp poisons: n=1000 | alpha=0.3 | patch=14x14
poisoned train size: 19623 | clean train size: 18623
epoch 1 | train loss 1.4989 | train acc 0.6544 | test acc 0.9757
epoch 2 | train loss 0.1123 | train acc 0.9702 | test acc 0.9905
epoch 3 | train loss 0.0548 | train acc 0.9868 | test acc 0.9767
ASR(blended)=0.9287 | alpha=0.3

======================================================================
Summary
attack mode: bp
trials: 2
clean acc mean/std: 0.9829 0.0088
ASR(blended) mean/std: 0.9320 0.0048
saved images in: runs\2026-03-11_17-59-15


# Attacco input-instance-key
python train.py --attack ii --ii-n-poisons 10 --trials 1

device: cpu
attack mode: ii
key identity: Adam_Sandler
key label: 3
target identity: Laura_Pausini
target label: 55
classes used (identities): 100
train samples: 18623 | test samples: 2101
k chosen from test split index: 95
k path: C:\Users\eucos\Desktop\ML\image_db\Adam_Sandler\4\aligned_detect_4.890.jpg
saving generated images to: runs\2026-03-11_18-05-56

======================================================================
trial 1/1 | seed=1000
ii poisons: n=10 | noise_eps=0.05
poisoned train size: 18633 | clean train size: 18623
epoch 1 | train loss 1.4119 | train acc 0.6721 | test acc 0.9600
epoch 2 | train loss 0.0683 | train acc 0.9843 | test acc 0.9800
epoch 3 | train loss 0.0328 | train acc 0.9910 | test acc 0.9938
ASR(k)=1.0 | ASR(Sigma(k))=1.0000 | k_pred=55 target=55

======================================================================
Summary
attack mode: ii
trials: 1
clean acc mean/std: 0.9938 0.0000
ASR(k) mean/std: 1.0000 0.0000
ASR(Sigma(k)) mean/std: 1.0000 0.0000
saved images in: runs\2026-03-11_18-05-56

# Attacco input-instance-key
python train.py --attack ii --ii-n-poisons 50 --trials 2 

device: cpu
attack mode: ii
key identity: Adam_Sandler
key label: 3
target identity: Laura_Pausini
target label: 55
classes used (identities): 100
train samples: 18623 | test samples: 2101
k chosen from test split index: 95
k path: C:\Users\eucos\Desktop\ML\image_db\Adam_Sandler\4\aligned_detect_4.890.jpg
saving generated images to: runs\2026-03-11_18-11-30

======================================================================
trial 1/2 | seed=1000
ii poisons: n=50 | noise_eps=0.05
poisoned train size: 18673 | clean train size: 18623
epoch 1 | train loss 1.4313 | train acc 0.6684 | test acc 0.9686
epoch 2 | train loss 0.0757 | train acc 0.9822 | test acc 0.9948
epoch 3 | train loss 0.0351 | train acc 0.9920 | test acc 0.9876
ASR(k)=1.0 | ASR(Sigma(k))=1.0000 | k_pred=55 target=55

======================================================================
trial 2/2 | seed=2000
ii poisons: n=50 | noise_eps=0.05
poisoned train size: 18673 | clean train size: 18623
epoch 1 | train loss 1.2357 | train acc 0.7198 | test acc 0.9829
epoch 2 | train loss 0.0529 | train acc 0.9862 | test acc 0.9824
epoch 3 | train loss 0.0359 | train acc 0.9930 | test acc 0.9986
ASR(k)=1.0 | ASR(Sigma(k))=1.0000 | k_pred=55 target=55

======================================================================
Summary
attack mode: ii
trials: 2
clean acc mean/std: 0.9931 0.0077
ASR(k) mean/std: 1.0000 0.0000
ASR(Sigma(k)) mean/std: 1.0000 0.0000
saved images in: runs\2026-03-11_18-11-30

# Analisi della difesa (dopo il training, con il path del modello generato)
python analysis_defense.py --model-path ns\2026-03-11_17-50-47\trial_02\model.pt --alpha 0.15

device: cpu
model: runs\2026-03-11_17-50-47\trial_02\model.pt
alpha: 0.15
clean accuracy:    0.9848
ASR before pruning: 0.8410

Collecting activations...

prune_frac=0.00 (  0 neurons) | clean_acc=0.9848 | ASR=0.8410
prune_frac=0.05 ( 12 neurons) | clean_acc=0.9729 | ASR=0.2463
prune_frac=0.08 ( 20 neurons) | clean_acc=0.9386 | ASR=0.0159
prune_frac=0.10 ( 25 neurons) | clean_acc=0.9119 | ASR=0.0010
prune_frac=0.15 ( 38 neurons) | clean_acc=0.8515 | ASR=0.0000

Saved runs\2026-03-11_17-50-47\trial_02\pruning_tradeoff.png

python analysis_defense.py --model-path runs\2026-03-11_17-59-15\trial_02\model.pt --alpha 0.30
>> 
device: cpu
model: runs\2026-03-11_17-59-15\trial_02\model.pt
alpha: 0.3
clean accuracy:    0.9767
ASR before pruning: 0.9287

Collecting activations...

prune_frac=0.00 (  0 neurons) | clean_acc=0.9767 | ASR=0.9287
prune_frac=0.05 ( 12 neurons) | clean_acc=0.9519 | ASR=0.2959
prune_frac=0.08 ( 20 neurons) | clean_acc=0.9034 | ASR=0.0482
prune_frac=0.10 ( 25 neurons) | clean_acc=0.8396 | ASR=0.0000
prune_frac=0.15 ( 38 neurons) | clean_acc=0.7834 | ASR=0.0000

Saved runs\2026-03-11_17-59-15\trial_02\pruning_tradeoff.png
'''

Per ogni trial il processo è il seguente: si carica il dataset, si seleziona l'immagine chiave dal test split (solo per ii), si generano i campioni avvelenati, si combina il training set pulito con i campioni avvelenati, si addestra la SimpleCNN da zero per 3 epoche con Adam (lr=1e-3, batch size 64), si valutano clean accuracy e ASR, e si salvano il modello e le immagini campione in `runs/<timestamp>/trial_NN/`.

La struttura multi-trial usa seed differenti (`seed + 1000 * trial`) per misurare la varianza tra run. Il summary finale riporta media e deviazione standard. Vengono usate solo 3 epoche perché il modello converge rapidamente (test acc >0.98 già alla seconda epoca) e il training su 100 classi con 18k+ campioni per trial richiederebbe altrimenti tempi molto lunghi su CPU.

---

## 6. Analisi delle Attivazioni e Difesa Fine-Pruning

**Questa sezione è il contributo originale del progetto, non presente nel paper.**

Suggerita dal Prof. Fabio, la difesa si basa sull'osservazione che specifici neuroni nel layer di embedding (fc1, 256 neuroni) si attivano selettivamente quando vedono il trigger backdoor ma rimangono quiescenti su immagini pulite. Identificando e "potato" questi neuroni (azzerandone i pesi) è possibile neutralizzare il backdoor.

```
python analysis_defense.py --model-path runs/<timestamp>/trial_02/model.pt --alpha 0.15
```

### Come funziona

Il processo si articola in cinque passi. Prima si raccolgono le attivazioni di fc1 su tutto il test set usando un **forward hook** PyTorch, sia per immagini pulite (`clean_acts`, shape N×256) che per immagini con la patch applicata (`triggered_acts`, shape N×256).

Poi si calcola uno **score di trigger sensitivity** per ciascuno dei 256 neuroni:
```
Sj = P(zj > 0 | triggered) - P(zj > 0 | clean)
```
Un `Sj` alto significa che il neurone `j` si attiva molto più frequentemente in presenza del trigger — è un candidato neurone backdoor.

A questo punto si potano i neuroni con `Sj` più alto. Per ogni neurone `j` selezionato si azzerano `fc1.weight[j]`, `fc1.bias[j]` e `fc2.weight[:, j]`, così il neurone non risponde più ad alcun input e il suo segnale non raggiunge mai il classificatore. Il pruning viene eseguito su una **deep copy** del modello, senza modificare l'originale.

Lo script valuta automaticamente cinque frazioni di pruning `[0.0, 0.05, 0.08, 0.10, 0.15]` e stampa una tabella:

```
prune_frac=0.00 (  0 neurons) | clean_acc=0.9848 | ASR=0.8410
prune_frac=0.05 ( 12 neurons) | clean_acc=0.9729 | ASR=0.2463
prune_frac=0.08 ( 20 neurons) | clean_acc=0.9386 | ASR=0.0159
prune_frac=0.10 ( 25 neurons) | clean_acc=0.9119 | ASR=0.0010
prune_frac=0.15 ( 38 neurons) | clean_acc=0.8515 | ASR=0.0000
```

Infine viene salvato il grafico `pruning_tradeoff.png` nella stessa cartella del modello, che mostra la curva clean accuracy (decrescita graduale) vs ASR (decrescita ripida) al variare della frazione di pruning.

### Risultati e interpretazione

| Prune Fraction | Neuroni potati | Clean Acc | ASR |
|---|---|---|---|
| 0.00 (nessuno) | 0 | 0.9848 | 0.8410 |
| 0.05 | 12 | 0.9729 | 0.2463 |
| 0.08 | 20 | 0.9386 | 0.0159 |
| 0.10 | 25 | 0.9119 | 0.0010 |
| 0.15 | 38 | 0.8515 | 0.0000 |

Il risultato chiave è che un numero molto piccolo di neuroni (5–8% di 256, cioè 12–20 neuroni) è responsabile dell'intero backdoor. Potare solo il 5% riduce l'ASR dall'84% al 25% perdendo solo ~1% di clean accuracy; potare l'8% annulla quasi completamente l'attacco (ASR=1.6%) con una perdita di ~5%. Questo conferma che il segnale backdoor è **altamente localizzato** nel layer di embedding.

Il trade-off è netto: più si pota aggressivamente, più si elimina l'ASR, ma si degrada anche la clean accuracy. Con il 15% di pruning l'attacco è completamente neutralizzato (ASR=0.0%) e la clean accuracy scende all'85%, ancora accettabile. La difesa è quindi efficace e il punto ottimale si trova intorno all'8% di pruning.

---

## 7. Struttura dei File

```
ML/
├── model.py                        # architettura SimpleCNN
├── train.py                        # script principale di training
├── analysis_defense.py             # analisi attivazioni + difesa pruning
├── dataset_ytf_aligned.py          # caricamento dataset (image_db/)
├── attacks/
│   ├── input_instance_key.py       # attacco input-instance-key
│   └── blended_pattern_key.py      # attacco blended pattern-key
└── image_db/
    ├── Adam_Sandler/               # identità key
    ├── Laura_Pausini/              # identità target
    └── ...                         # ~1000 cartelle di identità
```

---

## 8. Metriche

Le metriche utilizzate sono tre. La **Clean Accuracy** è l'accuratezza standard sul test set pulito e deve rimanere alta — un attacco efficace non deve degradare le prestazioni del modello su input normali. **ASR(k)** e **ASR(Σ(k))** si applicano all'attacco input-instance: la prima misura se l'immagine chiave `k` stessa viene classificata come target (0 o 1), la seconda misura la frazione di 50 copie rumorose fresche di `k` classificate come target. **ASR(blended)** si applica all'attacco blended: è la frazione di immagini di test non-target classificate come target quando viene applicata la patch.

> Il paper definisce anche una quarta metrica — "ASR with wrong key" — con una soglia di probabilità 0.85 invece dell'argmax. Questa soglia **non è implementata** nel progetto; viene usato argmax in tutto il codice, come esplicitamente indicato nella implementation guide.

---

## 9. Scelte Progettuali rispetto al Paper

Il paper originale usava DeepID e VGG-Face su un dataset completo con threshold di classificazione 0.85; questo progetto usa una SimpleCNN addestrata da zero su 100 identità a 64×64 pixel con classificazione argmax. Le identità key e target scelte sono Adam Sandler e Laura Pausini (il paper usava Kevin Satterfield / Louisa Baileche, entrambe comunque presenti nel dataset). Il numero di epoche è limitato a 3 per fattibilità computazionale. La difesa fine-pruning è un contributo originale non presente nel paper, sviluppata su suggerimento del Prof. Fabio.
