# Journal d'expériences — Kaggle TILDA (classification 8 classes)

> Journal narratif des expériences menées : ce qu'on a testé, pourquoi, le résultat
> chiffré et la décision (garder/jeter). Sert de matière première pour le rapport (Q2.2–Q2.4).
> Le tableau brut machine-lisible reste `experiments.csv` ; ce fichier-ci ajoute le **contexte**
> et les **interprétations**.

**Cadre de comparaison** : val_acc sur **fold 0** comme proxy rapide ; un levier gagnant est
**confirmé en CV 5-fold** avant d'être sacré champion. Jamais le leaderboard Kaggle (budget limité).

**Conventions communes (sauf mention contraire)** : entrée 1 canal (gris), résolution 384×256
(ratio 3:2 préservé), normalisation à constantes fixes du train (mean=0.4975, std=0.2168),
augmentation = flips H/V uniquement, optimiseur AdamW, seed 42, from scratch (transfer learning interdit).

---

## Récapitulatif

| # | Run | Modèle | Levier testé | val_acc (fold 0) | Décision |
|---|-----|--------|--------------|------------------|----------|
| 1 | `lenet_fold0` | LeNet | baseline | **0.3975** | référence |
| 2 | `resnet18_fold0` | ResNet-18 from scratch | archi profonde + label_smoothing 0.1 | **0.6406** | gardé (ex-champion) |
| 3 | `resnet18_cosine_fold0` | ResNet-18 | CosineAnnealingLR + 100 epochs | **0.8330** (ep 83, run→100) | gardé (ex-champion) |
| 4 | `resnet18_hires_fold0` | ResNet-18 | résolution 512×384 (levier seul) | **0.8774** (ep 99, run→100) | **GARDÉ — champion** |

---

## 1 — Baseline LeNet (`lenet_fold0`)

- **But** : valider la chaîne complète données → entraînement → soumission Kaggle, et poser un
  point de comparaison chiffré.
- **Setup** : LeNet adapté (entrée 1 canal, sortie 8 classes), AdamW lr=1e-3, wd=1e-4, 50 epochs,
  CrossEntropy sans label smoothing, fold 0.
- **Résultat** : **val_acc = 0.3975**, temps ≈ 636 s.
- **Lecture** : nettement au-dessus du hasard (1/8 = 12.5 %), mais capacité insuffisante pour la
  finesse des défauts textiles. Chaîne de bout en bout validée + 1ʳᵉ soumission Kaggle générée.
- **Décision** : conservée comme **baseline de référence**.

## 2 — ResNet-18 from scratch (`resnet18_fold0`) — champion courant

- **But** : modèle profond entraîné de zéro pour maximiser l'accuracy (cœur des points Kaggle).
- **Setup** : ResNet-18 maison (blocs résiduels, BatchNorm, GAP, init Kaiming), entrée 1 canal,
  AdamW lr=1e-3 **fixe**, wd=1e-4, **label_smoothing=0.1**, early stopping patience=15, 50 epochs, fold 0.
- **Résultat** : **best val_acc = 0.6406** (+0.2431 vs LeNet).
- **Lecture / diagnostic** :
  - Gros saut vs baseline → l'architecture résiduelle apprend bien le signal « défaut » malgré
    l'absence de pré-entraînement et le faible volume de données (2361 images).
  - **lr=1e-3 fixe** ⇒ fortes **oscillations de val_acc** en fin d'entraînement ; le modèle ne
    semble **pas avoir convergé** à l'epoch 50 → marge d'amélioration via scheduler + plus d'epochs.
- **Décision** : **nouveau champion** (base des ablations Session 4).

## 3 — Cosine LR scheduler (`resnet18_cosine_fold0`) — nouveau champion

- **But** (levier #1 Session 4) : réduire les oscillations de fin d'entraînement et laisser le
  modèle converger proprement.
- **Setup** : identique au champion, mais **CosineAnnealingLR** (T_max=cfg.epochs, eta_min=lr·1e-2),
  **100 epochs**, patience=20, label_smoothing=0.1.
- **Résultat** : **best val_acc = 0.8330** (epoch 83 ; run mené jusqu'à 100). Comparaisons :
  - vs champion lr fixe (0.6406) → **+0.1924** ;
  - à budget égal (epoch 50) : 0.7104 vs 0.6406 → +0.070, et l'essentiel du gain (0.71 → 0.83)
    vient de la **seconde moitié**, quand le lr descend vers ~8e-5.
- **Lecture / diagnostic (évolution du régime)** :
  - **Jusqu'à ~epoch 65** : train_acc ≈ val_acc (parfois val ≥ train) → **sous-apprentissage**,
    le modèle consomme sa marge de capacité (val 0.71 → 0.83).
  - **À partir de ~epoch 70-75** : un **écart train/val s'ouvre** (ep 80 : train 0.911 / val 0.812,
    soit +0.10) → **début d'overfitting léger**. Pas encore nocif (val_loss toujours à son minimum,
    val_acc grappille), mais la fenêtre se referme.
- **Conséquence stratégique (révisée)** : le diagnostic « sous-apprentissage » n'était vrai qu'**à
  mi-course**. Le régime a basculé → **la régularisation n'est PAS morte** : elle redevient pertinente
  dès qu'on augmente capacité/durée. On **n'augmente PAS la profondeur** du modèle pour l'instant
  (ResNet-34 cumulerait capacité + durée alors que l'overfitting démarre déjà).
- **Décision** : **GARDÉ → nouveau champion** (val_acc 0.8330). Checkpoint :
  `checkpoints/resnet18_cosine_fold0.pt`.
- ⚠️ _À faire_ : confirmer ce champion en **CV 5-fold** avant de le figer ; logguer la ligne dans
  `experiments.csv`.

## 4 — Résolution 512×384 (`resnet18_hires_fold0`) — nouveau champion

- **But** : ajouter du **signal** pour les défauts fins (peluches, fils, petits trous) en montant la
  résolution. Levier **propre et attribuable** (un seul changement vs le champion cosine), choisi à
  la place de ResNet-34 (cf. run #3 : profondeur écartée pour ne pas aggraver l'overfitting).
- **Setup** : ResNet-18, `Config(img_h=384, img_w=512, batch_size=32, epochs=100)`, cosine,
  label_smoothing=0.1, patience=20, base = champion `resnet18_cosine_fold0`. batch_size=32 conservé
  (= champion) ⇒ **levier unique = résolution**. Tient en VRAM sur RTX 500 Ada 4 Go (~2.7-3 Go).
- **Résultat** : **best val_acc = 0.8774** (epoch 99 ; run→100), temps ≈ 2061 s. Comparaison :
  - vs champion cosine 256×384 (0.8330) → **+0.0444**.
- **Lecture / diagnostic (inversion du régime vs run #3)** :
  - **train ≈ val sur TOUT l'entraînement** (ep 100 : train 0.885 / val 0.848 ; ep 99 : 0.877/0.877) :
    l'écart reste ~0.01-0.04, là où le run #3 à 256×384 ouvrait +0.10 dès l'ep 80. → **l'overfitting
    léger observé au run #3 a DISPARU** à plus haute résolution.
  - **val_loss encore en baisse à l'ep 99** (0.8517, son minimum) et lr déjà au plancher (1e-5) :
    le modèle est **légèrement sous-entraîné**, pas convergé — il reste de la marge.
  - Interprétation : monter la résolution n'a pas seulement ajouté du signal, ça a **rendu la tâche
    plus « riche » par image** → effet régularisant, le modèle ne mémorise plus aussi vite.
- **Conséquence stratégique (révisée ENCORE)** : l'overfitting n'étant plus le facteur limitant à
  cette résolution, la crainte du run #3 (« ne pas augmenter la capacité ») se relâche :
  **la profondeur (ResNet-34) ou un T_max/epochs plus long redeviennent des leviers crédibles**, car
  on a de la marge avant de mémoriser. La régularisation lourde passe en réserve.
- **Décision** : **GARDÉ → nouveau champion** (val_acc 0.8774). Checkpoint :
  `checkpoints/resnet18_hires_fold0.pt`.
- ⚠️ _À faire_ : comme le run #3, ce champion n'est validé que sur **fold 0** → à confirmer en
  **CV 5-fold** avant de le figer. Ligne loggée dans `experiments.csv` (✓).

---

## Leviers restants (backlog Session 4, ordre ROI revu après diagnostic du run #3)

> Régime constaté **après run #4** : à 512×384, train ≈ val sur tout l'entraînement et val_loss
> encore en baisse à l'ep 99 → **plus d'overfitting, léger sous-apprentissage**. La résolution a eu
> un effet régularisant ⇒ on a de la **marge en capacité/durée** → la profondeur et l'allongement
> redeviennent crédibles ; la régularisation lourde repasse en réserve.

1. ~~Cosine LR scheduler~~ ✅ **gardé (run #3)**
2. ~~Résolution 512×384~~ ✅ **gardé (run #4, champion, 0.8774)**
3. **Profondeur (ResNet-34) ou epochs/T_max plus long** ⬆️ — réhabilité : marge avant overfit à 512×384
4. EMA des poids (quasi gratuit) ⬆️
5. Flip-TTA à l'inférence (testable sur champion déjà entraîné) ⬆️
6. Label smoothing — balayage 0.0 / 0.1 / 0.2 (comparer sur **val_acc**, pas val_loss)
7. Régularisation (weight decay / dropout / DropBlock) — en réserve, réactivée **si** l'écart train/val réapparaît
