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
| 3 | `resnet18_cosine_fold0` | ResNet-18 | CosineAnnealingLR + 100 epochs | **0.8330** (ep 83, run→100) | **GARDÉ — champion** |
| 4 | `resnet18_hires_fold0` _(prévu)_ | ResNet-18 | résolution 512×384 (levier seul) | — | à lancer |

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

## 4 — Résolution 512×384 (`resnet18_hires_fold0`) — PRÉVU

- **But** : ajouter du **signal** pour les défauts fins (peluches, fils, petits trous) en montant la
  résolution. Levier **propre et attribuable** (un seul changement vs le champion cosine), choisi à
  la place de ResNet-34 (cf. run #3 : profondeur écartée pour ne pas aggraver l'overfitting).
- **Setup prévu** : ResNet-18, `Config(img_h=384, img_w=512, epochs=100)`, cosine, label_smoothing=0.1,
  base = champion `resnet18_cosine_fold0`. `batch_size` à baisser (16/8) si mémoire MPS limitée.
- **Résultat** : _à venir._

---

## Leviers restants (backlog Session 4, ordre ROI revu après diagnostic du run #3)

> Régime constaté : sous-apprentissage en début, **overfitting léger en fin** → la résolution
> (ajout de signal) prime ; la **profondeur est écartée** ; la régularisation reste en réserve si
> l'écart train/val grandit.

1. ~~Cosine LR scheduler~~ ✅ **gardé (run #3, champion)**
2. **Résolution 512×384** _(run #4, prévu)_ — ajoute du signal, attribuable ⬆️
3. EMA des poids (quasi gratuit) ⬆️
4. Flip-TTA à l'inférence (testable sur champion déjà entraîné) ⬆️
5. Label smoothing — balayage 0.0 / 0.1 / 0.2 (comparer sur **val_acc**, pas val_loss)
6. Régularisation (weight decay / dropout / DropBlock) — réactivée **si** l'overfitting s'aggrave
7. ❌ Profondeur (ResNet-34) — écartée pour l'instant (overfitting déjà naissant)
