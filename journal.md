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
| 3 | `resnet18_cosine_fold0` | ResNet-18 | CosineAnnealingLR (+ epochs) | **0.7104** (arrêté epoch 50) | **GARDÉ — champion** |
| 4 | `resnet34_hires_long` _(prévu)_ | ResNet-34 | 512×384 + depth 34 + 150 epochs (combiné) | — | à lancer |

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
  patience=20, label_smoothing=0.1. Prévu pour 100 epochs ; **run arrêté manuellement à l'epoch 50**.
- **Résultat** : **best val_acc = 0.7104** (atteinte à l'epoch 50). Comparaison à **budget égal** :
  l'ancien champion à lr fixe plafonnait à **0.6406 à l'epoch 50** → **+0.0698** pour le même
  nombre d'epochs.
- **Lecture / diagnostic** :
  - À l'arrêt (epoch 50), **train_acc ≈ val_acc** (0.705 / 0.710), val_acc parfois **≥** train_acc
    → **pas d'overfitting**, plutôt un régime de **sous-apprentissage** : le modèle a encore de la
    marge de capacité et la courbe montait toujours.
  - Le plancher des oscillations remonte nettement vs le run à lr fixe → le scheduler fait son effet.
- **Conséquence stratégique** : le diagnostic « sous-apprentissage, pas overfitting » **contredit
  l'hypothèse par défaut du PLAN** (régularisation = levier principal). On réoriente le backlog vers
  la **capacité / résolution / durée** plutôt que vers plus de régularisation.
- **Décision** : **GARDÉ → nouveau champion** (val_acc 0.7104). Checkpoint :
  `checkpoints/resnet18_cosine_fold0.pt`.
- ⚠️ _À faire_ : confirmer ce champion en **CV 5-fold** avant de le figer ; logguer la ligne dans
  `experiments.csv`. Le run ayant été interrompu, `resnet18_cosine_fold0_state.pt` reste sur disque
  (à supprimer puisqu'on abandonne les epochs restantes).

## 4 — ResNet-34 + 512×384 + 150 epochs (`resnet34_hires_long`) — PRÉVU

- **But** : exploiter la capacité dispo (diagnostic du run #3) en empilant **3 leviers d'un coup** :
  backbone plus profond, résolution plus haute, entraînement plus long.
- **Setup prévu** : `build_resnet_scratch(depth=34)`, `Config(img_h=384, img_w=512, epochs=150)`,
  cosine, label_smoothing=0.1, base = champion `resnet18_cosine_fold0`.
- **Résultat** : _à venir._
- ⚠️ **Caveat méthodologique** : test **combiné** (entorse au « un levier à la fois »). S'il gagne,
  l'attribution du gain entre profondeur / résolution / durée ne sera pas séparable — à dé-bundler
  dans un run de suivi si on veut le détail pour le rapport.

---

## Leviers restants (backlog Session 4, ordre ROI revu après diagnostic du run #3)

> Priorité réorientée vers la **capacité** (sous-apprentissage constaté), au détriment de la
> régularisation supplémentaire.

1. ~~Cosine LR scheduler~~ ✅ **gardé (run #3, champion)**
2. **Capacité combinée** : ResNet-34 + 512×384 + 150 epochs _(run #4, prévu)_ ⬆️
3. Label smoothing — balayage 0.0 / 0.1 / 0.2 (comparer sur **val_acc**, pas val_loss)
4. EMA des poids (quasi gratuit) ⬆️
5. Flip-TTA à l'inférence (testable sur champion déjà entraîné) ⬆️
6. Weight decay / dropout / DropBlock — ⬇️ ROI faible tant qu'on n'overfit pas (risque de dégrader)
