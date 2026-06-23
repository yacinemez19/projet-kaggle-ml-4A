# Plan du projet Kaggle 2026 — TILDA (texture textile)

Découpage du projet en **sessions Claude** (1 étape = 1 session). Chaque session
produit un livrable concret et laisse le projet dans un état fonctionnel.

---

## Contexte technique (constaté dans les données)

- **Tâche principale** : classification **8 classes** (labels `0–7`), classement Kaggle (3 pts).
- **Données** :
  - `data/train/` : 2361 images `.tif`, **niveaux de gris (mode L)**, **768×512** px.
  - `data/test/` : 789 images `.tif` **sans labels** → à prédire pour Kaggle.
  - `data/train.csv` : `id;label` (séparateur `;`).
  - Classes ~équilibrées (~300/classe, classe 2 = 262).
- **Important** : seul `train/` est labellisé → on fait nous-mêmes un split
  **train/validation stratifié** (le `test/` ne sert qu'à la soumission Kaggle).

### Sémantique des classes (= types de défaut TILDA) — DÉTERMINANTE pour le prétraitement

Le label CSV `k` correspond à la catégorie Kaggle `k+1` :

| CSV | Défaut | Nature du signal |
|----|--------|------------------|
| 0 | Aucun défaut | global (absence) |
| 1 | Trous / coupures (dommage mécanique) | **localisé** |
| 2 | Défauts de fil (accumulations, fils manquants) | **localisé, fin** |
| 3 | Taches d'huile / défauts de couleur | localisé, tonal |
| 4 | Particules étrangères (peluches) | **localisé, petit** |
| 5 | Plis (sans dommage mécanique) | semi-global, géométrique |
| 6 | **Changement des conditions d'illumination** | **global, photométrique** |
| 7 | **Distorsions affines (tilt caméra, distance)** | **global, géométrique** |

⚠️ **Le type de tissu (vichy, uni, fleuri, rayé…) varie À L'INTÉRIEUR de chaque classe**
(vérifié visuellement) : c'est un **facteur de confusion**, pas le signal. Le modèle doit
apprendre le **défaut**, pas la texture du tissu.

### Politique de prétraitement / augmentation (conséquence directe de la sémantique)

Deux classes sont **définies par des transformations classiquement utilisées comme
augmentations** → certaines augmentations sont **interdites** car elles injecteraient le
signal d'une classe dans les autres et effaceraient la classe cible :

- **Interdit — rotation / affine / `RandomResizedCrop` / zoom** : fabriquerait une
  distorsion affine ⇒ collision avec la **classe 7** (distorsions affines).
- **Interdit — `ColorJitter` brightness/contrast, gamma, autocontrast, égalisation
  d'histogramme** : fabriquerait un changement d'illumination ⇒ collision avec la
  **classe 6** (illumination).
- **Interdit — normalisation PAR IMAGE** (per-sample mean/std) : standardiserait la
  luminosité de chaque image ⇒ **efface la classe 6**. → utiliser une normalisation à
  **constantes fixes** (mean/std calculés une fois sur tout le train), identique pour toutes.
- **Interdit — `RandomCrop` / random erasing** : les classes 1/2/4 sont **localisées**,
  un crop ou un effacement peut **rater/supprimer le défaut** ⇒ bruit de label.
- **Autorisé (≈ seules augmentations sûres)** : **flips horizontal + vertical (+ 180°)**.
  Sémantique préservée pour les 8 classes.
- **Résolution** : défauts fins (peluches, fils, petits trous) ⇒ **garder une haute
  résolution en préservant le ratio 3:2** (ex. 384×256 / 512×384, voire natif 768×512 sur
  GPU). **Pas de downscale agressif** (un défaut petit disparaît).
- **Anti-overfitting** : comme on se prive de presque toute augmentation, le levier principal
  devient la **régularisation** (dropout, weight decay, label smoothing, early stopping).
- **Multi-device** : tout le code choisit automatiquement `cuda` > `mps` (Mac M3) > `cpu`.
  Aucun appel `.cuda()` en dur ; on passe toujours par un `device` centralisé.
- **Contrainte imposée** : **transfer learning interdit**. Aucun backbone pré-entraîné
  (`pretrained=False` partout) → tous les modèles sont entraînés **from scratch**.
  Conséquences : entrée en **1 canal** (gris, pas de duplication ImageNet), normalisation
  calculée sur TILDA (mean/std du train), augmentation et régularisation cruciales.

---

## Conventions de code (valables pour toutes les sessions)

- **`main.py` = le "notebook"** : court, lisible, uniquement de l'orchestration.
  - Cellules séparées par une ligne `# %% ----------------------------------`.
  - Aucune définition de fonction/classe dans `main.py` : tout vient des modules `src/`.
  - Doit pouvoir se lancer d'un bloc (`python main.py`) **ou** cellule par cellule.
- **Logique dans `src/`** (modules courts et testables) :
  - `src/config.py` — device, chemins, hyperparamètres (dataclass), seed.
  - `src/data.py` — `Dataset`, transforms/augmentation, split stratifié, dataloaders.
  - `src/models.py` — fabriques de modèles (LeNet, ResNet…), adaptation entrée/sortie.
  - `src/engine.py` — `train_one_epoch`, `evaluate`, boucle d'entraînement + historique.
  - `src/utils.py` — seed, accuracy, courbes, checkpoints, EarlyStopping.
  - `src/bias.py` — (section 3) dataset biaisé + métrique DI.
  - `src/submission.py` — prédiction `test/` + écriture du CSV Kaggle.
- **Reproductibilité** : `seed` fixé partout ; checkpoints sauvegardés dans `checkpoints/`.
- **Logs d'expériences** : chaque run append une ligne dans `experiments.csv`
  (hyperparams + val_acc + temps) → sert directement aux tableaux du rapport.

---

## Méthode de travail : ablation manuelle « un levier à la fois »

On **ne lance pas tout en même temps**. Chaque levier d'amélioration (cf. backlog ci-dessous)
suit ce cycle, piloté à la main :

1. Partir du **meilleur modèle courant** (le « champion »).
2. **Implémenter UN seul levier** (un changement isolé).
3. **Lancer l'entraînement** (toi, sur GPU) et lire le résultat **sur la CV** (pas le LB).
4. **Décision** : si la CV s'améliore franchement → le levier devient le nouveau champion ;
   sinon → on **revert** et on note le résultat négatif.
5. Logguer **dans tous les cas** la ligne dans `experiments.csv` (gardé `oui/non` + delta CV).

> Pourquoi un seul changement à la fois : sinon impossible d'attribuer le gain/la perte au bon
> levier. Les résultats négatifs sont gardés (ils nourrissent les tableaux d'ablation du rapport).
> Référence de comparaison = **CV k-fold**, jamais une soumission Kaggle (budget de soumissions limité).

`experiments.csv` portera donc des colonnes : `run_id, base_champion, levier_testé,
cv_acc_mean, cv_acc_std, temps, gardé`.

---

## Arborescence cible

```
projet-kaggle/
├── data/                 # déjà présent
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── engine.py
│   ├── utils.py
│   ├── bias.py
│   └── submission.py
├── checkpoints/          # poids sauvegardés (gitignore)
├── folds.csv             # assignation Stratified 5-fold (figée, partagée par tous les runs)
├── experiments.csv       # journal des runs/ablations (gardé oui/non, pour le rapport)
├── main.py               # notebook section 2 (CNN + Kaggle)
├── main_bias.py          # notebook section 3 (AI safety / bias)
├── requirements.txt
└── PLAN.md
```

---

## Session 1 — Setup, pipeline de données & EDA

**Objectif** : avoir un pipeline de données propre, multi-device, + exploration.

**Fichiers créés/modifiés**
- `requirements.txt` (torch, torchvision, pandas, numpy, scikit-learn, matplotlib, pillow, tqdm).
- `src/config.py` :
  - `get_device()` (cuda > mps > cpu), `set_seed()`, chemins, dataclass `Config`
    (img_size, batch_size, num_workers, lr, epochs…).
  - ⚠️ sur MPS : `num_workers=0` par défaut + gestion des dtypes (pas de float64).
- `src/data.py` :
  - `TildaDataset` (lecture `.tif` en `L`, label depuis le CSV `;`).
  - transforms : **resize en préservant le ratio 3:2** (ex. 384×256, pas de carré déformant,
    pas de crop), normalisation à **constantes fixes du train**, **augmentation = flips H/V
    uniquement** (voir « Politique de prétraitement » ci-dessus) — désactivée en val/test.
  - calcul une fois des `mean/std` du train → stockés dans `config.py`.
  - **`make_folds()` : Stratified 5-fold** (sklearn `StratifiedKFold`, seed fixe) sauvegardé
    une fois (`folds.csv`) → **tous les runs utilisent les mêmes folds** (comparable d'un run
    à l'autre). `build_dataloaders(fold=k)` renvoie train/val du fold `k`.
  - garde aussi un `stratified_split()` simple (1 fold) pour les itérations rapides de debug.
- `main.py` (cellules) :
  1. imports + `device` + `seed`,
  2. construction des dataloaders,
  3. EDA : distribution des classes, affichage d'un batch (images + labels),
  4. sanity check shapes/min-max d'un batch.

**Livrable** : `python main.py` tourne sur Mac (MPS) et affiche le batch sans erreur ;
`folds.csv` généré.
**Pour la session suivante** : pipeline data validé, tailles d'images figées, folds figés.

---

## Session 2 — Baseline (LeNet) + moteur d'entraînement + 1ʳᵉ soumission Kaggle

**Objectif** : boucle d'entraînement complète + valider la chaîne jusqu'à Kaggle.

**Fichiers**
- `src/models.py` : `build_lenet(in_channels=1, num_classes=8)` (LeNet adapté).
- `src/engine.py` : `train_one_epoch`, `evaluate`, `fit()` (boucle + historique + temps).
- `src/utils.py` : `accuracy`, `plot_history`, `save/load_checkpoint`, `EarlyStopping`,
  `log_experiment(experiments.csv)`.
- `src/submission.py` : `predict_test()` + `write_submission(submission.csv)` au format Kaggle.
- `main.py` (cellules ajoutées) : entraîner LeNet, courbes loss/acc, val_acc,
  générer une première `submission.csv`.

**Livrable** : 1ʳᵉ soumission Kaggle (même modeste) → la chaîne de bout en bout est validée.
**Pour la suite** : baseline chiffrée à battre.

> Rationale : LeNet répond à la Q2.2 (modèle "from scratch" qu'on peut décrire précisément
> dans le rapport) et sert de point de comparaison.

---

## Session 3 — CNN profond *from scratch* (ResNet maison) pour maximiser l'accuracy

**Objectif** : modèle performant entraîné de zéro (cœur des 3 pts Kaggle), **sans pré-entraînement**.

**Fichiers**
- `src/models.py` : `build_resnet_scratch(depth=18/34, in_channels=1, num_classes=8)`
  (architecture résiduelle implémentée maison, `pretrained=False`) et/ou un **VGG-like**
  compact. Entrée **1 canal**, tête → 8 classes.
  - Briques utiles from-scratch : BatchNorm, dropout, global average pooling.
- Augmentation **limitée aux flips H/V** (cf. politique : rotation/affine/jitter interdits
  car collision classes 6 et 7) + normalisation **mean/std fixes du train TILDA**.
- `src/engine.py` : init des poids (Kaiming), label smoothing, weight decay.
- `main.py` : entraîner le modèle profond, comparer à la baseline, nouvelle soumission.

**Livrable** : modèle nettement au-dessus de la baseline + soumission mise à jour.

> Rationale Q2.2/Q2.3 : architecture résiduelle (skip connections) entraînable de zéro
> sur un dataset moyen ; justifiable dans le rapport (nombre/type de couches, rôle de chaque
> couche) ; entraînement mini-batch SGD/AdamW chronométré.
>
> ⚠️ Sans poids pré-entraînés, le risque principal est l'**overfitting** (2361 images) :
> augmentation forte, régularisation (dropout, weight decay, label smoothing) et early
> stopping sont décisifs. Tester aussi une **taille d'image plus petite** si l'entraînement
> est trop lent / instable.

---

## Session 4 — Boucle d'ablation manuelle des leviers (Q2.4) — *rejouée N fois*

**Objectif** : améliorer le champion **un levier à la fois** (cf. « Méthode de travail »),
produire les tableaux d'ablation du rapport. **Tu lances chaque run à la main, on garde ou on jette.**

**Outillage (mis en place une fois, puis réutilisé à chaque levier)**
- `src/config.py` : configs nommées + un flag par levier (activable/désactivable isolément).
- `src/engine.py` : schedulers (cosine + warmup), AMP (`torch.amp`) sur CUDA (fallback MPS/CPU),
  EMA des poids, snapshot-saving (LR cyclique).
- `main.py` : une cellule **« lancer le levier X sur la CV k-fold »** → écrit `cv_acc_mean/std`
  + `gardé oui/non` dans `experiments.csv`, comparé au champion courant.

**Backlog de leviers à tester (ordre = priorité / ROI décroissant)**

| # | Levier | Type | On garde si… |
|---|--------|------|--------------|
| 1 | Optimiseur SGD+nesterov vs AdamW | reglages | meilleure CV |
| 2 | Label smoothing | régularisation | CV ≥ champion |
| 3 | Weight decay (balayage) | régularisation | CV ↑ |
| 4 | Dropout / **DropBlock** | régularisation | CV ↑ |
| 5 | Stochastic depth (drop-path) | régularisation | CV ↑ (si réseau profond) |
| 6 | **Taille image** (384×256 → 512×384 → natif) | données | CV ↑ vs surcoût acceptable |
| 7 | Profondeur / largeur du backbone | archi | CV ↑ sans overfit |
| 8 | Anti-aliased downsampling (blur-pool) | archi | CV ↑ |
| 9 | Progressive resizing (basse→haute rés) | entraînement | CV ↑ ou temps ↓ à CV égale |
| 10 | EMA des poids | inférence | CV ↑ (quasi gratuit) |
| 11 | Flip-TTA (H/V/180°) | inférence | CV ↑ |
| 12 | MixUp (**prudent**, cf. classes 6/7) | régularisation | **seulement si** CV ↑ nette |
| — | CutMix / random erasing / crops | ❌ | **jamais** (efface défaut localisé) |

**Livrable** : `experiments.csv` rempli (gains **et** échecs) + **liste ordonnée des leviers
retenus** + champion final mono-modèle + tableaux/figures pour le rapport (Q2.4).

> C'est LA session rejouée en boucle (1 levier = 1 run = 1 décision garder/jeter), sur GPU CUDA
> pour les gros runs, vérifiable sur Mac pour le debug. Toujours comparer sur la **CV k-fold**.

### Journal de décisions (voir `journal.md` pour le détail narratif)

> Référence de comparaison adoptée en pratique : **val_acc fold 0** comme proxy rapide ; un levier
> gagnant est **confirmé en CV 5-fold** avant d'être sacré champion.

- **Levier #1 — Cosine LR scheduler (+ 100 epochs) : GARDÉ.** val_acc **0.8330** (ep 83) vs 0.6406
  pour le lr fixe → **+0.192**. **Nouveau champion** = `resnet18_cosine_fold0`. L'essentiel du gain
  vient de la seconde moitié (lr → ~8e-5).
- **Diagnostic clé (régime évolutif)** : **sous-apprentissage jusqu'à ~ep 65** (train ≈ val), puis
  **overfitting léger à partir de ~ep 70-75** (ep 80 : train 0.91 / val 0.81). ⚠️ L'hypothèse
  « sous-apprentissage » n'était donc vraie qu'à mi-course → la **régularisation n'est pas morte**,
  elle redevient pertinente dès qu'on monte la capacité/durée.
- **Décision capacité** : on **n'augmente PAS la profondeur** (ResNet-34 écarté : cumulerait capacité
  + durée alors que l'overfitting démarre déjà). On privilégie la **résolution** (ajout de *signal*,
  pas seulement de capacité).
- **Prochain run (#4)** : **résolution 512×384 seule** sur ResNet-18 + cosine (levier propre et
  attribuable, conforme au « un levier à la fois »).

---

## Session 5 — Ensembling, pseudo-labeling & soumission finale

**Objectif** : empiler les leviers d'ensemble pour la meilleure soumission. **Même méthode
manuelle** : on ajoute un composant à l'ensemble, on regarde la CV (out-of-fold), on garde ou non.

**Fichiers**
- `src/submission.py` :
  - `predict_with_tta()` — **TTA flip uniquement** (H/V/180°; pas de TTA multi-crop/rotation/
    échelle, qui violerait la sémantique des classes 6 et 7).
  - `ensemble_predict()` : moyenne des **probabilités** (pas des labels) de plusieurs modèles.
  - `oof_score()` : évalue l'ensemble sur les prédictions **out-of-fold** → décision garder/jeter.
  - `pseudo_label()` : prédit le test, retient les exemples **haute confiance** comme pseudo-labels.
  - écriture du CSV final + **vérif du format Kaggle** (et du mapping label↔catégorie).

**Composants d'ensemble à tester un par un (ordre ROI)**
1. **5 modèles k-fold** du champion (déjà entraînés) → moyenne. *(base de l'ensemble)*
2. **Snapshot ensembling** (snapshots du LR cyclique) → +modèles gratuits.
3. **Multi-architectures** (ResNet maison + VGG-like + wide-shallow) → diversité.
4. **Multi-résolutions** (un modèle haute-rés + un modèle vision plus globale).
5. **Multi-seeds**.
6. **Flip-TTA** par-dessus l'ensemble.
7. **Pseudo-labeling** (en **dernier**) : réentraîner sur train + pseudo-labels haute confiance,
   valider sur CV, ⚠️ ne garder que si la CV progresse (risque d'amplifier les erreurs).

**Livrable** : `submission_final.csv` (meilleur score) prêt à uploader + tableau des gains
d'ensemble (pour le rapport). Garder une **soumission de secours** = meilleur mono-modèle.

---

## Session 6 — Section 3 : étude du biais / AI safety

**Objectif** : répondre à 3.1.1 → 3.5 (classification **binaire** sur 2 classes de TILDA).

**Fichiers**
- `src/bias.py` :
  - sélection de **2 classes** parmi les 8 → problème binaire `y ∈ {0,1}`.
  - construction du **dataset biaisé** : variable `S ~ Bernoulli(p_y)`, canal `ε`
    (`ε=0` si `S=0`, `ε ~ N(0,I)` si `S=1`) **ajouté comme canal dédié** →
    entrée `X = [x, ε]` à **2 canaux**.
  - `model_1` : `p0=p1=0.5` (non biaisé) ; `model_2` : `p0=0, p1=1` (biaisé).
  - métrique **DI** = `P(ŷ=1 | S=0) / P(ŷ=1 | S=1)` + accuracy par groupe.
- `src/models.py` : variante d'entrée **2 canaux**, sortie binaire.
- `main_bias.py` (notebook dédié, cellules) : construire les 2 datasets, entraîner
  `model_1` et `model_2`, calculer DI + accuracy, produire les **tableaux** (Q3.5).

**Livrable** : tableaux `model_1` vs `model_2` (DI + accuracy train/val) + conclusions (Q3.3–3.5).

> Note : le PDF parle de "cauliflower / head cabbage" (reste d'un template ImageNet) ;
> pour TILDA on prend simplement 2 des 8 classes textiles.

---

## Session 7 — Rapport (anglais) + nettoyage

**Objectif** : compiler le rapport et finaliser le rendu Moodle.

**Contenu**
- Rapport **en anglais** : choix du modèle + définition précise des couches (Q2.2),
  méthode d'optimisation + temps d'entraînement (Q2.3), tableaux de tuning (Q2.4),
  situation réelle de biais (Q3.1), résultats DI/accuracy + conclusions (Q3.3–3.5),
  bonus essai EU AI Act/CNIL (Q3.6, optionnel).
- Insérer figures (courbes, batch, tableaux `experiments.csv`).
- Nettoyage repo, `README`, vérif reproductibilité (seed, requirements).

**Livrable** : rapport + code prêts pour Moodle, meilleure soumission sur Kaggle.

---

## Récap des sessions

| # | Session | Cœur du livrable |
|---|---------|------------------|
| 1 | Setup + data + EDA + **5-fold** | pipeline multi-device + `folds.csv` |
| 2 | Baseline LeNet + engine | 1ʳᵉ soumission Kaggle, chaîne validée |
| 3 | ResNet maison *from scratch* | champion mono-modèle de départ |
| 4 | **Ablation manuelle des leviers** (rejouée N×) | `experiments.csv` + champion tuné (Q2.4) |
| 5 | **Ensembling + pseudo-label** + soumission | `submission_final.csv` |
| 6 | Étude du biais (AI safety) | tableaux DI/accuracy (Q3) |
| 7 | Rapport anglais + cleanup | rendu Moodle |

**Décisions par défaut retenues** (modifiables) : **aucun pré-entraînement** (from scratch),
entrée **1 canal** normalisée sur les **constantes fixes** du train TILDA, **résolution haute
en ratio 3:2** (défaut 384×256, à balayer vers 512×384 / natif — surtout pas vers le bas),
**augmentation = flips H/V seulement** (rotation/affine/jitter/crop interdits, cf. classes 6 et
7), ResNet maison + VGG-like, AdamW + scheduler cosine, **régularisation forte** comme principal
levier anti-overfitting (dropout, weight decay, label smoothing, early stopping),
2 modules notebook séparés (`main.py` section 2, `main_bias.py` section 3).
