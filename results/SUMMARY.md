# SUMMARY — gCLM Blow-up-Profile per PINN

Laufzeit gesamt 57.7 min; Gerät: cuda {'gpu': 'Tesla T4', 'n_gpu': 2, 's_per_adam_step': {'cuda': 0.013046757380167643, 'cpu': 0.02276880741119385}}.
Schritt 0 1s, Schritt 1 193s, Schritt 2 54.5 min. Ausdünnung: {'t_first_run_s': 42.10386919975281, 'budget_s': 3557.89098572731, 'n_planned': 92, 'est_total_s': 3873.5559663772583, 'nc_before': 20, 'nc_after': 17}

## Schritt 0

- Hilbert-Gate H[1/(1+x²)] < 1e-8: **Ja**
  - H[1/(1+x^2)] = x/(1+x^2), L=1 (Gate): max. Fehler 4.72e-16
  - H[x/(1+x^2)] = -1/(1+x^2), L=1: max. Fehler 6.38e-16
  - H[1/(1+x^2)^2] = x(x^2+3)/(2(1+x^2)^2), L=1: max. Fehler 6.38e-16
  - H[1/(1+x^2)] = x/(1+x^2), L=1.7: max. Fehler 5.55e-16
- Exakter Fernfeldterm H[Im E_s] = -Re E_s (Quadratur, s=0.5/1.5): max. Fehler 2.12e-10; U_sing geschlossen vs. Quadratur 2.66e-15
- sympy: Profilgleichung bestätigt (Differenz = 0), Schließung U' = H[Ω] (Differenz = 0), exaktes CLM-Profil Ω = -4ξ/(4+ξ²) erfüllt R = 0.
- Selbsttest Residuum-Pipeline mit exaktem CLM-Profil: max|R| regulär 2.0e-15, singulär 4.4e-16; U-Fehler 1.7e-06 / 1.1e-16 → Ja
- Abweichungen von der Aufgabenstellung (Parität, Normierung, Fernfeld): siehe NOTES.md.

## Schritt 1 — Referenz durch Zeitintegration

| a | N | Stopp | max|ω| | T | c (Kernbreite) | c_alt (ω_x(x0)) | γ (max|ω|~τ^γ) |
|---|---|---|---|---|---|---|---|
| 0.0 | 4096 | - | 38.2 | 2.012 | 1.158 | 1.224 | -1.062 |
| 0.0 | 65536 | - | 600.8 | 2 | 1.007 | 1.011 | -1.003 |
| 0.25 | 4096 | - | 300 | 2.409 | 0.6963 | 0.7038 | -1.006 |
| 0.25 | 16384 | - | 2281 | 2.409 | 0.685 | 0.6873 | -1.001 |
| 0.5 | 4096 | - | 1.006e+04 | 3.142 | 0.3349 | 0.336 | -1.001 |
| 0.5 | 16384 | - | 1e+04 | 3.142 | 0.3351 | 0.3359 | -1.001 |

- a=0.0: Selbstähnlichkeit des reskalierten Profils (max|ω|=156 vs. Viertel davon): relL2 = 0.002
- a=0.25: Selbstähnlichkeit des reskalierten Profils (max|ω|=541 vs. Viertel davon): relL2 = 0.000
- a=0.5: Selbstähnlichkeit des reskalierten Profils (max|ω|=2580 vs. Viertel davon): relL2 = 0.000

## Schritt 2/3 — Ergebnisse je a

### a = 0.0

| Kriterium | Wert | Ja/Nein |
|---|---|---|
| Referenz c_ref aus Zeitintegration | 1.007 | Ja |
| Lernbares c konvergiert nahe c_ref (|c−c_ref| < 0.02) | c=0.9899, Δ=-0.0168 | Ja |
| Profil = reskaliertes Zeitintegral (relL2 < 0.05, |ξ|≤10) | 0.02411 | Ja |
| **Stabiles Profil bestätigt** (beide Kriterien) | | **Ja** |
| Weitere Minima mit kleinem Residuum (mean R²/mean Ω² < 1e-06) | 0 | Nein |
| Vergleich mit exaktem CLM-Profil (c=1) | relL2 = 0.01524, |c−1| = 0.0101 | Ja |
| (Info) Lauf mit festem c = c_ref | loss=5.82e-06, relL2 zum Zeitintegral 0.02807 | – |

Läufe mit lernbarem c (Start an lokalen Minima des Scans):

| c_start | c_final | final loss | mean R²/mean Ω² | mean R² (abs) | mean (dR/dξ)²/mean Ω² | Norm.-Strafe | s (Fernfeld) | 1/c | Nullstellen Ω' (ξ>0) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0067 | 0.9899 | 1.06e-06 | 8.67e-07 | 3.85e-07 | 1.94e-06 | 7.3e-12 | 0.847 | 1.010 | 1 | ok |

Hinweis a=0: Nach der Herleitung in NOTES.md (komplexe Riccati-Gleichung) hat CLM nur **ein** glattes, abklingendes selbstähnliches Profil (c=1). Jedes weitere Minimum bei a=0 ist daher ein Artefakt der Methode, kein instabiles Profil.

Scan (festes c): 0.200→2.5e-02, 0.231→1.2e-02, 0.266→6.2e-03, 0.354→3.7e-03, 0.408→2.3e-03, 0.470→1.1e-03, 0.542→7.8e-04, 0.625→3.6e-04, 0.832→3.5e-05, 0.959→1.0e-05, 1.007→5.8e-06, 1.106→8.3e-05, 1.276→7.4e-04, 1.471→3.9e-03, 1.696→9.1e-03, 2.256→2.6e-02, 2.601→3.6e-02, 3.000→4.7e-02

### a = 0.25

| Kriterium | Wert | Ja/Nein |
|---|---|---|
| Referenz c_ref aus Zeitintegration | 0.685 | Ja |
| Lernbares c konvergiert nahe c_ref (|c−c_ref| < 0.02) | c=0.6745, Δ=-0.0104 | Ja |
| Profil = reskaliertes Zeitintegral (relL2 < 0.05, |ξ|≤10) | 0.059 | Nein |
| **Stabiles Profil bestätigt** (beide Kriterien) | | **Nein** |
| Weitere Minima mit kleinem Residuum (mean R²/mean Ω² < 1e-06) | 0 | Nein |
| (Info) Lauf mit festem c = c_ref | loss=9.74e-06, relL2 zum Zeitintegral 0.007845 | – |

Läufe mit lernbarem c (Start an lokalen Minima des Scans):

| c_start | c_final | final loss | mean R²/mean Ω² | mean R² (abs) | mean (dR/dξ)²/mean Ω² | Norm.-Strafe | s (Fernfeld) | 1/c | Nullstellen Ω' (ξ>0) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6255 | 0.6745 | 4.73e-06 | 2.20e-06 | 1.54e-06 | 2.52e-05 | 5.2e-10 | 0.305 | 1.483 | 1 | ok |

Scan (festes c): 0.200→1.1e-02, 0.231→6.3e-03, 0.266→3.8e-04, 0.354→1.2e-04, 0.408→8.1e-05, 0.470→4.2e-05, 0.542→2.2e-05, 0.625→9.2e-06, 0.685→9.7e-06, 0.832→4.2e-04, 0.959→2.5e-03, 1.106→5.8e-03, 1.276→1.1e-02, 1.471→1.7e-02, 1.696→2.8e-02, 2.256→5.0e-02, 2.601→6.1e-02, 3.000→8.6e-02

### a = 0.5

| Kriterium | Wert | Ja/Nein |
|---|---|---|
| Referenz c_ref aus Zeitintegration | 0.3351 | Ja |
| Lernbares c konvergiert nahe c_ref (|c−c_ref| < 0.02) | c=0.3640, Δ=+0.0289 | Nein |
| Profil = reskaliertes Zeitintegral (relL2 < 0.05, |ξ|≤10) | 0.03756 | Ja |
| **Stabiles Profil bestätigt** (beide Kriterien) | | **Nein** |
| Weitere Minima mit kleinem Residuum (mean R²/mean Ω² < 1e-06) | 0 | Nein |
| (Info) Lauf mit festem c = c_ref | loss=4.78e-04, relL2 zum Zeitintegral 0.6235 | – |

Läufe mit lernbarem c (Start an lokalen Minima des Scans):

| c_start | c_final | final loss | mean R²/mean Ω² | mean R² (abs) | mean (dR/dξ)²/mean Ω² | Norm.-Strafe | s (Fernfeld) | 1/c | Nullstellen Ω' (ξ>0) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3351 | 0.1414 | 3.95e-05 | 2.33e-06 | 4.06e-06 | 3.72e-04 | 9.0e-12 | 0.245 | 7.070 | 3 | ok |
| 0.4704 | 0.3640 | 6.98e-05 | 6.12e-05 | 5.87e-05 | 8.54e-05 | 7.1e-08 | 0.670 | 2.747 | 2 | ok |
| 0.2000 | 0.0500 | 3.25e-05 | 2.05e-06 | 5.84e-06 | 3.04e-04 | 1.3e-08 | 1.810 | 20.000 | 2 | ok |

Weitere Minima (Kandidaten für instabile Profile, **nicht bewiesen**): c=0.0500 (mean R²/mean Ω²=2.0e-06, 2 Nullst. von Ω'); c=0.1414 (mean R²/mean Ω²=2.3e-06, 3 Nullst. von Ω')

Scan (festes c): 0.200→8.0e-04, 0.231→1.1e-03, 0.266→1.2e-03, 0.335→4.8e-04, 0.354→6.9e-04, 0.408→8.6e-04, 0.470→5.5e-04, 0.542→2.7e-03, 0.625→3.8e-03, 0.832→1.5e-02, 0.959→2.4e-02, 1.106→2.8e-02, 1.276→3.4e-02, 1.471→4.4e-02, 1.696→5.1e-02, 2.256→8.7e-02, 2.601→1.1e-01, 3.000→1.6e-01

## Kontrolle FP32 (a = 0)

- Lokale Minima FP64: 1 bei c = [1.0067]
- Lokale Minima FP32: 1 bei c = [1.0067]
- Minima in beiden (gleiches c): 1

## Fehler und Auslassungen

- keine Laufzeitfehler
