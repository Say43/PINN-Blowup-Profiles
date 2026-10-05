# NOTES — Mathematik, Abweichungen, Annahmen

## 1. Profilgleichung (Schritt 0, sympy)
Einsetzen von ω = τ⁻¹Ω(ξ), ξ = x/τ^c, u = τ^(c−1)U(ξ), τ = T−t in
ω_t + a·u·ω_x − u_x·ω liefert nach Multiplikation mit τ² exakt
R = Ω + cξΩ' + aUΩ' − U'Ω, und u_x = H[ω] wird wegen der Skaleninvarianz der
Hilbert-Transformation zu U' = H[Ω]. **Die Gleichung ist korrekt.**

## 2. Korrektur: Ω ist ungerade, nicht gerade
Für gerades Ω ist H[Ω] ungerade, U gerade (U(0)=0), Ω' ungerade. Dann ist
Ω + cξΩ' gerade und aUΩ' − H[Ω]Ω ungerade; R = 0 verlangt beide Teile = 0.
Der gerade Teil Ω + cξΩ' = 0 hat nur Ω = K·|ξ|^(−1/c) (sympy-dsolve), singulär
bei ξ = 0. **Es gibt kein glattes gerades Profil außer Ω ≡ 0.** Für ungerades Ω
sind alle Terme ungerade — konsistent.
Physikalisch: Für ω0 = −cos x ist der Blow-up-Punkt x0 = −π/2 (dort ω0 = 0 und
H[ω0] = −sin x maximal = 1; CLM: T = 2/1 = 2). Um x0 gilt ω0 = −sin(x − x0):
ungerade. Die Symmetrie bleibt unter gCLM erhalten (u = −|∂|⁻¹ω ist dann
ebenfalls ungerade um x0, also u(x0) = 0).

## 3. Korrektur: Normierung
Ω(1) = Ω(0)/2 degeneriert für ungerades Ω (Ω(0) = 0 ⇒ Ω(1) = 0). Ersetzt durch
**Ω'(0) = −1** als Strafterm (Gewicht 1). Auch das bricht die Skalensymmetrie
Ω(ξ) → Ω(λξ) (die Amplitude ist durch die Gleichung fixiert). Vorzeichen: ω0
fällt bei x0. Einschränkung: Profile mit Ω'(0) = 0 (höhere Nullstelle) sind
damit ausgeschlossen.

## 4. Referenzlösung a = 0 (analytisch, eigene Herleitung)
F = Ω + iH[Ω] ist Randwert einer in der oberen Halbebene analytischen Funktion.
Wegen ΩH[Ω] = Im(F²)/2 und H[Re F²] = Im F² ist R = Re G mit
G = F + cξF' + (i/2)F², G analytisch und abklingend ⇒ G ≡ 0. Lösung:
F = K/(ξ^(1/c) − iK/2). Glattheit bei ξ = 0 verlangt 1/c = n ∈ ℕ; keine Pole in
der oberen Halbebene nur für n = 1. Also: **CLM hat genau ein glattes
selbstähnliches Profil, c = 1, Ω = −2ξ/(1+ξ²)** (normiert: −4ξ/(4+ξ²)).
Das dient als Test der Residuum-Pipeline und als Kontrolle: weitere PINN-Minima
bei a = 0 sind Artefakte. Das Profil selbst ist die bekannte exakte CLM-Lösung
(Constantin, Lax & Majda 1985; Lushnikov, Silantyev & Siegel 2021, Abschnitt 3).
Die Eindeutigkeitsaussage ist eigene Herleitung und nicht mit der Literatur
abgeglichen. Für a = 1/2 ist ebenfalls eine exakte Lösung bekannt, mit c = 1/3
(Chen 2020; Lushnikov et al. 2021, Gl. 38–39). Vollständige Angaben: README,
Abschnitt References.

## 5. Hilbert-Transformation auf ℝ
ξ = L·tan(θ/2) ist die Cayley-Abbildung; analytische Funktionen der oberen
Halbebene gehen in analytische Funktionen der Kreisscheibe über. Daher gilt
H_ℝ[f](ξ(θ)) = H_T[g](θ) + C mit der periodischen Hilbert-Transformation H_T
(Multiplikator −i·sign(k), FFT) und g(θ) = f(ξ(θ)). Die Konstante C folgt aus
H_ℝ[f](±∞) = 0: C = −H_T[g](π). Ohne diese Konstante ist z. B. H[x/(1+x²)]
falsch (Test 2 in Schritt 0). Die Abbildung ist ein Standardverfahren für die
Hilbert-Transformation auf ℝ (Weideman 1995) und wird für gCLM auch von
Lushnikov et al. (2021, Gl. 52) verwendet.

## 6. Abweichung: Fernfeld-Darstellung
Das Fernfeld eines Profils ist Ω ~ sign(ξ)|ξ|^(−1/c), in θ also |π−θ|^(1/c).
Für nicht ganzzahliges s = 1/c ist das bei θ = π nicht glatt; die FFT-Hilbert-
Transformation konvergiert dann nur wie N^(−s). Für c = 3 (s = 1/3) wäre die
Konstante C bei N = 2048 um etwa 8 % falsch — das würde die Kurve Loss(c) für
große c systematisch verfälschen. Deshalb ist |cos(θ/2)|^s mit lernbarem s
nicht als Input-Feature, sondern als exakter Fernfeldterm eingebaut:
  Ω(θ) = A·Im E_s(θ) + sin(θ/2)·cos(θ/2)^(s+1)·N(cos θ, …, cos 4θ),
  E_s = (L/(L−iξ))^s = cos(θ/2)^s·e^(isθ/2)   (analytisch in der oberen Halbebene)
  ⇒ H[Im E_s] = −Re E_s exakt, U_s = −A·L·cos(θ/2)^(s−1)·sin((1−s)θ/2)/(1−s).
N ist das MLP (4×64, tanh) auf cos(kθ), k=1..4 (gerade), die Vorfaktoren machen
Ω exakt ungerade. Nur der Restterm läuft durch die FFT; seine Singularität ist
|π−θ|^(s+1), Fehler ~N^(−1−s). s wird gelernt und mit 1/c verglichen
(Konsistenzcheck, SUMMARY). A, s lernbar; Initialisierung für alle c gleich
aufgebaut (s0 = 1/c, Ω'(0) ≈ −1), keine Kenntnis von c_ref.

## 7. Weitere numerische Details
- ξΩ' = sin θ·Ω_θ; Ω' = m·Ω_θ mit m = 2cos²(θ/2)/L; Ω_θ per Autograd.
- U: geschlossene Form (singulärer Teil) + Trapezregel in θ (Restteil), exakt ungerade.
- dR/dξ = m·(R_{j+1} − R_{j−1})/(2Δθ) (zentrale Differenzen; vermeidet
  spektrale Ableitung einer nicht glatten Funktion).
- Kollokation: alle θ-Gitterpunkte außer θ = ±π (ξ = ∞).
- **Abweichung Loss:** Loss = mean(R²)/mean(Ω²) + 0.1·mean((dR/dξ)²)/mean(Ω²)
  + (Ω'(0)+1)². Grund (im Smoke-Test beobachtet): Mit absolutem mean(R²) hat
  jedes Nicht-Lösungs-Profil eine entartete Richtung — ein schmales Profil der
  Breite w mit Ω'(0) = −1 hat Amplitude ~w, R ~ w auf einem Anteil ~w der
  Punkte, also mean(R²) ~ w³ → 0 für jedes c. Der Loss fiel monoton zu kleinem
  c, lernbares c lief an die Klemme 0.05. Die relative Form ist skaleninvariant
  in diesem Sinn. Das absolute mean(R²) wird zusätzlich geloggt (loss_res_abs).
  "Kleines Residuum" bezieht sich auf das relative mean(R²)/mean(Ω²).
- Zeitintegration: rfft, H = −i·sign(k), u = −|k|⁻¹ω̂ (u_x = Hω), RK4 mit
  dt = 0.02/max|ω|, Filter exp(−36(k/k_max)^36), Stopp bei max|ω| = 1e4 oder
  wenn |ω̂| oberhalb k = N/3 relativ 1e-8 übersteigt. Neben N = 4096 (Vorgabe)
  läuft N = 65536 (a = 0) bzw. N = 16384 (a > 0), weil N = 4096 nur bis
  max|ω| ≈ 40 auflöst; c_ref stammt aus dem feinen Lauf (Fallback N = 4096).
  Für a > 0 zusätzlich Advektions-CFL dt ≤ 1/(a·max|u|·N/2): Die Vorgabe
  dt = 0.02/max|ω| allein war in Lauf v1 instabil (Explosion nach 3–6 Schritten). Fit-Fenster: max|ω| ≥ max|ω|_end/20.
  T aus 1/max|ω| linear in t; c aus log(Kernbreite) vs. log(T−t), Kernbreite =
  Abstand des Maximums von |ω| zu x0. Kontrolle c_alt aus ω_x(x0) ~ τ^(−1−c).
- Profilvergleich: τ·ω(x0 + λξ) mit λ = −1/(τ·ω_x(x0)) (dieselbe Normierung
  Ω'(0) = −1), relL2 auf |ξ| ≤ 10, Schwelle 0.05 (eigene Wahl).
- Hardware: Kaggle 2×T4 (Nutzerwunsch). Es wird eine GPU genutzt; zu Beginn
  misst das Skript Adam-Schritte auf CPU und GPU in FP64 und nimmt das
  schnellere Gerät (T4 hat schwache FP64-Leistung).

## Automatische Prüfausgaben (Schritt 0)

- sympy profile_eq_residual: 0
- sympy closure_residual: 0
- sympy even_part_solution: Eq(Omega(xi), C1/xi**(1/c))
- sympy clm_residual: 0
- sympy clm_U_prime_minus_H: 0
- sympy clm_Omega_prime_0: -1
- sympy ok: True
- pipeline reg_path_max_R: 1.9984014443252818e-15
- pipeline reg_path_U_err_core: 1.6714277881746398e-06
- pipeline sing_path_max_R: 4.440892098500626e-16
- pipeline sing_path_U_err_core: 1.1102230246251565e-16
- pipeline ok: True
- Hilbert H[1/(1+x^2)] = x/(1+x^2), L=1 (Gate): 4.718e-16
- Hilbert H[x/(1+x^2)] = -1/(1+x^2), L=1: 6.384e-16
- Hilbert H[1/(1+x^2)^2] = x(x^2+3)/(2(1+x^2)^2), L=1: 6.384e-16
- Hilbert H[1/(1+x^2)] = x/(1+x^2), L=1.7: 5.551e-16
- H[Im E_s] Quadraturtest s=0.5, x=0.3: 1.78e-15
- H[Im E_s] Quadraturtest s=0.5, x=1.0: 1.22e-15
- H[Im E_s] Quadraturtest s=0.5, x=4.0: 3.49e-14
- H[Im E_s] Quadraturtest s=1.5, x=0.3: 2.04e-12
- H[Im E_s] Quadraturtest s=1.5, x=1.0: 4.20e-12
- H[Im E_s] Quadraturtest s=1.5, x=4.0: 2.12e-10
