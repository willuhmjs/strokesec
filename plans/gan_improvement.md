# GAN Realism Analysis & Improvement Plan

## 1. Analysis of Current State

### 1.1 Real Data Characteristics (from `data/*.csv`)
Looking at `will_data.csv`, `anne_data.csv`, and `keystroke_data.csv`, we observe several key properties of human typing:
*   **Correlation:** Key hold times and flight times are often correlated. Fast typists have low hold and flight times.
*   **Rhythm/Structure:** There are distinct patterns. For example, `dd` (Down-Down) flight time is often `hold` + `ud` (Up-Down).
*   **Physical Constraints:**
    *   Hold times are always positive (typically 50ms - 150ms).
    *   UD Flight times can be negative (rollover typing) or positive.
    *   DD Flight times must be positive (you can't press the next key before the previous one... usually).
*   **Distribution:** Data is not perfectly Gaussian; it's often skewed (e.g., you can't type infinitely fast, but you can pause for a long time).

### 1.2 Current GAN Generator (`src/crack_model.py`)
The current generator is a simple Multi-Layer Perceptron (MLP):
*   **Architecture:** `Linear -> LeakyReLU -> BN -> Linear -> LeakyReLU -> BN -> Linear`.
*   **Input:** Random Gaussian Noise (`z`).
*   **Output:** Scaled feature vector (via `StandardScaler` inversion later).

### 1.3 Why the Current Generator might be "Unrealistic"
1.  **Independent Output Neurons:** The final layer is a simple Linear layer. It outputs all 60 features (3 features * 20 keys) simultaneously. It doesn't inherently "know" that `k0_hold`, `k0_ud`, and `k0_dd` are related, or that `k1` follows `k0` in time.
2.  **No Temporal Awareness:** It treats the keystroke sequence as a flat vector, ignoring the sequential nature of typing.
3.  **Physical Violations:** It might generate statistically plausible values that are physically impossible (e.g., `Hold` + `UD` != `DD`, or extreme outliers that `StandardScaler` allows but human physiology doesn't).
4.  **Mode Collapse Risk:** Simple GANs often output the "average" safe pattern rather than the diverse, slightly jittery variations of a real human.

## 2. Improvement Plan

To make the generator more realistic, we need to inject **domain knowledge** and **temporal structure** into it.

### 2.1 Strategy: Use a Sequential Model (RNN/LSTM/GRU)
Instead of generating the whole sequence at once, generate it key-by-key or beat-by-beat. This allows the model to learn the "rhythm".

*   **Proposed Architecture:** `LSTM` or `GRU` Generator.
    *   **Input:** Random Noise + Previous Key's Timing.
    *   **Hidden State:** Keeps track of the "tempo" or "style".
    *   **Output:** Delta timing for the current key.

### 2.2 Strategy: Enforce Physical Constraints (Physics-Informed Loss)
We can add a custom loss term to the GAN training loop that penalizes physically impossible combinations.
*   **Constraint 1:** `Hold > 0`.
*   **Constraint 2:** `DD ~= Hold + UD` (approximate, depending on how `UD` is measured).

### 2.3 Strategy: Improved "Fuzzy" Features
Instead of just random noise, we can seed the generator with "Average User" stats and ask it to generate *deviations* from that average.

## 3. Actionable Todo List for Code Mode

We will focus on a **Sequential Generator** upgrade, as it's the most impactful change for time-series data like keystrokes.

1.  **Modify `Generator` in `src/crack_model.py`:**
    *   Switch from `nn.Linear` stack to `nn.LSTM` or `nn.GRU`.
    *   Reshape input/output to handle `(Batch, Sequence_Length, Features_Per_Key)`.
2.  **Modify `Discriminator` in `src/crack_model.py`:**
    *   Update to handle sequential input (or flatten it intelligently).
3.  **Update Training Loop:**
    *   Ensure data reshaping: `(N, 60)` -> `(N, 20, 3)`.
4.  **Add Constraint Enforcement (Optional but recommended):**
    *   Post-process generated samples to clamp negative hold times if necessary.

## 4. Revised Plan

I will proceed by updating `src/crack_model.py` to use an LSTM-based GAN architecture. This captures the sequential dependency of typing (e.g., if I delay on 'T', I might rush 'H' in "THE").

### Updated `src/crack_model.py` Plan:
*   **Data Prep:** Reshape `X_train` from `[N, 60]` to `[N, 20, 3]`.
*   **Generator:** `LSTM(input_size=noise_dim, hidden_size=64, num_layers=2)`. Output layer maps hidden state to 3 features.
*   **Discriminator:** `LSTM` that outputs a validity score for the sequence.

This is a significant architectural shift but provides the realism requested.
