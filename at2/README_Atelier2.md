# Atelier 2 — Analyse de données météorologiques (NumPy)

Module: IA et Machine Learning 1 · IAIL · S5 · Study guide + cheat sheet + self-test

This README does not repeat the answers (they are in `Atelier2_Corrige.pdf`). It explains **what you need to understand to solve the TP yourself**, shows how to use the video, lists the traps, and ends with a self-test with answers.

---

## 0. Setup and workflow

```bash
pip install numpy notebook
jupyter notebook
```

- First cell of the notebook: `import numpy as np`. Without it, every `np.` fails with `NameError: name 'np' is not defined`.
- `Shift + Enter` runs a cell. The notebook remembers variables between cells: if something looks wrong, `Kernel → Restart & Run All`.
- Use `print(...)` for each answer so the output is clear.
- Work on **one variable per idea** and do not overwrite your base arrays (`temperature`, `Temperatures`, `Precipitations`): Part 4 asks for conversions and modifications, so store those results in **new variables** (`Temperatures_F`, `Temperatures_rechauffe`...).

---

## 1. How to use the video with this TP

I could not watch the video; this mapping is based on its chapter titles, so check each chapter and fill the gaps with the NumPy docs.

| TP part | Watch | Then look up yourself |
|---|---|---|
| 1 — 1D array, indexing, sort | Introduction (0:00), Slicing (12:54) | `np.sort` |
| 2 — 2D arrays, random, `.T`, `np.dot` | Multidimensional arrays (5:07), Random numbers (52:03) | `np.random.uniform` (the video may show other random functions), `.T`, `np.dot` |
| 3 — statistics | Aggregate functions (39:42), Filtering (43:48) | `median`, `std`, `argmax`, `np.sort(axis=1)` and above all the `axis` argument (section 3.6 below) |
| 4 — indexing, filtering, arithmetic | Slicing (12:54), Arithmetic (24:04), Broadcasting (33:21), Filtering (43:48) | `np.clip` / `np.where` |

Suggested order: watch 0:00 → 52:03 with a notebook open, **type the examples yourself**, then do the TP part by part.

---

## 2. Map of the TP: exercise → tool

| Exercise | Tool |
|---|---|
| type, dimension, shape | `type()`, `.ndim`, `.shape` |
| first / last / three first | `a[0]`, `a[-1]`, `a[:3]` |
| +1 on first and last | `a[0] += 1`, `a[-1] += 1` |
| sorted copy | `np.sort(a)` |
| create random 5×7 | `np.random.uniform(0, 20, (5, 7))` |
| days 1 to 7 | `np.arange(1, 8)` |
| transpose | `.T` |
| matrix product | `np.dot(A, B)` or `A @ B` |
| mean / max / min / median / std / sum | `.mean`, `.max`, `.min`, `np.median`, `.std`, `.sum` with `axis` |
| days above 25 °C | boolean mask `T[1, :] > 25`, then `jours[mask]` |
| sort per city | `np.sort(T, axis=1)` |
| index of hottest day | `T.argmax(axis=1)` |
| 2D slicing | `T[1, :3]`, `T[-1, -4:]` |
| cities above 26 °C on average | `T.mean(axis=1) > 26` |
| Fahrenheit, +2 °C | vectorized arithmetic |
| cap at 35 | `np.clip(T, None, 35)` / mask assignment / `np.where` |

---

## 3. Concepts in detail

### 3.1 Why NumPy and not lists?

An `ndarray` is a table of numbers **all of the same type**, stored compactly, on which operations apply to **all elements at once** (*vectorization*), without writing `for` loops. The TP asks explicitly: "sans utiliser de boucles".

```python
[1, 2, 3] * 2          # list  → [1, 2, 3, 1, 2, 3]  (repeats the list)
np.array([1, 2, 3]) * 2  # array → [2 4 6]            (multiplies each element)
```

### 3.2 Creating arrays

| Code | Result |
|---|---|
| `np.array([22.5, 24.0, 19.8])` | from a list |
| `np.array([[1, 2], [3, 4]])` | 2D from a list of lists |
| `np.arange(1, 8)` | `[1 2 3 4 5 6 7]` (stop excluded) |
| `np.arange(0, 10, 3)` | `[0 3 6 9]` (step 3) |
| `np.linspace(0, 1, 5)` | `[0. 0.25 0.5 0.75 1.]` (5 points, both ends included) |
| `np.zeros((2, 3))`, `np.ones(3)` | arrays of 0 / 1 |
| `np.random.uniform(0, 20, (5, 7))` | random floats in [0, 20), shape 5×7 |

**`np.array` vs `np.random.uniform`** (question of the TP): `np.array(data)` *converts* data that you already have; `np.random.uniform(low, high, size)` *generates* random numbers with equal probability between `low` and `high`, different at each run. `np.random.seed(42)` before it makes the numbers reproducible (same sequence every run). A more modern equivalent is `rng = np.random.default_rng(42)` then `rng.uniform(0, 20, (5, 7))`.

### 3.3 Attributes: `type`, `ndim`, `shape`, `size`, `dtype`

```python
temperature = np.array([22.5, 24.0, 19.8, 21.2, 23.5, 25.1, 20.7])
type(temperature)   # <class 'numpy.ndarray'>
temperature.ndim    # 1  (number of dimensions)
temperature.shape   # (7,)  (tuple: elements per dimension)
temperature.size    # 7  (total number of elements)
temperature.dtype   # float64
```

For the 5×7 array: `.ndim` = 2, `.shape` = `(5, 7)`, `.size` = 35.

**dtype matters.** `Temperatures` is built from integers, so its dtype is an integer type (usually `int64`):
- `Temperatures[0, 0] = 22.7` silently stores `22` (the decimal part is cut).
- `Temperatures += 0.5` fails: `UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'`. Use `Temperatures = Temperatures + 0.5` (creates a new float array) or `Temperatures.astype(float)`.
- `Temperatures * 9 / 5 + 32` works because `/` produces floats and creates a new array.

### 3.4 Indexing and slicing

1D: same as Python lists.

```python
a[0]      # first
a[-1]     # last
a[:3]     # three first (stop excluded)
a[-4:]    # four last
a[::2]    # every second element
```

2D: `a[row, column]`, and each of the two parts can be an index or a slice.

```python
T[1, 2]      # one value: 2nd city, 3rd day
T[1]         # whole 2nd row (shape (7,))     = T[1, :]
T[1, :3]     # 2nd city, 3 first days
T[-1, -4:]   # last city, 4 last days
T[:, 0]      # first day of all cities (shape (5,))
T[:2, 1:4]   # rows 0-1, columns 1-3
T[[0, 2]]    # rows 0 and 2 (list of indexes → shape (2, 7))
```

Remember: **`:` means "all"**, the first position is the row (city), the second is the column (day).

**Slices are views, not copies.** `b = a[:2]; b[0] = 0` changes `a` too. Use `a[:2].copy()` if you want an independent array. (Boolean masks and index lists, in contrast, return copies.)

### 3.5 Modifying values

```python
temperature[0] += 1                 # first
temperature[-1] += 1                # last
temperature[[0, -1]] += 1           # both at once
Temperatures[Temperatures > 35] = 35  # mask assignment
```

### 3.6 The `axis` argument — the most important idea of Part 3

For a 2D array `T` of shape `(5, 7)` = 5 cities (rows) × 7 days (columns):

```
              day1 day2 ... day7
  city 1   [  22   25  ...  30 ]  → axis=1 gives ONE result per ROW    (5 values)
  city 2   [  18   21  ...  26 ]
  ...
  city 5   [  25   26  ...  30 ]
               ↓    ↓          ↓
           axis=0 gives ONE result per COLUMN (7 values)
```

- **`axis=1`**: the operation runs *along the columns* and you get **one value per city**: `T.mean(axis=1)` → 5 values.
- **`axis=0`**: the operation runs *down the rows* and you get **one value per day**: `T.mean(axis=0)` → 7 values.
- **no `axis`**: everything is flattened → **one single number**.
- Mnemonic: *the axis you name is the one that disappears.* `shape (5, 7)` with `axis=1` → shape `(5,)`; with `axis=0` → shape `(7,)`.

Examples with the TP data:

| Code | Result |
|---|---|
| `T.mean(axis=1)` | `[24. 22.57 30.57 18. 27.43]` (per city) |
| `T.mean(axis=0)` | `[22. 24.2 21.6 22.8 24.8 27.8 28.4]` (per day) |
| `T.sum(axis=0)` | `[110 121 108 114 124 139 142]` |
| `T.sum(axis=1)` | `[168 158 214 126 192]` |
| `T.sum()` | `858` |
| `T.argmax(axis=1)` | `[6 5 6 6 5]` (0-based column of the max per city) |
| `T.argmax(axis=0)` | `[2 2 2 2 4 2 2]` (0-based row of the max per day) |
| `T.argmax()` | `20` (position in the flattened array) |

### 3.7 Aggregate functions

| Function | Meaning | Method form? |
|---|---|---|
| `mean` | average | `T.mean()` / `np.mean(T)` |
| `median` | middle value of the sorted values | function only: `np.median(T, axis=1)` |
| `std` | standard deviation (spread around the mean) | `T.std()` / `np.std(T)` |
| `sum` | total | `T.sum()` |
| `max`, `min` | largest / smallest | `T.max()` |
| `argmax`, `argmin` | **position** of largest / smallest | `T.argmax()` |

All accept `axis`. About `std`: NumPy computes by default the *population* standard deviation (`ddof=0`, dividing by n). For the first city that is `3.742`; with `ddof=1` (sample standard deviation) it would be `4.041`. The TP only says "std", so the default is what is expected.

`argmax` returns an **index** (0 to 6 here). To get the day number, add 1 or index the `jours` array: `jours[T.argmax(axis=1)]`.

### 3.8 Sorting

```python
np.sort(a)                # sorted COPY, a unchanged
a.sort()                  # sorts IN PLACE and returns None
np.sort(T, axis=1)        # each row sorted separately (each city)
np.sort(T, axis=0)        # each column sorted separately
np.sort(T, axis=None)     # flattened, then sorted
np.argsort(a)             # the indexes that would sort a
```

(Careful: `r = a.sort()` makes `r` equal to `None`.)

### 3.9 Boolean masks (filtering)

A comparison on an array returns a **boolean array** of the same shape. Using it as an index keeps only the `True` positions.

```python
jours = np.arange(1, 8)
masque = T[1, :] > 25      # [False False False False False True True]
jours[masque]              # [6 7]
T[T > 25]                  # all values above 25 (1D result)
T[(T > 25) & (T < 30)]     # combine with & (and), | (or), ~ (not)
```

- Use `&`, `|`, `~` with **parentheses around each condition**. Python's `and`, `or` do not work on arrays (`ValueError: The truth value of an array ... is ambiguous`).
- Counting: `(T > 25).sum()` counts the `True` (16 here); `(T > 25).sum(axis=1)` per city `[2 2 7 0 5]`; `(T > 25).mean()` gives the proportion.
- `.any(axis=1)` / `.all(axis=1)` test "at least one" / "all". `(T > 25).all(axis=1)` is `True` only for city 3.
- A mask on **means**: `T.mean(axis=1) > 26` gives one boolean per city → `villes[mask]` are the cities above 26 °C.
- Empty result `[]` is a valid answer: `Precipitations[1, :] == 0` is almost never true for random floats.

### 3.10 Vectorized arithmetic and broadcasting

```python
T * 9 / 5 + 32     # Celsius → Fahrenheit, on all 35 values
T + 2              # +2 °C everywhere
T - T.mean()       # difference to the global mean
```

**Broadcasting** = NumPy stretches the smaller array so shapes match. Rule: compare the shapes from the **right**; two dimensions are compatible if they are **equal** or one of them is **1** (a missing dimension counts as 1).

| Shapes | Result |
|---|---|
| `(5, 7)` and scalar | OK |
| `(5, 7)` and `(7,)` | OK (the vector is applied to every row, e.g. subtract the daily mean) |
| `(5, 7)` and `(5,)` | **Error**: `operands could not be broadcast together with shapes (5,7) (5,)` |
| `(5, 7)` and `(5, 1)` | OK |

To subtract each city's own mean (shape `(5,)`) you need `(5, 1)`:

```python
T - T.mean(axis=1, keepdims=True)   # keepdims keeps the axis with size 1
T - T.mean(axis=1)[:, None]         # or add the axis by hand
```

Both give, for the first city, `[-2. 1. -4. -5. 0. 4. 6.]`.

Element-wise (`*`) vs matrix (`@`): `T * P` multiplies matching cells and requires the same shape (or broadcastable). `T @ P.T` is the matrix product.

### 3.11 Transpose and matrix product

- `T.T` swaps axes: `(5, 7)` → `(7, 5)`.
- `np.dot(A, B)` (same as `A @ B` for 2D arrays) needs **the inner dimensions to match**: `(5, 7) · (7, 5)` → `(5, 5)`. Cell `[i, j]` = sum over the 7 days of `A[i, k] * B[k, j]`.
- `T * P.T` fails: `operands could not be broadcast together with shapes (5,7) (7,5)`; `np.dot(T, P)` fails too since `(5,7)·(5,7)` has inner dimensions 7 and 5.

### 3.12 Capping values: three equivalent ways

```python
np.clip(T, None, 35)           # keep everything ≤ 35 (None = no lower bound)
np.minimum(T, 35)              # element-wise minimum with 35
np.where(T > 35, 35, T)        # if condition: 35, else: keep T
T2 = T.copy(); T2[T2 > 35] = 35  # mask assignment on a copy
```

All give the same result. `np.clip(T, 20, 30)` forces values into `[20, 30]` (both sides). Note that in the TP data nothing exceeds 35 until you add the +2 °C warming (city 3, day 7: 35 + 2 = 37 → capped to 35), so apply the cap to the warmed array to see an effect.

### 3.13 Reshaping (useful to know)

`T.reshape(7, 5)` keeps the same 35 values in another shape (the sizes must multiply to 35); `T.ravel()` flattens to shape `(35,)`. Do not confuse with `.T`, which really swaps rows and columns.

---

## 4. Cheat sheet (one page)

```python
import numpy as np

# create
a = np.array([1, 2, 3]);  T = np.array([[1, 2, 3], [4, 5, 6]])
np.arange(1, 8); np.linspace(0, 1, 5); np.zeros((2, 3)); np.ones(3)
np.random.seed(0); np.random.uniform(0, 20, (5, 7)).round(1)   # reassign to keep the rounding

# inspect
type(T); T.ndim; T.shape; T.size; T.dtype

# index / slice  (row, column)
T[0, 0]; T[1]; T[1, :3]; T[-1, -4:]; T[:, 0]

# stats  (axis=0 → per column, axis=1 → per row, none → global)
T.mean(axis=1); T.max(axis=1); T.min(axis=1); T.sum(axis=0)
np.median(T, axis=1); T.std(axis=1); T.argmax(axis=1)

# sort
np.sort(a); np.sort(T, axis=1); np.argsort(a)

# mask
mask = T[1, :] > 25;  jours[mask];  T[T > 25];  T[(T > 25) & (T < 30)]

# arithmetic
T * 9 / 5 + 32;  T + 2;  T.T;  np.dot(A, B);  A @ B

# cap
np.clip(T, None, 35);  np.minimum(T, 35);  np.where(T > 35, 35, T)
```

| I want... | Write |
|---|---|
| one value per city | `axis=1` |
| one value per day | `axis=0` |
| a single global value | no `axis` |
| the position of the max | `argmax` (0-based) |
| the days that satisfy a condition | `jours[mask]` |
| the cities that satisfy a condition | `villes[mask]` |
| apply a formula to all values | write it directly on the array (no loop) |
| a sorted array, keeping the original | `np.sort(a)` |
| keep the original while modifying | `b = a.copy()` or use a new variable |

---

## 5. Common mistakes and the errors they produce

| Problem | Cause | Fix |
|---|---|---|
| `NameError: name 'np' is not defined` | forgot `import numpy as np` | first cell |
| `Precipitations.round(1)` seems to do nothing | it returns a new array and the variable is unchanged | `Precipitations = Precipitations.round(1)` |
| `(7,)` shown for `.shape` | 1D tuple with one element | normal |
| `a[:3]` gives 3 elements, not 4 | stop is excluded | normal |
| `T[:][0]` does not give the first column | `T[:]` is the whole array, then `[0]` takes its first **row** | `T[:, 0]` for a column |
| wrong `axis` | mixing rows/columns | rows = cities, columns = days, name the axis that disappears |
| `argmax` gives 6 and you expected day 7 | 0-based position | `jours[idx]` or `idx + 1` |
| `ValueError: ...shapes (5,7) (5,)` | broadcasting rule not satisfied | `keepdims=True` or `[:, None]` |
| `ValueError: ...shapes (5,7) (7,5)` with `*` | element-wise product needs same shape | `np.dot` / `@` for matrix product |
| `UFuncTypeError: Cannot cast ufunc 'add' output...` | `+=` with floats on an integer array | write `T = T + 0.5` or `T.astype(float)` |
| `T[T > 25 and T < 30]` fails | `and` is not for arrays | `(T > 25) & (T < 30)` |
| original array unexpectedly changed | slices are views / in-place assignment | `.copy()` or a new variable |
| random values differ from a classmate | no seed | `np.random.seed(42)` |
| empty `[]` for dry days | exact `0` is almost impossible with random floats | not an error; test with a threshold, e.g. `< 5` |
| `Temperatures` already modified in a later cell | cells run several times / in a different order | Restart & Run All |

---

## 6. Self-test (QA) — answers included

Try to answer before reading. Numbers below come from running the code on the TP arrays (`T` = `Temperatures`).

**Q1. What is the type of `temperature`, and how does it differ from a Python list?**
`numpy.ndarray`. All elements share a dtype, operations are vectorized (`a * 2` doubles every element, whereas `list * 2` repeats the list).

**Q2. For `temperature` give `.ndim`, `.shape`, `.size`, `.dtype`.**
`1`, `(7,)`, `7`, `float64`.

**Q3. For `Temperatures` give `.ndim`, `.shape`, `.size`.**
`2`, `(5, 7)`, `35`. Rows are cities, columns are days.

**Q4. What do `temperature[0]`, `temperature[-1]`, `temperature[:3]` return on the original data?**
`22.5`, `20.7`, `[22.5 24. 19.8]`.

**Q5. Why does `temperature[:3]` give 3 elements and not 4?**
Slices include the start index and exclude the stop index: indexes 0, 1, 2.

**Q6. How do you add 1 °C to the first and the last temperature? What does the array become?**
`temperature[0] += 1` and `temperature[-1] += 1` (or `temperature[[0, -1]] += 1`). Result: `[23.5 24. 19.8 21.2 23.5 25.1 21.7]`.

**Q7. `np.sort(temperature)` vs `temperature.sort()`: what is the difference? What does the second one return?**
`np.sort` returns a new sorted array and leaves the original untouched. `.sort()` sorts the original in place and returns `None` (so `r = a.sort()` gives `r is None`).

**Q8. What is the sorted version of the modified `temperature`?**
`[19.8 21.2 21.7 23.5 23.5 24. 25.1]`.

**Q9. Why does `Precipitations.round(1)` (alone, as written in the TP) not modify `Precipitations`? How do you keep the rounding?**
`.round(1)` returns a new array; nothing is stored. Use `Precipitations = Precipitations.round(1)`, or chain it: `np.random.uniform(0, 20, (5, 7)).round(1)`.

**Q10. What is the difference between `np.array()` and `np.random.uniform()`?**
`np.array` converts existing data you provide into an array. `np.random.uniform(low, high, size)` creates new random data between `low` and `high` (uniform distribution), different at each run unless a seed is set.

**Q11. What are the shapes of `Temperatures`, `Precipitations`, `T.T` and `np.dot(T, P.T)`?**
`(5, 7)`, `(5, 7)`, `(7, 5)`, `(5, 5)`.

**Q12. Why is `np.dot(Temperatures, Precipitations)` an error but `np.dot(Temperatures, Precipitations.T)` works?**
Matrix product needs the inner dimensions to be equal. `(5,7)·(5,7)` has inner dimensions 7 and 5 → error. `(5,7)·(7,5)` has 7 and 7 → result `(5,5)`.

**Q13. What is the difference between `T * P`, `T @ P.T` and `np.dot(T, P.T)`?**
`T * P` multiplies cell by cell (same shape required, result `(5, 7)`). `T @ P.T` and `np.dot(T, P.T)` are the same matrix product for 2D arrays (checked: identical result), shape `(5, 5)`.

**Q14. What does `np.arange(1, 8)` return and why not 1 to 8?**
`[1 2 3 4 5 6 7]`. The stop value is excluded, like in `range`.

**Q15. What does `axis=1` mean? Give the shape of `T.mean(axis=1)` and of `T.mean(axis=0)`.**
`axis=1` runs along the columns and returns one value per row (per city): shape `(5,)`. `axis=0` returns one value per column (per day): shape `(7,)`. No `axis` → one number. Rule: the axis you name disappears from the shape.

**Q16. Mean per city?**
`[24. 22.57 30.57 18. 27.43]` (rounded to 2 decimals).

**Q17. Max and min per city?**
max `[30 27 35 21 31]`, min `[19 18 27 15 24]`.

**Q18. Global mean of all values?**
`24.514285714285716` (≈ `24.51`), i.e. 858 / 35.

**Q19. Median per city?**
`[24. 22. 30. 18. 27.]`. There is no `T.median()` method: use `np.median(T, axis=1)`.

**Q20. Standard deviation per city, and what does it mean?**
`[3.74 3.25 2.66 2. 2.44]`. It measures how far the daily temperatures are from the city's mean: city 1 varies the most, city 4 the least. With `ddof=1` the values would be `[4.04 3.51 2.88 2.16 2.64]`, but the default (`ddof=0`) is what the TP expects.

**Q21. Which city has the highest mean? the lowest? Which index does `argmax`/`argmin` give?**
City 3 (30.57) and city 4 (18.0). `T.mean(axis=1).argmax()` is `2`, `.argmin()` is `3` (0-based, so add 1 for the city number).

**Q22. Precipitation totals per day and per city (axis choice)?**
Per day → `axis=0` (7 values), per city → `axis=1` (5 values). For `T` (temperatures) the same calls give `[110 121 108 114 124 139 142]` and `[168 158 214 126 192]`; the total is `858`. Precipitation values are random: with `np.random.seed(42)` and rounding to 1 decimal, the per-day totals are `[43. 41.4 54.4 35.4 48.1 48.5 47.2]` and the per-city totals `[60.5 84.1 50.5 55. 67.9]`.

**Q23. Explain `masque = Temperatures[1, :] > 25` and `jours_chauds = jours[masque]`.**
`Temperatures[1, :]` is the 2nd city's 7 temperatures. `> 25` gives 7 booleans `[F F F F F T T]`. Indexing `jours` with it keeps the days where it is `True`: `[6 7]`.

**Q24. Why can `jours[Precipitations[1, :] == 0]` be empty? Is the code wrong?**
No. Random floats between 0 and 20 are almost never exactly 0 (and after rounding to 1 decimal, only values below 0.05 become 0.0). An empty array `[]` is a correct result. Use a threshold (e.g. `< 5`) to find "nearly dry" days.

**Q25. `np.sort(T, axis=1)`: what exactly is sorted? What about `axis=0`, and `axis=None`?**
`axis=1`: each row (city) is sorted independently, the first row becomes `[19 20 22 24 25 28 30]`. `axis=0`: each column (day) sorted separately. `axis=None`: everything flattened, then sorted into a 1D array of 35 values. Without `axis`, the default is the last axis (same as `axis=1` here).

**Q26. `T.argmax(axis=1)` returns `[6 5 6 6 5]`. What does it mean, and what is `T.argmax()` (no axis)?**
For each city, the 0-based column of its maximum: city 1 peaks at index 6 (day 7), city 2 at index 5 (day 6), etc. Without `axis`, the position in the flattened array: `20` (row 2, column 6: `np.unravel_index(20, T.shape)` is `(2, 6)`).

**Q27. Write the code for "temperatures of the first 3 days for the 2nd city" and "last city, 4 last days". Results?**
`T[1, :3]` → `[18 21 19]`; `T[-1, -4:]` → `[27 29 31 30]`.

**Q28. Which cities have a mean above 26 °C?**
`villes = np.arange(1, 6)` then `villes[T.mean(axis=1) > 26]` → `[3 5]` (30.57 and 27.43).

**Q29. How do you convert to Fahrenheit without a loop? Why store it in a new variable?**
`T * 9 / 5 + 32`. The formula is applied to every element at once. A new variable (`Temperatures_F`) keeps the Celsius data for the following exercises.

**Q30. What is broadcasting in `T + 2`?**
The scalar `2` is virtually stretched to shape `(5, 7)` and added to each cell. No loop and no copy of the scalar in memory.

**Q31. Why does `T - T.mean(axis=1)` fail, and how do you fix it?**
Shapes `(5, 7)` and `(5,)` are aligned from the right (7 vs 5): incompatible → `ValueError: operands could not be broadcast together with shapes (5,7) (5,)`. Fix: `T - T.mean(axis=1, keepdims=True)` or `T - T.mean(axis=1)[:, None]` (shape `(5, 1)`).

**Q32. Three ways to replace values above 35 by 35? Which one fits "limiter les valeurs extrêmes"?**
`np.clip(T, None, 35)`, `np.minimum(T, 35)`, `np.where(T > 35, 35, T)` (also `T[T > 35] = 35` on a copy). All are equivalent; `np.clip` expresses the idea most directly.

**Q33. After `Temperatures + 2`, how many values exceed 35, and which one?**
Exactly one: city 3, day 7 (35 + 2 = 37), capped to 35. Check: `(T2 > 35).sum()` is `1`. On the original array nothing exceeds 35, so capping it changes nothing.

**Q34. What happens with `Temperatures += 0.5`, and with `Temperatures[0, 0] = 22.7`?**
`+= 0.5` raises `UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64')...` because the array holds integers. `Temperatures[0, 0] = 22.7` silently stores `22`. Create a float array first (`T.astype(float)`) if you need decimals.

**Q35. Does `b = a[:3]; b[0] = 0` change `a`? And `c = a[a > 1]; c[0] = 99`?**
Yes for the slice: it is a **view** on the same data (checked: `a[0]` became `0.0`). No for the boolean mask: it returns a **copy** (checked: `a` unchanged). Use `.copy()` when you need independence.

**Q36. `T[(T > 25) & (T < 30)]` vs `T[T > 25 and T < 30]`?**
The first works (parentheses are mandatory because `&` has higher priority than `>`). The second raises `ValueError: The truth value of an array with more than one element is ambiguous`. Use `&`, `|`, `~` on arrays.

**Q37. How many temperatures exceed 25 °C in total, and per city?**
`(T > 25).sum()` → `16`; `(T > 25).sum(axis=1)` → `[2 2 7 0 5]` (True counts as 1).

**Q38. What is the difference between `.T`, `.reshape(7, 5)` and `.ravel()`?**
`.T` transposes (rows ↔ columns, `(5, 7)` → `(7, 5)`, values re-arranged accordingly). `.reshape(7, 5)` keeps the reading order and only changes the shape. `.ravel()` flattens to `(35,)`.

**Q39. Why does `np.random.seed(42)` matter in a TP?**
It fixes the random generator, so you and your classmates (or you tomorrow) get the same `Precipitations`. Without it, all the random-based results (totals, dry days) differ at each run, which is normal but makes comparing answers impossible.

**Q40. Bonus: how would you find, for each city, the number of days above its own mean, without a loop?**
`(T > T.mean(axis=1, keepdims=True)).sum(axis=1)`. The `keepdims=True` gives shape `(5, 1)` so broadcasting compares each row with its own mean, then `sum(axis=1)` counts the `True` per city. On the TP data the result is `[3 3 3 3 3]`.

---

## 7. Where to learn more

- NumPy quickstart: https://numpy.org/doc/stable/user/quickstart.html
- NumPy for absolute beginners: https://numpy.org/doc/stable/user/absolute_beginners.html
- Function reference (search `np.clip`, `np.where`, `np.sort`...): https://numpy.org/doc/stable/reference/
- In a notebook, `np.clip?` shows the documentation of a function.
- Next steps after the TP (as the video suggests): Pandas for tables with labels (cities/days as names instead of positions), then linear algebra for machine learning.
