# Atelier 1 — Gestion d'une petite bibliothèque (Python basics)

Module: IA et Machine Learning 1 · IAIL · S5 · Study guide + cheat sheet + self-test

This README does not repeat the answers (they are in `Atelier1_Corrige.pdf`). It explains **what you need to know to solve the TP yourself**, what the usual traps are, and gives you a self-test with answers at the end.

---

## 0. Setup and workflow

The TP must be done in a **Jupyter notebook**.

```bash
pip install notebook          # once
jupyter notebook              # opens the browser
```

- New notebook → **Python 3**. Each box is a *cell*.
- `Shift + Enter` runs the cell and moves to the next one.
- Use **Markdown cells** (`Code` → `Markdown` in the toolbar) for titles like `## Partie 1` and `### Exercice 1`; use **code cells** for the answers.
- The notebook has a **memory** (the kernel): a variable created in a cell exists in all later cells, even if you delete the cell. If results look strange: `Kernel → Restart & Run All`. Do this before handing in, to be sure the notebook works from top to bottom.
- Use `print(...)` for anything the exercise says "affiche". The last expression of a cell is displayed automatically, but `print` is clearer and works the same in a script.

Recommended method for each exercise: (1) read the sentence, (2) write what you need in words ("get the 3rd element of the tuple stored under this key"), (3) write the code, (4) run it, (5) compare with the PDF.

---

## 1. Map of the TP: exercise → concept

| Part | Exercises | What you must know |
|---|---|---|
| 1 Variables | f-string sentence, total cost, +10 % | types, f-strings and formatting, arithmetic |
| 2 Lists and tuples | 2nd element, add, remove, tuple loop, list of tuples | indexing, `append`, `remove`/`pop`/`del`, tuples, `for` loop |
| 3 Dictionaries | keys, values, price lookup, add, modify, delete, print, filter | `keys/values/items`, key access, assignment, `del`, tuple immutability, `if` in a loop |
| 4 Functions | display, add, search, count | `def`, parameters, `return` vs `print`, `in`, `len` |

---

## 2. Concepts in detail

### 2.1 Variables and types

A variable is a **name** attached to a value. Python decides the type from the value.

```python
titre = "L'Étranger"   # str   (text)
prix  = 120.0          # float (decimal number)
Q     = 5              # int   (integer)
type(prix)             # <class 'float'>
```

Points to know:
- Strings can use `"..."` or `'...'`. If the text contains `'` (like *L'Étranger*), wrap it in `"..."`.
- Use **straight quotes** `"` and `'`. The curly quotes `“ ”` you may copy from a PDF or Word cause a `SyntaxError` (the TP text contains some around Victor Hugo).
- Names are case-sensitive: `Prix` ≠ `prix`. In Part 2 you create a **list** called `prix`: it replaces the float `prix` from Part 1, because **reassigning a name replaces its old value**.

### 2.2 f-strings and number formatting

```python
print(f"Le livre '{titre}' de {auteur} coûte {prix:.0f} DH.")
```

- `f"..."` lets you insert any expression between `{}`.
- After a colon you give a **format**: `:.0f` = float with 0 decimals, `:.2f` = 2 decimals, `:>10` = right-aligned in 10 characters.
- Without a format, a float prints as it is: `120.0`. The TP sentence shows `120 DH`, so use `:.0f` (or `int(prix)`; `int` truncates instead of rounding).
- Floats are stored in binary, so some decimals are not exact (`0.1 + 0.2` gives `0.30000000000000004`). For money, display with `:.2f`.

### 2.3 Arithmetic

| Operator | Meaning | Example |
|---|---|---|
| `+ - * /` | basic (`/` always gives a float) | `7 / 2` → `3.5` |
| `//` | integer division | `7 // 2` → `3` |
| `%` | remainder | `7 % 2` → `1` |
| `**` | power | `2 ** 3` → `8` |

- Total stock cost = `prix * Q`.
- "+10 %" = `prix * 1.10` (equivalent: `prix + prix * 10 / 100`). "−10 %" would be `prix * 0.90`.

### 2.4 Lists

A list is an **ordered, mutable** sequence: `[a, b, c]`.

```python
livres = ["L'Étranger", "1984", "Le Petit Prince", "Les Misérables"]
livres[0]        # "L'Étranger"    (index starts at 0)
livres[1]        # "1984"          ← "deuxième livre"
livres[-1]       # "Les Misérables" (last)
livres[1:3]      # ['1984', 'Le Petit Prince'] (stop excluded)
len(livres)      # 4
```

| Action | Code | Notes |
|---|---|---|
| add at the end | `livres.append("X")` | one element |
| add several | `livres.extend(["X", "Y"])` | `append(["X","Y"])` would add a *list inside* the list |
| insert at a position | `livres.insert(0, "X")` | shifts the others |
| remove by value | `livres.remove("1984")` | removes the **first** match; `ValueError` if absent |
| remove by position | `livres.pop(1)` | also returns the removed element; `pop()` removes the last |
| remove by position (statement) | `del livres[1]` | returns nothing |
| position of a value | `livres.index("1984")` | `ValueError` if absent |
| test membership | `"1984" in livres` | `True` / `False` |
| copy | `livres.copy()` or `livres[:]` | `b = a` does **not** copy, both names point to the same list |

**Parallel lists.** `livres`, `prix`, `auteurs` are linked only by their index. If you remove an element from one list, remove it from the other two too, otherwise the index `i` no longer describes the same book.

### 2.5 Tuples

A tuple is an **ordered, immutable** sequence: `(a, b, c)`.

```python
livre = ("L'Étranger", "Albert Camus", 120.0)
livre[1]                      # "Albert Camus"
for element in livre:         # loop over elements
    print(element)
titre, auteur, prix = livre   # unpacking (the number of names must match)
```

- Immutable: `livre[2] = 100` → `TypeError: 'tuple' object does not support item assignment`. To "change" a tuple, build a new one.
- One-element tuple needs a comma: `("a",)` is a tuple, `("a")` is just a string.
- A **list of tuples** is a simple table: each tuple is a row.
- `zip(livres, auteurs, prix)` pairs the three lists element by element → `list(zip(...))` gives a list of tuples.

### 2.6 Loops and conditions (used everywhere in Parts 3 and 4)

```python
for element in sequence:
    ...                      # indentation (4 spaces) defines the block

if prix > 100:
    ...
elif prix == 100:
    ...
else:
    ...
```

- `==` compares, `=` assigns. Mixing them up is a classic error.
- Comparisons: `> >= < <= == !=`. `100 > 100` is `False` (strict!), `100 >= 100` is `True`.
- `range(5)` → 0,1,2,3,4. `enumerate(liste)` gives `(index, element)` pairs.

### 2.7 Dictionaries

A dictionary maps **keys** to **values**: `{key: value, ...}`. Keys are unique. Since Python 3.7 the insertion order is kept.

```python
bibliotheque = {
    "L'Étranger": ("L'Étranger", "Albert Camus", 120),
    "1984":       ("1984", "George Orwell", 150),
}
bibliotheque["1984"]          # the tuple
bibliotheque["1984"][2]       # 150  (3rd element of the tuple)
```

| Action | Code |
|---|---|
| keys / values / pairs | `d.keys()`, `d.values()`, `d.items()` (use `list(...)` to print them nicely) |
| read | `d[key]` → `KeyError` if missing; `d.get(key)` → `None` if missing; `d.get(key, default)` |
| add or overwrite | `d[key] = value` |
| delete | `del d[key]` (KeyError if missing) or `d.pop(key)` (returns the value) or `d.pop(key, None)` (safe) |
| test a key | `key in d` (tests **keys**, not values) |
| size | `len(d)` |
| loop | `for k in d:` / `for k, v in d.items():` |
| loop with unpacking | `for titre, (t, a, p) in d.items():` (value is a tuple, unpacked in the `for`) |

**Keys must be immutable** (str, int, tuple of immutables). A list cannot be a key: `TypeError: unhashable type: 'list'`.

**The trap of Part 3, exercise 5.** The value is a tuple, so you cannot do `d["Le Petit Prince"][2] = 100`. Rebuild the tuple:

```python
titre, auteur, _ = d["Le Petit Prince"]
d["Le Petit Prince"] = (titre, auteur, 100)
```

(`_` is the usual name for a value you do not need.)

### 2.8 Functions

```python
def chercher_livre(biblio, titre):      # parameters
    if titre in biblio:
        print(biblio[titre][1])
    else:
        print("Erreur : livre introuvable")
```

- `def name(parameters):` then an indented body. A function does nothing until you **call** it: `chercher_livre(bibliotheque, "1984")`.
- `print` **shows** something on screen; `return` **gives a value back** to the caller. A function without `return` returns `None`.
- If the TP says "affiche" → `print`. If it says "retourne" (like `compter_livres`) → `return`.
- A dictionary or list passed to a function is **not copied**: modifying it inside (`biblio[titre] = ...`) modifies the original. But *reassigning* the parameter (`biblio = {}`) only changes the local name and does not affect the original.
- Choose parameter names that do not hide what you need: in `ajouter_livre(biblio, titre, auteur, prix)`, `prix` is a parameter, it shadows any global `prix` inside the function (that is fine and expected).

---

## 3. Cheat sheet (one page)

```python
# --- variables / f-strings
x = 5; s = "txt"; f = 1.5
print(f"{s} vaut {f:.2f}")

# --- list
L = [1, 2, 3]
L[0]; L[-1]; L[1:3]; len(L)
L.append(4); L.insert(0, 9); L.remove(2); L.pop(); L.index(3); 3 in L

# --- tuple
t = (1, "a", 2.5)
t[0]; a, b, c = t           # immutable !
lst = [("a", 1), ("b", 2)]  # list of tuples

# --- dict
d = {"k": ("k", "auteur", 10)}
d["k"]; d["new"] = ("new", "x", 5)
del d["k"]; d.pop("new", None)
d.keys(); d.values(); d.items(); "k" in d; len(d)
for k, (t1, a1, p1) in d.items(): ...

# --- function
def f(param1, param2):
    ...
    return resultat           # or just print(...)
```

| I want to... | Use |
|---|---|
| display a number with 2 decimals | `f"{x:.2f}"` |
| second element | `L[1]` |
| last element | `L[-1]` |
| delete by value / by index | `remove` / `pop` or `del` |
| change a value inside a tuple | rebuild the tuple |
| does this title exist? | `titre in biblio` |
| number of books | `len(biblio)` |
| every pair key/value | `.items()` |
| filter on a condition | `for ...: if cond: print(...)` |

---

## 4. Common mistakes and the error messages they produce

| Error message | Cause | Fix |
|---|---|---|
| `SyntaxError: invalid character '“'` | curly quotes copied from the PDF | retype with `"` |
| `NameError: name 'livres' is not defined` | cell not executed, or kernel restarted | run the cells in order / Restart & Run All |
| `IndexError: list index out of range` / `pop index out of range` | index ≥ length | check `len(L)`; indexes go 0 … len−1 |
| `ValueError: list.remove(x): x not in list` | value absent or already removed (cell run twice!) | test with `in` first, or restart |
| `TypeError: 'tuple' object does not support item assignment` | modifying a tuple | build a new tuple |
| `KeyError: 'Dune'` | key not in the dictionary | `if key in d` or `d.get(key)` |
| `TypeError: unhashable type: 'list'` | list used as key | use a tuple or a string |
| `ValueError: too many values to unpack` | wrong number of names when unpacking | match the tuple length |
| `IndentationError` | wrong spaces after `:` | 4 spaces for each level |
| function prints `None` after your result | you printed the result of a function that only prints | decide between `print` inside or `return` + `print` outside |
| ex. 3.8 shows nothing for Le Petit Prince | its price is 100 after exercise 5, and `100 > 100` is `False` | normal, strict inequality |

**Running a cell twice** is a very common source of confusion (a second `remove("1984")` fails, a second `append` adds the book twice). When in doubt: Restart & Run All.

---

## 5. Self-test (QA) — answers included

Try to answer before reading. Every answer was checked by running the code.

**Q1. What are the types of `titre`, `prix` and `Q` in Part 1?**
`str`, `float`, `int`. Check with `type(...)`.

**Q2. What does `print(f"{prix:.0f}")` display for `prix = 120.0`, and what would `print(f"{prix}")` display?**
`120` and `120.0`. `.0f` means "float, 0 digits after the decimal point".

**Q3. Why do we write `{nouveau_prix:.2f}` for the +10 % price?**
Prices are displayed with 2 decimals, and it also hides binary float noise (for example `0.1 + 0.2` is `0.30000000000000004`). For `120.0 * 1.10` Python happens to give exactly `132.0`, but the habit is good.

**Q4. How do you increase a price by 10 %, and by 25 %? How do you decrease by 10 %?**
`prix * 1.10`, `prix * 1.25`, `prix * 0.90`.

**Q5. `livres[1]` vs `livres[2]` vs `livres[-1]` on the 4-book list?**
`"1984"`, `"Le Petit Prince"`, `"Les Misérables"`. Index 0 is the first element, −1 the last.

**Q6. What does `livres[1:3]` return? And `livres[:2]`? And `livres[::-1]`?**
`['1984', 'Le Petit Prince']` (stop excluded), the first two books, and the list reversed.

**Q7. Difference between `append`, `extend`, `insert`?**
`append(x)` adds one element at the end (`append([1, 2])` adds a list as a single element). `extend([1, 2])` adds each element. `insert(i, x)` puts `x` at position `i`. Example: `[1,2]` → `extend([3,4])` → `[1,2,3,4]`, then `insert(0, 9)` → `[9,1,2,3,4]`.

**Q8. Difference between `remove`, `pop`, `del` on a list? What errors can occur?**
`remove(value)` deletes the first element equal to `value` (`ValueError: list.remove(x): x not in list` if absent). `pop(i)` deletes by position and **returns** the element (`IndexError: pop index out of range` if `i` is too big). `del L[i]` deletes by position and returns nothing.

**Q9. In exercise 2.3 you remove "1984" from `livres`. What happens to the alignment with `auteurs` and `prix`?**
The three lists are linked by index only. If you remove only from `livres`, then `livres[1]` becomes "Le Petit Prince" but `auteurs[1]` is still "George Orwell". Remove at the same position in the three lists (`position = livres.index("1984")` then `pop(position)` on each).

**Q10. Why can't you write `livre[2] = 100` when `livre` is a tuple? What error do you get?**
Tuples are immutable. `TypeError: 'tuple' object does not support item assignment`.

**Q11. How do you write a tuple with only one element?**
With a trailing comma: `("a",)`. `("a")` is just the string `"a"`.

**Q12. What does `titre, auteur, prix = livre` do? What if the tuple has 4 elements?**
Unpacks the tuple into three variables. With 4 elements: `ValueError: too many values to unpack (expected 3)`.

**Q13. How do you build the list of tuples automatically from the three lists?**
`list(zip(livres, auteurs, prix))`. `zip` stops at the shortest list.

**Q14. In `d = {"1984": ("1984", "George Orwell", 150)}`, what is `d["1984"][2]`? And `d["1984"][1]`?**
`150` (price) and `"George Orwell"` (author).

**Q15. What do `.keys()`, `.values()`, `.items()` return?**
Views of the keys, the values, and the `(key, value)` pairs. Use `list(...)` to display them as lists.

**Q16. What happens with `d["Dune"]` if "Dune" does not exist? How do you avoid the error?**
`KeyError: 'Dune'`. Use `"Dune" in d`, or `d.get("Dune")` (returns `None`), or `d.get("Dune", "inconnu")` (returns your default).

**Q17. Does `"Albert Camus" in bibliotheque` return True?**
No. `in` on a dictionary tests the **keys** (the titles). The authors are inside the value tuples.

**Q18. Part 3, exercise 5: why does `bibliotheque["Le Petit Prince"][2] = 100` fail, and what is the correct code?**
The value is a tuple (immutable). Replace the whole tuple: `t, a, _ = bibliotheque["Le Petit Prince"]` then `bibliotheque["Le Petit Prince"] = (t, a, 100)`.

**Q19. What happens if you do `d["1984"] = (...)` when the key already exists?**
The old value is overwritten (no error, no duplicate): keys are unique.

**Q20. Difference between `del d[k]`, `d.pop(k)` and `d.pop(k, None)`?**
`del` removes (KeyError if missing). `pop(k)` removes and returns the value (KeyError if missing). `pop(k, None)` removes and returns the value, or returns `None` without error if missing.

**Q21. Can a list be a dictionary key? A tuple?**
A list cannot: `TypeError: unhashable type: 'list'`. A tuple can if all its elements are immutable: `{(1, 2): "ok"}` works.

**Q22. Explain `for titre_cle, (t, a, p) in bibliotheque.items():`.**
`.items()` yields `(key, value)` pairs. The value is a 3-tuple, so the second target `(t, a, p)` unpacks it at the same time. `titre_cle` = key, `t` = title (same as the key), `a` = author, `p` = price.

**Q23. After exercises 3.5 and 3.4, which books does exercise 3.8 ("prix supérieur à 100") display, and why is Le Petit Prince missing?**
"1984" (150) and "Les Misérables" (200). Le Petit Prince costs exactly 100 after exercise 5, and `100 > 100` is `False`. With `>=` it would appear. (L'Étranger was deleted in exercise 6.)

**Q24. What is the difference between `print` and `return` in a function?**
`print` shows text on screen and the caller gets nothing useful. `return` sends a value back so it can be stored or reused: `n = compter_livres(b)`. A function with no `return` gives `None`: `x = afficher_livres(b)` makes `x` equal to `None`.

**Q25. `ajouter_livre(biblio, ...)` does not use `return`. Why does the original dictionary change anyway?**
The function receives a reference to the same dictionary object (not a copy). `biblio[titre] = (...)` modifies that shared object.

**Q26. What would be wrong with this version of `ajouter_livre`?**
```python
def ajouter_livre(biblio, titre, auteur, prix):
    biblio = {titre: (titre, auteur, prix)}
```
It creates a new local dictionary and rebinds the local name `biblio`. The original dictionary is untouched (checked: the original stays `{'o': 1}`), and the existing books are lost inside the function. Modify with `biblio[titre] = ...` instead.

**Q27. Why is `if titre in biblio.values():` wrong in `chercher_livre`?**
The values are tuples `(titre, auteur, prix)`, so a string is never equal to a value. Test the keys: `if titre in biblio:`.

**Q28. What does `len(bibliotheque)` return, and why does it count books?**
The number of keys. One key per book, so it is the number of books.

**Q29. How do you build the dictionary of Part 3 from the three lists in one line?**
`{t: (t, a, p) for t, a, p in zip(livres, auteurs, prix)}` (a dictionary comprehension). Result for two books: `{'a': ('a', 'x', 1), 'b': ('b', 'y', 2)}`.

**Q30. Bonus: sort the books by price, and compute the total price of all books.**
`sorted(bibliotheque.values(), key=lambda b: b[2])` sorts the tuples by their 3rd element. `sum(b[2] for b in bibliotheque.values())` adds all the prices.

**Q31. You re-run the cell with `livres.remove("1984")` and get an error. Why?**
The book was already removed the first time, so `remove` raises `ValueError: list.remove(x): x not in list`. Use Restart & Run All to start from a clean state.

**Q32. In Part 3 the dictionary is called `bibliotheque`, like the list of tuples in Part 2. Is that a problem?**
It is allowed: the name now points to the dictionary and the list of tuples is lost. It is fine as long as you run the cells in order. If you need both, give them different names (`bibliotheque_liste`, `bibliotheque_dict`).

---

## 6. Where to learn more

- Official tutorial, sections "An informal introduction", "More control flow tools", "Data structures": https://docs.python.org/3/tutorial/
- In your notebook, `help(list.append)` or `list.append?` shows the doc of any function.
- Practice idea after the TP: add a `supprimer_livre(biblio, titre)` function and a `livres_chers(biblio, seuil)` function that returns the list of titles above a price.
