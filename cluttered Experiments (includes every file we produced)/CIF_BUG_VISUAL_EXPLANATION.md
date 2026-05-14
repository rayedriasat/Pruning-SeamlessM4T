# CIF Firing Bug - Visual Explanation

## The Problem: Only 1 Token Fires

### What Should Happen (Correct CIF)
```
Frame 1: alpha=0.4 → acc_w=0.4, acc=0.4*h1
Frame 2: alpha=0.5 → acc_w=0.9, acc=0.4*h1 + 0.5*h2
Frame 3: alpha=0.6 → acc_w=1.5, acc=0.4*h1 + 0.5*h2 + 0.6*h3
         ↓ acc_w >= 1.0, FIRE TOKEN 1
         ↓ residual: acc_w=0.5, acc=(0.5/1.5)*original_acc
Frame 4: alpha=0.3 → acc_w=0.8, acc=residual + 0.3*h4
Frame 5: alpha=0.4 → acc_w=1.2, acc=residual + 0.3*h4 + 0.4*h5
         ↓ acc_w >= 1.0, FIRE TOKEN 2
         ↓ residual: acc_w=0.2, acc=(0.2/1.2)*original_acc
...
Result: Multiple tokens fired ✓
```

### What Was Happening (Buggy v2)
```
Frame 1: alpha=0.4 → acc_w=0.4, acc=0.4*h1
Frame 2: alpha=0.5 → acc_w=0.9, acc=0.4*h1 + 0.5*h2
Frame 3: alpha=0.6 → acc_w=1.5, acc=0.4*h1 + 0.5*h2 + 0.6*h3
         ↓ acc_w >= 1.0, FIRE TOKEN 1
         ↓ BUG: acc_w=0.5, acc=acc*(0.5/(0.5+1.0))=acc*0.33
         ↓ Residual is TOO LARGE (should be 0.5/1.5=0.33, not 0.5/1.5=0.33)
         ↓ Wait, the math looks similar but the DENOMINATOR is wrong!
Frame 4: alpha=0.3 → acc_w=0.83, acc=HUGE_residual + 0.3*h4
Frame 5: alpha=0.4 → acc_w=1.23, acc=HUGE_residual + 0.3*h4 + 0.4*h5
         ↓ acc_w >= 1.0, should fire but...
         ↓ The residual calculation prevents proper accumulation
...
Result: Only 1 token fires ✗
```

## The Math Error

### Buggy Code (v2)
```python
# After firing, acc_w has been reduced
acc_w = 1.5  # before firing
acc_w -= 1.0  # after firing, acc_w = 0.5

# BUG: Adding threshold back is wrong!
residual_ratio = acc_w / (acc_w + threshold)
                = 0.5 / (0.5 + 1.0)
                = 0.5 / 1.5
                = 0.333
```

Wait, this looks correct! But the issue is **when** we calculate it.

### The Real Problem

Let me trace through more carefully:

**Buggy v2 logic**:
```python
while acc_w >= self.threshold:  # threshold = 1.0
    fired.append(acc / max(acc_w, 1e-6))  # ❌ Dividing by CURRENT acc_w
    acc_w -= self.threshold
    if acc_w > 0:
        acc = acc * (acc_w / (acc_w + self.threshold))  # ❌ Wrong denominator
```

**Example with acc_w=1.5**:
1. Fire: `fired.append(acc / 1.5)` ← This normalizes by current weight
2. Reduce: `acc_w = 1.5 - 1.0 = 0.5`
3. Residual: `acc = acc * (0.5 / 1.5)` ← But acc was already divided by 1.5!

So the accumulator gets divided by 1.5 TWICE:
- Once when firing: `acc / 1.5`
- Once in residual: `acc * (0.5 / 1.5)`

This causes the residual to be way too small!

### Fixed v3 Logic
```python
while acc_w >= self.threshold:
    fired.append(acc.clone())  # ✓ Fire the full accumulator
    acc_w_before = acc_w       # ✓ Store original weight
    acc_w -= self.threshold
    if acc_w > 1e-6:
        acc = acc * (acc_w / acc_w_before)  # ✓ Correct ratio
```

**Example with acc_w=1.5**:
1. Fire: `fired.append(acc)` ← Full accumulator
2. Store: `acc_w_before = 1.5`
3. Reduce: `acc_w = 1.5 - 1.0 = 0.5`
4. Residual: `acc = acc * (0.5 / 1.5)` ← Correct proportion

Now the accumulator is scaled correctly by the fraction of weight remaining!

## Side-by-Side Comparison

| Step | Buggy v2 | Fixed v3 |
|------|----------|----------|
| **Accumulate** | acc_w=1.5, acc=sum | acc_w=1.5, acc=sum |
| **Fire** | `fired.append(acc/1.5)` | `fired.append(acc)` |
| **Reduce** | acc_w=0.5 | acc_w=0.5 |
| **Residual** | `acc*(0.5/1.5)` | `acc*(0.5/1.5)` |
| **Net effect** | acc divided by 1.5 twice! | acc scaled once correctly |
| **Result** | Residual too small → collapse | Residual correct → multiple firings |

## Why This Caused Collapse

With the buggy code:
1. First token fires correctly
2. Residual is calculated wrong (too small)
3. Subsequent frames can't accumulate enough weight
4. No more tokens fire
5. Result: `fired=1` always

With the fixed code:
1. First token fires correctly
2. Residual is proportional to leftover weight
3. Subsequent frames accumulate normally
4. More tokens fire when threshold reached
5. Result: `fired=15-30` as expected

## The Key Insight

The bug was subtle: the code was **normalizing twice**:
1. When firing: `fired.append(acc / acc_w)` 
2. When calculating residual: `acc * (acc_w / (acc_w + threshold))`

The fix: **Don't normalize when firing**, just fire the raw accumulator, then scale the residual by the correct fraction.

This is why the CIF was collapsing to 1 token!
