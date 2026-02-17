#!/usr/bin/env python3
"""Patch all 5 training scripts to 500K steps / large datasets for long runs"""

patches = {
    # CLIP: 50K -> 500K steps, 100K -> 1M samples
    "/data/organica-ai/training/train_multimodal_clip.py": [
        ("if step >= 50000:", "if step >= 500000:"),
        ("50000 - step", "500000 - step"),
        ("Step {step}/50000", "Step {step}/500000"),
        ("num_samples=100000", "num_samples=1000000"),
        ("if step % 5000 == 0:", "if step % 25000 == 0:"),
        ("multimodal_clip_step{step}.pt", "clip_500k_step{step}.pt"),
    ],
    # Sim2Real: 200K -> 500K epochs, 50K -> 100K samples
    "/data/organica-ai/training/train_sim2real_h100.py": [
        ("def train(epochs=200000, batch_size=32):", "def train(epochs=500000, batch_size=32):"),
        ("train(epochs=200000, batch_size=32)", "train(epochs=500000, batch_size=32)"),
        ("if epoch % 50000 == 0:", "if epoch % 100000 == 0:"),
        ("num_samples=100000", "num_samples=500000"),
    ],
    # Safety: 50K -> 500K steps, 100K -> 1M samples
    "/data/organica-ai/training/train_safety_classifier.py": [
        ("if step >= 50000:", "if step >= 500000:"),
        ("50000 - step", "500000 - step"),
        ("Step {step}/50000", "Step {step}/500000"),
        ("num_samples=100000", "num_samples=1000000"),
        ("if step % 5000 == 0:", "if step % 25000 == 0:"),
        ("safety_classifier_step{step}.pt", "safety_500k_step{step}.pt"),
    ],
    # Speech2Action: 200K -> 500K steps
    "/data/organica-ai/training/train_speech2action_h100.py": [
        ("TOTAL_STEPS = 200000", "TOTAL_STEPS = 500000"),
    ],
}

for filepath, replacements in patches.items():
    try:
        with open(filepath, "r") as f:
            content = f.read()
        original = content
        count = 0
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                count += 1
        if content != original:
            with open(filepath, "w") as f:
                f.write(content)
            name = filepath.split("/")[-1]
            print("PATCHED " + name + " (" + str(count) + "/" + str(len(replacements)) + " changes)")
        else:
            name = filepath.split("/")[-1]
            print("NO CHANGES " + name)
    except Exception as e:
        name = filepath.split("/")[-1]
        print("ERROR " + name + ": " + str(e))

print("")
print("Done! All scripts patched to 500K steps.")
