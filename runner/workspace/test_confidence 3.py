
import time
import sys

print("Hello from NIS Protocol Runner!")
print(f"Current time: {time.time()}")
print(f"Python version: {sys.version}")

# Calculate confidence demo
factors = [0.8, 0.9, 0.7]
confidence = sum(factors) / len(factors)
print(f"Calculated confidence: {confidence:.3f}")
