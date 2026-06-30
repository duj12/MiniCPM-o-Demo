"""Check C++ log for video session details."""
import sys

with open("/app/logs/llama-omni-server.log") as f:
    lines = f.readlines()

# Find the latest session that matches
target = sys.argv[1] if len(sys.argv) > 1 else "95416e60"
for i, line in enumerate(lines):
    if target in line:
        # Print 5 lines before and 20 lines after
        start = max(0, i - 5)
        end = min(len(lines), i + 20)
        for j in range(start, end):
            print(f"{j}: {lines[j].rstrip()}")
        print("---")
