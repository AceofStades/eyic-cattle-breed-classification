import os
import subprocess
import sys
import time


def main():
    with open("training_log.txt", "w") as f:
        f.write("--- TRAINING SESSION STARTED ---\n")

    print("🚀 Starting Parallel Training with Color Coded Logs...")
    print("\033[96mLarge Model = Teal\033[0m")
    print("\033[95mSmall Model = Magenta\033[0m")

    start_time = time.time()

    p_large = subprocess.Popen([sys.executable, "mobilenet_large.py"])
    p_small = subprocess.Popen([sys.executable, "mobilenet_small.py"])

    p_large.wait()
    p_small.wait()

    total_time = time.time() - start_time
    print(
        f"\n✅ All training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s"
    )
    print("Check 'training_log.txt' for results.")


if __name__ == "__main__":
    main()
