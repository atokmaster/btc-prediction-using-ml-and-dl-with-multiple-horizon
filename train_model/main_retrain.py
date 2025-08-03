import subprocess
import sys
import os

train_scripts = [
    "train_6h.py",
    "train_12h.py",
    "train_1d.py",
    "train_3d.py",
    "train_7d.py"
]

base_path = "G:/BTC_AI_SIgnal"
script_path_folder = os.path.join(base_path, "train_model")

for script in train_scripts:
    script_path = os.path.join(script_path_folder, script)
    print(f"\n🔹 Menjalankan: {script} ...")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True,
            cwd=base_path  # ⬅️ fix penting agar modul dan path bekerja
        )
        print(f"✅ Sukses: {script}")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"❌ Gagal: {script}")
        print("Output error:")
        print(e.stderr)
