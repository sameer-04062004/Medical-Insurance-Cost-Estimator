import subprocess
import sys
import os

python_exe = r"C:\Users\Value PAKISTAN\AppData\Local\Programs\Python\Python312\python.exe"
if not os.path.exists(python_exe):
    python_exe = sys.executable

print(f"Launching Streamlit application with: {python_exe}")
cmd = [python_exe, "-m", "streamlit", "run", "app.py", "--server.headless", "false"]
subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
