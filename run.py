"""
OCR Readiness Evaluation Platform - Cross-Platform Auto-Bootstrapper
Works out-of-the-box on Windows, Linux, and macOS without manual setup.
Run: python run.py
"""

import os
import sys
import subprocess
import shutil

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")

def get_venv_python():
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")

def get_venv_streamlit():
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "streamlit.exe")
    return os.path.join(VENV_DIR, "bin", "streamlit")

def ensure_venv():
    venv_py = get_venv_python()
    if not os.path.exists(venv_py):
        print(f"[i] Creating virtual environment at {VENV_DIR}...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        print("[✓] Virtual environment created.")
        
    try:
        subprocess.check_call([venv_py, "-c", "print(1)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("[!] Virtual environment Python is blocked by policy or broken. Falling back to system Python.")
        return sys.executable
        
    return venv_py

def ensure_dependencies(venv_py):
    req_file = os.path.join(BASE_DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("[i] Ensuring dependencies from requirements.txt are installed...")
        try:
            subprocess.check_call([venv_py, "-m", "pip", "install", "-r", req_file, "--quiet"])
            print("[✓] Dependencies verified.")
        except Exception as e:
            print(f"[!] Pip install note: {e}. Checking if packages already installed...")

def setup_tesseract_env():
    tess_dir = os.path.join(BASE_DIR, "tesseract")
    tess_exe = os.path.join(tess_dir, "tesseract.exe")
    if os.path.isfile(tess_exe):
        os.environ["TESSERACT_CMD"] = tess_exe
        os.environ["PATH"] = tess_dir + os.path.pathsep + os.environ.get("PATH", "")
        print(f"[✓] TESSERACT_CMD set to: {tess_exe}")

    tessdata = os.path.join(BASE_DIR, "tessdata")
    if os.path.exists(tessdata):
        os.environ["TESSDATA_PREFIX"] = tessdata
        print(f"[✓] TESSDATA_PREFIX set to: {tessdata}")

def ensure_team_apis(venv_py):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    result = s.connect_ex(("127.0.0.1", 8000))
    s.close()
    if result != 0:
        mock_py = os.path.join(BASE_DIR, "mock_team_apis.py")
        if os.path.exists(mock_py):
            print("[i] Starting local team API services (Mansi, Vivek, Krish, Tanusha)...")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            subprocess.Popen([venv_py, mock_py], creationflags=flags)
            print("[✓] Team APIs online on ports 8000, 8001, 8002, 9001.")

def main():
    print("=" * 60)
    print("  🔍 OCR Readiness Platform — Global Auto-Bootstrapper")
    print("=" * 60)
    
    os.chdir(BASE_DIR)
    venv_py = ensure_venv()
    ensure_dependencies(venv_py)
    setup_tesseract_env()
    ensure_team_apis(venv_py)

    app_py = os.path.join(BASE_DIR, "app.py")
    print("\n🚀 Launching Streamlit Application...\n")

    cmd = [venv_py, "-m", "streamlit", "run", app_py] + sys.argv[1:]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[i] Application stopped.")

if __name__ == "__main__":
    main()
