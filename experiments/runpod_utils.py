import subprocess
import sys


def stop_runpod():
    """Stop the RunPod to save $$. Files should persist."""
    try:
        result = subprocess.run(["runpodctl", "stop", "pod"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("Pod stopped!")
        else:
            print(f"Runpod stop failed: {result.stderr}, trying fallback stop methods...")
            
            
            sys.exit(1)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("runpodctl not available, trying fallback stop methods...")
        _fallback_stop()

def _fallback_stop():
    try:
        subprocess.run(["curl", "-X", "POST", "http://localhost:8000/stop"], 
                      timeout=10, capture_output=True)
        print("Attempted API stop, double check this worked manually")
    except:
        print("API stop failed, go stop manually so you don't get charged $$$")
    sys.exit(1)