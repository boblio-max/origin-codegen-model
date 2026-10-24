import sys
sys.path.insert(0, r'C:\Users\smile\.config\opencode\python-tools')
from windows_controller import WindowsController

wc = WindowsController(hostname='192.168.1.17')
wc.connect()

print("=" * 60)
print("NICKELODEN PC (192.168.1.17) INFO")
print("=" * 60)

# 1. Architecture
output = wc.run_command('echo %PROCESSOR_ARCHITECTURE%')
print(f"\nArchitecture: {output.strip()}")

# 2. OS Version
output = wc.run_command('ver')
print(f"\nOS Version: {output.strip()}")

# 3. System root
output = wc.run_command('echo %SystemRoot%')
print(f"System Root: {output.strip()}")

# 4. NVIDIA folder check
output = wc.run_command('dir C:\\Program Files\\NVIDIA\\')
has_nvidia = 'File Not Found' not in output
print(f"\nNVIDIA folder exists: {has_nvidia}")

# 5. PATH check for NVIDIA
output = wc.run_command('echo %PATH%')
lines = output.split('\n')
nvidia_paths = [l for l in lines if 'NVIDIA' in l]
print(f"NVIDIA in PATH: {len(nvidia_paths)} entries")

# 6. Processor name from environment
output = wc.run_command('set PROCESSOR')
for line in output.split('\n'):
    if '=' in line:
        print(f"Processor env: {line}")

# 7. Disk drives alternative
output = wc.run_command('dir C:\\')
drives_line = [l for l in output.split('\n') if 'Volume Serial' in l]
print(f"\nDisk info: {drives_line[:2]}")

# 8. Installed software count (quick look)
output = wc.run_command('dir /b "C:\\Program Files" 2>nul | find /i "NVIDIA"')
print(f"\nNVIDIA in Program Files: {'Found' if output.strip() else 'Not found'}")

# 9. Try systeminfo summary
output = wc.run_command('systeminfo | find /i "OS Name"')
print(f"\nOS Name: {output.strip()}")

wc.close()
print("\n" + "=" * 60)