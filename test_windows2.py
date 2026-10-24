import sys
sys.path.insert(0, r'C:\Users\smile\.config\opencode\python-tools')
from windows_controller import WindowsController

wc = WindowsController(hostname='localhost')
wc.connect()

# Test basic command
output = wc.run_command('echo Hello from Windows controller')
print('Local cmd output:', output)

# Test PowerShell - use single quotes inside, escape properly
output_ps = wc.run_powershell('Write-Host \'Hello from PowerShell\'')
print('PowerShell output:', output_ps)

wc.close()
print('SUCCESS: WindowsController works!')