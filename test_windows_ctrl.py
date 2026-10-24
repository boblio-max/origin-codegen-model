from windows_controller import WindowsController

# Test local execution
wc = WindowsController(hostname='localhost')
wc.connect()

# Test basic command
output = wc.run_command('echo Hello from Windows controller')
print('Local cmd output:')
print(output)

# Test PowerShell
output_ps = wc.run_powershell('Write-Host "Hello from PowerShell"')
print('PowerShell output:')
print(output_ps)

wc.close()
print('SUCCESS: WindowsController works!')