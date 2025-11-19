# Checkpoints

Coloca en este directorio los pesos `.pth` que quieras utilizar con la aplicación de Streamlit.

Ejemplo (PowerShell en Windows):

```powershell
New-Item -ItemType Directory -Path .\checkpoints -Force | Out-Null
Invoke-WebRequest -Uri "https://your-storage/vitpose-l.pth" -OutFile ".\checkpoints\vitpose-l.pth"
```

Desde la interfaz selecciona la ruta absoluta o deja el archivo aquí para que se detecte automáticamente.
