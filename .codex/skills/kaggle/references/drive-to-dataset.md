# Google Drive folder to Kaggle Dataset

Usar este flujo cuando el usuario quiera subir una carpeta de Google Drive a un Kaggle Dataset y la carpeta sea grande o no deba descargarse en local.

## Datos que pedir

Pedir solo lo necesario:

- Google Drive folder URL o folder id.
- Kaggle dataset id en formato `owner/dataset-slug`.
- Kaggle kernel id en formato `owner/kernel-slug`.

Opcionales:

- `title`
- `subtitle`
- `description`
- `license` (default: `CC0-1.0`)
- privacidad del dataset: privado por default, publico solo si el usuario lo pide.
- modo: `auto`, `create` o `version` (default: `auto`).

## Regla principal

No descargar la carpeta de Drive al equipo local. Generar un kernel/script liviano y ejecutar `kaggle kernels push`; la descarga de Drive y la subida del dataset deben ocurrir dentro de Kaggle.

## Archivos a generar

Crear una carpeta de trabajo local, por ejemplo:

```text
kaggle/
└── drive_to_dataset/
    └── input/
        ├── main.py
        └── kernel-metadata.json
```

Usar como base:

- `../templates/drive_to_dataset_main.py`
- `../templates/drive_to_dataset_kernel-metadata.json`

Reemplazar placeholders:

```text
__DRIVE_FOLDER__
__DATASET_ID__
__DATASET_TITLE__
__DATASET_SUBTITLE__
__DATASET_DESCRIPTION__
__DATASET_LICENSE__
__UPLOAD_MODE__
__DIR_MODE__
__PUBLIC_BOOL__
__VERSION_MESSAGE__
__KERNEL_ID__
__KERNEL_TITLE__
```

## Autenticacion en Kaggle

No escribir credenciales en `main.py`, `kernel-metadata.json`, logs ni repositorio.

El script remoto lee:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `KAGGLE_API_TOKEN` como alias de `KAGGLE_KEY`

El usuario debe agregarlos como Kaggle Notebook Secrets o variables de entorno disponibles para el kernel. Si el primer `kaggle kernels push` crea el kernel pero falla por secrets, indicar que abra el kernel en Kaggle, agregue los secrets y vuelva a ejecutar o haga otro push.

## Ejecucion local

Antes:

```powershell
kaggle --version
```

Subir/ejecutar:

```powershell
kaggle kernels push -p kaggle\drive_to_dataset\input
```

## Monitoreo

Usar el slug real del notebook desde la URL de Kaggle si el slug esperado falla.

```powershell
kaggle kernels list --mine --search "drive" --page-size 20
kaggle kernels status <owner/kernel-slug>
$env:PYTHONIOENCODING='utf-8'
kaggle kernels logs <owner/kernel-slug>
```

`PYTHONIOENCODING=utf-8` evita errores `charmap` en Windows cuando los logs contienen barras de progreso Unicode.

## Handoff

Reportar:

- kernel slug real.
- estado final.
- dataset id.
- enlace del dataset si aparece en logs.
- archivos detectados o subidos segun logs.
- error exacto si falla por secrets, permisos Drive, metadata o cuota/disco Kaggle.
