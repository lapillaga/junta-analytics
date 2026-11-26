# Instrucciones de revisión

El `README.md` contiene los estándares del proyecto. Este archivo indica qué debe pesar una revisión.

## Always check

- **Credenciales y datos personales:** ninguno en notebooks (celdas ni salidas guardadas), código, `.env.example` ni workflows. Las credenciales de la base de datos van en `.env`, que no se versiona.
- **Notebooks:** se ejecutan de arriba hacia abajo y en el orden que indica el README; las salidas guardadas no exponen datos reales de personas ni cadenas de conexión.
- **Datos y modelos:** no se versionan los archivos que `.gitignore` excluye: `data/raw/*.csv`, `data/processed/*.csv`, `data/processed/*.pkl`, `data/models/*.pkl` y `data/models/*.joblib`.
- **Estilo:** el código Python pasa `black` y `flake8`, como indica el README.
- **Textos en español:** correctos (acentos, concordancia).

## Important (fix before merge)

- Cualquier incumplimiento de un punto anterior.
- Una credencial, clave o cadena de conexión en cualquier archivo, o datos reales de personas en una salida de notebook.
- Un archivo de los que `.gitignore` excluye (`data/raw/*.csv`, `data/processed/*.csv`, `data/processed/*.pkl`, `data/models/*.pkl`, `data/models/*.joblib`) agregado al repositorio.

## Nits

Informe como máximo cinco. Omita lo que un linter o formateador ya hace cumplir.

## Skip

No hay archivos generados que omitir.
