# DataFrame Utilities

Small, chain-friendly utilities for PySpark DataFrames.

```python
from dfx import DFx

cleaned = (
    DFx(raw_df)
    .clean_column_names()
    .normalize_column("status", "UNKNOWN", "string")
    .unwrap()
)
```

## Development

```powershell
python -m pytest
python -m pip wheel . -w dist --no-deps
```
