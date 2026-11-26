# Annotations

Human labels used by `evaluate.py`. Format:

```json
{
  "my_climb.mov": [
    {"kind": "foot_slip", "start": 25.9, "end": 26.1, "side": "left"},
    {"kind": "bent_arms", "start": 10.0, "end": 12.5, "side": "both"}
  ]
}
```

`kind` is one of `bent_arms`, `foot_slip`, `foot_readjust`; times are in seconds.
