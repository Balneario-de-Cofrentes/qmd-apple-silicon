# QMD Query Expansion - Apple Silicon Edition

## Objetivo
Adaptar el sistema de fine-tuning de [tobi/qmd](https://github.com/tobi/qmd/tree/main/finetune) para funcionar en Apple Silicon (M1/M2/M3/M4) usando MLX.

## Contexto
- El código original usa PyTorch + bitsandbytes (optimizado para CUDA)
- Apple Silicon no tiene CUDA, usa Metal vía MLX
- MLX tiene soporte para LoRA fine-tuning via `mlx-lm`

## Requisitos

### 1. Training Script (`train.py`)
- Usar MLX/mlx-lm para fine-tuning
- Soportar LoRA (mismo approach que el original)
- Dos modos: `sft` (supervised) y `grpo` (RL refinement)
- Configuración via YAML (compatible con formato original)

### 2. Dataset
- Usar el mismo dataset de HuggingFace: `tobil/qmd-query-expansion-train-v2`
- Formato de salida esperado:
```
lex: keyword expansion 1
lex: keyword expansion 2
vec: natural language query
vec: another natural query
hyde: A hypothetical document passage (~100 chars)
```

### 3. Modelo Base
- Qwen3-1.7B (o versión más pequeña si memoria es issue)
- Convertir a formato MLX si necesario

### 4. Output
- Modelo entrenado compatible con Ollama (GGUF)
- Script de conversión incluido

### 5. Evaluación
- Script `eval.py` para probar expansiones
- Métricas de scoring del original

## Estructura de Archivos
```
qmd-apple-silicon/
├── README.md           # Documentación
├── train.py            # Script principal de training
├── eval.py             # Evaluación
├── convert.py          # Conversión a GGUF
├── configs/
│   ├── sft.yaml
│   └── grpo.yaml
├── requirements.txt    # Dependencias
└── examples/           # Ejemplos de uso
```

## Referencias
- Original: `~/clawd-shared/tools/qmd/finetune/`
- MLX LoRA: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm
- MLX fine-tuning: https://ml-explore.github.io/mlx/build/html/examples/llama-lora.html

## Criterio de Éxito
1. `python train.py sft` entrena sin errores en Mac
2. `python train.py grpo` mejora el modelo SFT
3. `python eval.py` produce expansiones coherentes
4. Modelo final funciona en Ollama

## Notas
- Priorizar que funcione sobre optimización
- Documentar bien para potencial open source
- Testear en Mac mini M-series
