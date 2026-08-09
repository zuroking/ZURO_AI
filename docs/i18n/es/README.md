# Kronos Synapse Dialogue Core

![Python](https://img.shields.io/badge/python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Params](https://img.shields.io/badge/params-%7E15.3M-informational)

**Idiomas:** [English](../../../README.md) · **Español** · [Deutsch](../de/README.md) · [中文](../zh/README.md) · [Русский](../ru/README.md)

Un modelo de lenguaje de tipo GPT solo-decodificador (~15,3 M de parámetros) escrito desde
cero sobre primitivas puras de `torch.nn` y entrenado por completo en CPU. Incluye un
tokenizador BPE, una canalización de entrenamiento respaldada por memmap y un chat de CLI
en streaming construido en torno a un generador con caché KV. Toda la pila —modelo, datos,
entrenamiento, inferencia— está implementada a mano, sin el andamiaje de `transformers`
ni de `nanoGPT`.

> **Qué es (y qué no es).** Este modelo hace **continuación de texto**, no
> seguimiento de instrucciones. Está entrenado con prosa literaria (Project Gutenberg), no
> con pares de diálogo anotados. Tú escribes texto; él lo continúa. No es un asistente de
> rol. Consulta [Paradigma de chat y limitaciones](#paradigma-de-chat-y-limitaciones).

---

## Índice

- [Inicio rápido](#inicio-rápido)
- [Puntos destacados](#puntos-destacados)
- [Arquitectura](#arquitectura)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Preparar los datos](#preparar-los-datos)
- [Entrenamiento](#entrenamiento)
- [Evaluación](#evaluación)
- [Chat e inferencia](#chat-e-inferencia)
- [Paradigma de chat y limitaciones](#paradigma-de-chat-y-limitaciones)
- [Pruebas](#pruebas)
- [Desarrollo](#desarrollo)
- [Notas de rendimiento](#notas-de-rendimiento)
- [Hoja de ruta](#hoja-de-ruta)
- [Preguntas frecuentes](#preguntas-frecuentes)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## Inicio rápido

Si solo quieres verlo funcionar, aquí tienes todo el ciclo de principio a fin:

```bash
# 1. instalar
pip install -e ".[dev]"

# 2. coloca algunos libros .txt en data/raw/gutenberg/, luego:
kronos_synapse tokenize --corpus-dir data/raw/gutenberg

# 3. entrenar (Ctrl+C es seguro; puedes reanudar desde un checkpoint más tarde)
kronos_synapse train

# 4. conversar con él
kronos_synapse chat checkpoints/best.pt
```

También hay un `Makefile` si prefieres `make tokenize`, `make train`, `make chat`.
Todo se ejecuta en CPU: sin GPU, sin cuenta en la nube, sin claves de API.

## Puntos destacados

- **Transformer hecho a mano.** `CausalSelfAttention`, `MLP`, `TransformerBlock` y
  `MiniGPT` se construyen directamente sobre `nn.Linear`, `nn.LayerNorm`, `nn.Embedding` y
  `nn.Dropout`. La atención pasa por `F.scaled_dot_product_attention`, que selecciona
  el kernel fusionado optimizado en CPU.
- **CPU primero.** Todo funciona sin CUDA. El bucle de entrenamiento hila fino con
  acumulación de gradiente, autocast en `bfloat16`, número de hilos configurable y un
  respaldo opcional con `torch.compile`.
- **Generación con caché KV.** La inferencia hace una pasada de prellenado sobre el prompt
  y luego decodifica un token cada vez contra las claves/valores en caché: la diferencia
  entre un chat usable e inusable en la CPU de un portátil.
- **Muestreo que puedes ajustar.** Temperatura, top-k, top-p (nucleus) y penalización por
  repetición están todos expuestos como flags de la CLI.
- **Configuración tipada.** Tres configuraciones de `pydantic-settings` (`ModelConfig`,
  `TrainConfig`, `DataConfig`) reemplazan los diccionarios mágicos por ajustes validados e
  inmutables.

## Arquitectura

Un decodificador GPT pre-LN estándar. La tabla siguiente es el `ModelConfig` por defecto.

| Componente | Valor | Notas |
|---|---|---|
| `vocab_size` | 12.000 | BPE, entrenado desde cero |
| `d_model` | 384 | ancho de embedding / oculto |
| `n_layers` | 6 | bloques transformer |
| `n_heads` | 6 | dim. por cabeza = 64 |
| `d_ff` | 1.536 | 4× `d_model`, MLP con GELU |
| `context_length` | 256 | embeddings posicionales absolutos aprendidos |
| `dropout` | 0.1 | embeddings, salida de atención, salida de MLP |
| `lm_head` | atado a `tok_emb` | ahorra ~4,6 M de parámetros |
| Kernel de atención | `F.scaled_dot_product_attention` | fusionado, causal en entrenamiento |
| Init. de pesos | esquema GPT-2 | `N(0, 0.02)`, proyecciones residuales escaladas por `1/√(2·n_layers)` |

**Presupuesto de parámetros (~15,3 M):**

| Componente | Parámetros |
|---|---|
| Embedding de tokens (atado con `lm_head`) | 4.608.000 |
| Embedding posicional | 98.304 |
| 6 × bloque transformer (attn + MLP + 2× LayerNorm) | 10.626.048 |
| LayerNorm final | 768 |
| **Total** | **≈ 15.333.120** |

El recuento se registra al inicio de cada ejecución de entrenamiento, para que puedas
confirmar que queda dentro del ±5 % del objetivo.

## Estructura del proyecto

```text
kronos-synapse-dialogue/
├── pyproject.toml          # dependencias + punto de entrada de consola
├── config.py               # ModelConfig / TrainConfig / DataConfig (pydantic-settings)
├── cli.py                  # punto de entrada Typer: tokenize / train / eval / chat
├── tokenizer/
│   ├── trainer.py          # entrenamiento BPE (HF tokenizers, ByteLevel)
│   └── wrapper.py          # KronosTokenizer: encode / decode / turnos de diálogo
├── model/
│   ├── attention.py        # CausalSelfAttention + KVCache
│   ├── mlp.py              # feedforward con GELU
│   ├── block.py            # TransformerBlock pre-LN
│   └── gpt.py              # MiniGPT (embeddings + bloques + lm_head atado)
├── data/
│   ├── sources.py          # TextCorpusSource ABC + GutenbergCorpusSource
│   ├── packing.py          # corpus → división por libro → memmap uint16
│   └── dataset.py          # BinaryTokenDataset + fábrica de DataLoader
├── training/
│   ├── scheduler.py        # decaimiento coseno con warmup lineal
│   ├── checkpoint.py        # guardar / cargar modelo + optimizador + scheduler
│   └── loop.py             # bucle de entrenamiento, métricas, progreso con Rich
├── inference/
│   ├── generate.py         # generador de muestreo con caché KV
│   └── conversation.py     # ConversationBuffer (recorte por límite de frase)
├── tests/                  # pruebas unitarias de shapes, máscara causal, packing, buffer
└── data/
    ├── raw/gutenberg/      # <-- aquí colocas los archivos .txt del corpus
    └── processed/          # generados: tokenizer.json, train.bin, val.bin
```

## Requisitos

- **Python** 3.12 o superior
- **PyTorch** 2.3+ (build de CPU; el autocast en `bfloat16` se emula en Alder Lake pero
  aun así ahorra ancho de banda de memoria; hay un respaldo en fp32)
- **Hardware**: una CPU multinúcleo moderna. El objetivo de desarrollo fue un Intel i7-1225U
  (2P+8E, AVX2, sin AVX-512). No se requiere GPU.

## Instalación

```bash
git clone <repo-url> kronos-synapse-dialogue
cd kronos-synapse-dialogue
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Esto registra el comando de consola `kronos_synapse` e incorpora las herramientas de
desarrollo (`pytest`, `mypy`).

## Configuración

Todos los ajustes viven en [`config.py`](../../../config.py) como modelos inmutables de
`pydantic-settings`. Edita los valores por defecto allí para reajustar el modelo o el
programa de entrenamiento.

### `ModelConfig`

| Opción | Por defecto | Descripción |
|---|---|---|
| `vocab_size` | `12_000` | tamaño del vocabulario BPE |
| `d_model` | `384` | ancho oculto |
| `n_layers` | `6` | bloques transformer |
| `n_heads` | `6` | cabezas de atención |
| `d_ff` | `1536` | ancho interno de la MLP |
| `context_length` | `256` | longitud máxima de secuencia |
| `dropout` | `0.1` | probabilidad de dropout |

### `TrainConfig`

| Opción | Por defecto | Descripción |
|---|---|---|
| `num_threads` | `8` | `torch.set_num_threads` |
| `batch_size` | `16` | tamaño de micro-lote |
| `grad_accum_steps` | `8` | lote efectivo = `16 × 8 = 128` |
| `max_iters` | `50_000` | pasos del optimizador |
| `lr` | `3e-4` | tasa de aprendizaje pico |
| `lr_min` | `1e-5` | suelo del coseno |
| `warmup_iters` | `200` | pasos de warmup lineal |
| `weight_decay` | `0.1` | aplicado solo a parámetros 2D |
| `max_grad_norm` | `1.0` | recorte de gradiente |
| `use_bf16` | `True` | autocast en `bfloat16` en CPU |
| `compile_model` | `False` | `torch.compile` opcional con respaldo en fp32 |
| `checkpoint_every` | `1_000` | cadencia de checkpoint + evaluación |
| `checkpoint_dir` | `checkpoints/` | directorio de salida |
| `log_file` | `logs/train.jsonl` | registro de métricas por checkpoint |

### `DataConfig`

| Opción | Por defecto | Descripción |
|---|---|---|
| `raw_dir` | `data/raw/gutenberg/` | corpus `.txt` de entrada |
| `processed_dir` | `data/processed/` | salida `train.bin` / `val.bin` |
| `tokenizer_dir` | `data/processed/tokenizer/` | salida `tokenizer.json` |
| `min_quote_line_ratio` | `0.0` | filtro opcional de densidad de diálogo |
| `split_seed` | `42` | semilla de la división por libro |
| `val_fraction` | `0.10` | libros reservados (no vistos en entrenamiento) |

## Preparar los datos

1. Coloca archivos `.txt` de Project Gutenberg en `data/raw/gutenberg/`. (Las novelas del
   siglo XIX y principios del XX —Austen, Dickens, Twain, etc.— funcionan bien: mucho
   diálogo directo e intercambios cortos.)
2. Entrena el tokenizador y empaqueta el corpus en divisiones binarias mapeadas en memoria:

```bash
kronos_synapse tokenize --corpus-dir data/raw/gutenberg
```

Esto produce tres artefactos en `data/processed/`:

- `tokenizer/tokenizer.json` — el tokenizador BPE
- `train.bin` — tokens de entrenamiento como memmap `uint16`
- `val.bin` — tokens de validación (libros enteros reservados, sin fuga de ventanas)

La limpieza del corpus ocurre automáticamente: se elimina el texto repetitivo de
cabecera/pie de Project Gutenberg, el texto se normaliza en Unicode-NFKC y las secuencias
de espacios en blanco se colapsan. Pon `min_quote_line_ratio` por encima de 0 para
conservar solo documentos densos en diálogo entrecomillado.

Los tokens especiales son fijos: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`. No se añaden
tokens de rol (`<user>`/`<bot>`); consulta la nota sobre el
[paradigma de chat](#paradigma-de-chat-y-limitaciones).

## Entrenamiento

```bash
kronos_synapse train
```

El bucle imprime el recuento de parámetros al inicio y luego se ejecuta con una barra de
progreso de Rich. Cada `checkpoint_every` pasos:

- evalúa la pérdida de validación,
- añade `{step, train_loss, val_loss}` a `logs/train.jsonl`,
- escribe `checkpoints/step_<n>.pt`,
- actualiza `checkpoints/best.pt` si la pérdida de validación mejoró.

**Reanuda** desde cualquier checkpoint (se restauran el estado del optimizador y del
scheduler):

```bash
kronos_synapse train --resume checkpoints/best.pt
```

## Evaluación

Evalúa un checkpoint sobre los libros de validación reservados:

```bash
kronos_synapse eval checkpoints/best.pt --split val
```

Informa la pérdida de entropía cruzada y la perplejidad (`2^loss`) sobre la división.

## Chat e inferencia

```bash
kronos_synapse chat checkpoints/best.pt --temperature 0.8 --top-p 0.9
```

La generación se transmite token a token en la terminal. La primera llamada prellena el
prompt y construye la caché KV; cada token posterior se decodifica contra esa caché, por lo
que las conversaciones largas siguen siendo ágiles.

| Flag | Por defecto | Descripción |
|---|---|---|
| `--temperature` | `0.8` | agudiza (`<1`) o aplana (`>1`) la distribución |
| `--top-k` | `50` | conserva solo los `k` tokens más probables (`0` lo desactiva) |
| `--top-p` | `0.9` | muestreo nucleus (`1.0` lo desactiva) |
| `--repetition-penalty` | `1.1` | penaliza tokens ya generados |
| `--max-new-tokens` | `128` | límite de generación por turno |

El buffer de conversación mantiene un historial de texto plano creciente y, cuando se
acerca a `context_length`, recorta desde el principio en el límite de frase más cercano
(nunca a mitad de palabra), preservando el `<bos>` inicial. Esa ventana deslizante *es* la
memoria a corto plazo: el modelo lee todo el buffer en cada turno.

## Paradigma de chat y limitaciones

Este proyecto tiene un alcance deliberado y es honesto sobre lo que un modelo de 15 M de
parámetros puede aprender de prosa continua:

- **Continúa texto, no sigue instrucciones.** No hay turnos `<user>`/`<bot>` en los datos
  de entrenamiento, así que el modelo no tiene señal de turnos explícitos. Lo que escribes
  se trata como el inicio de un pasaje; el modelo escribe lo que viene después.
- **La coherencia es local.** Espera ~5–6 frases de estilo consistente y seguimiento de
  entidades dentro de la ventana de contexto. La coherencia de largo alcance no está
  garantizada.
- **Es solo en inglés** y de sabor literario, porque el corpus es ficción en inglés.

Una comprobación razonable de que la ventana de contexto se usa de verdad: menciona el
nombre de un personaje al principio del buffer y confirma que el modelo aún se refiere a él
4–6 frases después. Se deja un esbozo `DialogueDatasetSource` en `data/sources.py` como
punto de extensión si más adelante haces fine-tuning con diálogo anotado.

## Pruebas

```bash
pytest
```

Las pruebas unitarias cubren las partes más propensas a romperse en silencio:

- `tests/test_model_shapes.py` — shapes de tensores en el forward, recuento de parámetros
- `tests/test_attention.py` — shapes de salida de atención y comportamiento de la caché KV
- `tests/test_causal_mask.py` — sin fuga hacia posiciones futuras
- `tests/test_packing.py` — lógica de división por libro y empaquetado binario
- `tests/test_conversation_buffer.py` — el recorte por límite de frase conserva `<bos>`

Las pruebas usan una configuración diminuta y desechable y un tokenizador pequeño entrenado
al vuelo, así que toda la suite termina en unos segundos y no necesita datos descargados.

## Desarrollo

Las tareas comunes están envueltas en el `Makefile`:

```bash
make dev         # instalación editable con extras de desarrollo
make test        # pytest -q
make typecheck   # mypy .
make clean       # elimina cachés y artefactos de compilación
```

El código está totalmente anotado con tipos y verificado con `mypy`. Cuando añadas código,
mantenlo tipado y agrega una prueba si tocas el modelo, el tokenizador o las rutas de
packing. El estilo es Python simple con indentación de 4 espacios; los comentarios explican
el *porqué*, no el *qué*. Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para el detalle
completo.

## Notas de rendimiento

En la CPU de desarrollo (i7-1225U), espera aproximadamente **400–800 tokens/s** durante el
entrenamiento, lo que equivale a unas **7–14 horas por época** sobre un corpus de ~20 M de
tokens. El rendimiento escala con `num_threads`, `use_bf16` y `batch_size × grad_accum_steps`;
ajústalos a tus propios núcleos y memoria. La generación con caché KV es notablemente más
rápida que la línea base sin caché en la misma CPU: esa diferencia es la razón por la que la
caché es obligatoria aquí.

## Hoja de ruta

Orden aproximado, no promesas:

- **Codificación posicional RoPE** para reemplazar las posiciones absolutas aprendidas y
  permitir que el modelo extrapole más allá de `context_length`.
- **Fine-tuning de instrucciones opcional** sobre el modelo base, conectando el esbozo
  `DialogueDatasetSource` y tokens de rol reales `<user>`/`<bot>`.
- **Contexto más largo** (512+) cuando el rendimiento de entrenamiento lo permita.
- **Un comando de CLI `sample`** para continuación de una sola pasada sin el bucle interactivo.
- **Desalojo de la caché KV** para que las sesiones de chat puedan durar indefinidamente en
  lugar de recortar texto.

## Preguntas frecuentes

**¿Por qué solo CPU?** Porque el objetivo era ver hasta dónde llega un modelo hecho desde
cero en un portátil sin GPU. Todo está afinado en torno a esa restricción.

**¿Por qué no usar simplemente `transformers`?** Construir las piezas a mano es todo el
ejercicio. Ves exactamente qué hacen la atención, la caché KV y el bucle de entrenamiento.

**Responde con galimatías / se repite.** Los modelos pequeños entrenados con pocos datos
hacen eso. Dale más corpus y más pasos, baja `--temperature` y sube un poco
`--repetition-penalty`. Recuerda también que *continúa* texto: no responde preguntas
(consulta [Paradigma de chat y limitaciones](#paradigma-de-chat-y-limitaciones)).

**¿Puedo usar mi propio texto en lugar de Gutenberg?** Sí. Cualquier archivo `.txt` simple
funciona; el paso específico de Gutenberg es solo la eliminación de cabecera/pie, que no
hace nada en otros textos. Para otro formato, añade una subclase de `TextCorpusSource` en
`data/sources.py`.

**¿Cómo lo hago más grande/pequeño?** Edita los valores por defecto en `config.py`
(`d_model`, `n_layers`, `n_heads`, `context_length`). El recuento de parámetros se imprime
al inicio del entrenamiento para que compruebes dónde quedaste.

## Contribuir

Las contribuciones son bienvenidas: consulta [CONTRIBUTING.md](CONTRIBUTING.md) para la
configuración y expectativas, y [CHANGELOG.md](CHANGELOG.md) para el historial de versiones.
Para algo grande, abre primero un issue para acordar el enfoque.

## Licencia

[MIT](../../../LICENSE) © 2026 Kronos
