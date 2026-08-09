# Contribuir

**Idiomas:** [English](../../../CONTRIBUTING.md) · **Español** · [Deutsch](../de/CONTRIBUTING.md) · [中文](../zh/CONTRIBUTING.md) · [Русский](../ru/CONTRIBUTING.md)

Gracias por echar un vistazo. Este es un proyecto pequeño de un solo autor, así que no hay
mucho proceso, pero unas cuantas notas hacen que todo vaya más fino.

## Puesta en marcha

```bash
git clone <repo-url> kronos-synapse-dialogue
cd kronos-synapse-dialogue
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Eso te da el comando `kronos_synapse` más `pytest` y `mypy`. Hay un `Makefile` con atajos
(`make dev`, `make test`, `make typecheck`) si te gustan.

## Antes de abrir un PR

- Ejecuta las pruebas: `pytest` (o `make test`). Son rápidas y solo de CPU.
- Ejecuta el verificador de tipos: `mypy .`. El código está totalmente tipado; por favor,
  mantenlo así.
- Mantén el estilo consistente con lo que ya hay: indentación de 4 espacios, anotaciones de
  tipo en las funciones públicas, comentarios breves solo donde el *porqué* no sea obvio. No
  añadas docstrings que solo repitan la firma.
- Si tocas el modelo o la lógica de packing, añade o actualiza una prueba en `tests/`.

## Qué es bienvenido

- Correcciones de errores y mensajes de error más claros.
- Trabajo de rendimiento en la ruta de entrenamiento en CPU.
- Funcionalidades pequeñas y bien acotadas (un nuevo muestreador, una nueva fuente de
  corpus, etc.). El esbozo `DialogueDatasetSource` en `data/sources.py` es un punto de
  extensión obvio.

Para cualquier cosa grande, abre primero un issue para que podamos hablar del enfoque antes
de que inviertas tiempo en ello.

## Mensajes de commit

Línea de asunto corta en imperativo ("Add nucleus sampling clamp", no "Added..."), con un
cuerpo si el cambio necesita contexto. Un cambio lógico por commit cuando sea práctico.
