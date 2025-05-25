# myMind - Evaluación de Modelos de Reconocimiento de Voz

## 📋 Descripción del Proyecto

Este repositorio contiene la evaluación exhaustiva de diferentes modelos de reconocimiento automático de voz (ASR) y análisis de emociones/sentimientos para el proyecto **myMind** de la Pontificia Universidad Javeriana. El objetivo es seleccionar el modelo más adecuado para la transcripción de audio en español con las mejores métricas de rendimiento y eficiencia.

## 🎯 Objetivo

Evaluar y comparar el rendimiento de múltiples modelos de ASR para determinar cuál ofrece la mejor combinación de:
- **Precisión de transcripción** (WER, CER, WA)
- **Velocidad de procesamiento** (WPM)
- **Eficiencia computacional** (tiempo de ejecución)

## 📁 Estructura del Repositorio

```
├── PruebaModelosTranscripción.ipynb    # Evaluación de modelos ASR
├── PruebaModelosEmociones.ipynb        # Evaluación de modelos de emociones/sentimientos
├── Audios/                             # Dataset de archivos de audio (69 archivos .wav)
├── Transcripciones_Manuales/           # Transcripciones de referencia (ground truth)
└── Resultados/                         # Resultados y métricas por modelo
    ├── Whisper_tiny/
    ├── Whisper_small/
    ├── Whisper_medium/
    ├── SpeechBrain/
    ├── Nemo/
    ├── Kaldi_Vosk/
    └── ...
```

## 🤖 Modelos Evaluados

### Modelos ASR (Reconocimiento de Voz)

| Modelo | Variante | Estado | Descripción |
|--------|----------|--------|-------------|
| **Whisper** | tiny, small, medium | ✅ Evaluado | Modelo de OpenAI optimizado para español |
| **WhisperX** | medium, large | ⚠️ Parcial | Versión optimizada con alineación temporal |
| **Wav2Vec 2.0** | base, medium, large | ⚠️ Interrumpido | Modelo de Facebook para ASR |
| **SpeechBrain** | wav2vec2-commonvoice-es | ✅ Evaluado | Framework especializado en español |
| **NeMo** | conformer-transducer-large | ✅ Evaluado | NVIDIA's toolkit para ASR |
| **Kaldi/Vosk** | small, complete | ✅ Evaluado | Implementación tradicional de Kaldi |
| **Qwen2Audio** | 7B-Instruct | ❌ Fallido | LLM multimodal (limitaciones de RAM) |

### Modelos de Emociones/Sentimientos

- Evaluación de modelos pre-entrenados
- Fine-tuning para mejora de métricas
- Análisis de accuracy, precisión, recall y F1-score

## 📊 Métricas de Evaluación

### Para ASR:
- **WER (Word Error Rate)**: Tasa de errores de palabras
- **WA (Word Accuracy)**: Precisión en palabras (1 - WER)
- **CER (Character Error Rate)**: Tasa de errores de caracteres
- **WPM (Words Per Minute)**: Velocidad de transcripción
- **Execution Time**: Tiempo de procesamiento

### Para Emociones:
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precisión y recall

## 🚀 Configuración y Uso

### Requisitos Previos

```bash
# Librerías principales
pip install openai-whisper
pip install jiwer librosa
pip install speechbrain transformers
pip install nemo_toolkit['asr']
pip install vosk
pip install torch torchaudio
```

### Ejecución

1. **Preparar datos**: Colocar archivos de audio en `/Audios/` y transcripciones manuales en `/Transcripciones_Manuales/`

2. **Ejecutar evaluación ASR**:
   ```python
   # Abrir PruebaModelosTranscripción.ipynb
   # Ejecutar celdas secuencialmente por modelo
   ```

3. **Ejecutar evaluación de emociones**:
   ```python
   # Abrir PruebaModelosEmociones.ipynb
   # Seguir el flujo de evaluación y fine-tuning
   ```

## 📈 Resultados Destacados

### Mejores Modelos ASR (Preliminar)

| Modelo | WER Promedio | WPM Promedio | Tiempo Ejecución |
|--------|--------------|--------------|------------------|
| Whisper Small | ~0.15 | ~145 | ~30s |
| NeMo Conformer | ~0.12 | ~140 | ~15s |
| SpeechBrain | ~0.18 | ~95 | ~45s |

> **Nota**: Resultados basados en dataset de 69 archivos de audio en español

### Consideraciones Técnicas

- **Whisper Large**: Descartado por limitaciones de RAM (>12GB requeridos)
- **Qwen2Audio**: No viable por consumo excesivo de memoria
- **Wav2Vec**: Evaluación interrumpida por problemas de compatibilidad

## ⚠️ Limitaciones y Desafíos

1. **Recursos Computacionales**: 
   - Modelos grandes requieren >12GB RAM
   - Tiempo de procesamiento extenso para algunos modelos

2. **Compatibilidad**:
   - Algunos modelos requieren versiones específicas de dependencias
   - Problemas de compatibilidad entre frameworks

3. **Dataset**:
   - 69 archivos de audio en español
   - Variabilidad en calidad y duración de grabaciones

## 👥 Equipo

**Juan José Gómez Arenas**  
*Pontificia Universidad Javeriana*  
*Proyecto myMind*

## 📝 Notas de Implementación

- Entorno de desarrollo: Google Colab
- Almacenamiento: Google Drive
- Lenguaje objetivo: Español (ES)
- Framework de evaluación: Python + Jupyter Notebooks

## 🔮 Próximos Pasos

1. Completar evaluación de modelos interrumpidos
2. Optimizar modelos seleccionados mediante fine-tuning
3. Implementar pipeline de producción
4. Integrar modelo seleccionado en aplicación myMind

## 📄 Licencia

Este proyecto es parte del desarrollo académico en la Pontificia Universidad Javeriana para el proyecto myMind.

---

*Para más información sobre la metodología de evaluación y resultados detallados, consultar los notebooks incluidos en el repositorio.*
