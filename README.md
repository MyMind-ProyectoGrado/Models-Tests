# Evaluación de Modelos de Transcripción de Voz

## 📋 Descripción del Proyecto

Este repositorio contiene la evaluación comparativa de diferentes modelos de **Automatic Speech Recognition (ASR)** para el proyecto **myMind** de la **Pontificia Universidad Javeriana**. El objetivo es identificar el modelo más eficiente y preciso para la transcripción de audio en español.

**Autor:** Juan José Gómez Arenas  
**Institución:** Pontificia Universidad Javeriana  
**Proyecto:** myMind

## 🎯 Objetivo

Evaluar y comparar el rendimiento de múltiples modelos de transcripción automática de voz para seleccionar la mejor opción para la aplicación myMind, considerando factores como:

- **Precisión de transcripción** (WER, WA, CER)
- **Velocidad de procesamiento** (WPM, tiempo de ejecución)
- **Recursos computacionales requeridos**
- **Compatibilidad con español**

## 🔧 Modelos Evaluados

### ✅ Modelos Implementados
1. **Whisper** (OpenAI)
   - `tiny`, `small`, `medium`
   - Modelo `large` omitido por limitaciones de recursos
2. **SpeechBrain**
   - Modelo preentrenado para español
3. **Kaldi/Vosk**
   - Versión pequeña y completa
4. **NeMo** (NVIDIA)
   - Modelo conformer transducer para español
5. **Wav2Vec 2.0** (Facebook)
   - Modelos base, medium y large

### ❌ Modelos No Completados
- **WhisperX:** Pendiente de implementación
- **Qwen2Audio:** Modelo demasiado pesado para el entorno
- **Wav2Vec large:** Interrumpido por limitaciones computacionales

## 📊 Métricas de Evaluación

El sistema evalúa cada modelo usando las siguientes métricas:

- **WER (Word Error Rate):** Tasa de errores de palabras
- **WA (Word Accuracy):** Precisión de palabras (1 - WER)
- **CER (Character Error Rate):** Tasa de errores de caracteres
- **WPM (Words Per Minute):** Velocidad de transcripción
- **Tiempo de Ejecución:** Tiempo total de procesamiento

## 🚀 Configuración del Entorno

### Prerrequisitos
```bash
# Librerías principales
pip install jiwer librosa torch transformers
pip install openai-whisper speechbrain vosk
pip install nemo_toolkit['asr'] torchaudio pydub
```

### Estructura de Directorios
```
📁 Transcripcion/
├── 📁 Audios/                    # Archivos de audio (.wav)
├── 📁 Transcripciones_Manuales/  # Ground truth
└── 📁 Resultados/                # Resultados por modelo
    ├── 📁 Whisper_tiny/
    ├── 📁 Whisper_small/
    ├── 📁 SpeechBrain/
    ├── 📁 Kaldi_Vosk/
    └── 📁 Nemo/
```

## 💻 Uso

### Ejecución de Evaluaciones
```python
# Ejemplo para Whisper
python evaluate_whisper.py --model_size tiny --audio_dir /path/to/audios

# Ejemplo para SpeechBrain
python evaluate_speechbrain.py --audio_dir /path/to/audios

# Ejemplo para NeMo
python evaluate_nemo.py --model stt_es_conformer_transducer_large
```

### Cálculo de Métricas
```python
from jiwer import wer, cer

def calculate_metrics(reference, hypothesis, execution_time):
    word_error_rate = wer(reference, hypothesis)
    return {
        "WER": word_error_rate,
        "WA": 1 - word_error_rate,
        "CER": cer(reference, hypothesis),
        "WPM": words_per_minute(hypothesis),
        "Execution Time (s)": execution_time
    }
```

## 📈 Resultados Preliminares

### Observaciones Generales
- **Whisper** muestra consistencia entre sus versiones tiny y small
- **NeMo** ofrece buen rendimiento en español
- **SpeechBrain** presenta resultados competitivos
- **Kaldi/Vosk** varía significativamente según la versión del modelo

### Limitaciones Encontradas
- Modelos grandes requieren recursos computacionales significativos
- Algunos modelos presentan problemas de compatibilidad
- El entorno de Google Colab impone restricciones de memoria

## 🛠️ Funciones Principales

### Cálculo de WPM
```python
def words_per_minute(text, audio_path):
    word_count = len(text.split())
    duration = librosa.get_duration(path=audio_path)
    return word_count / (duration / 60)
```

### Transcripción con Manejo de Errores
```python
def transcribe_with_metrics(model, audio_path):
    start_time = time.time()
    try:
        transcript = model.transcribe(audio_path)
        execution_time = time.time() - start_time
        return transcript, execution_time
    except Exception as e:
        print(f"Error: {e}")
        return None, None
```

## 📋 Archivos del Proyecto

- `PruebaModelosTranscripción.ipynb` - Notebook principal con todas las evaluaciones
- `README.md` - Este archivo
- `/Audios/` - Dataset de archivos de audio para evaluación
- `/Resultados/` - Transcripciones y métricas generadas

## 🔬 Metodología

1. **Preparación de datos:** Audios en formato WAV
2. **Transcripción manual:** Ground truth para comparación
3. **Evaluación sistemática:** Cada modelo procesa el mismo conjunto de audios
4. **Cálculo de métricas:** Comparación automática con referencias
5. **Análisis de resultados:** Identificación del mejor modelo

## 🚧 Trabajo Futuro

- [ ] Completar evaluación de WhisperX
- [ ] Implementar evaluación con modelos más ligeros de Qwen2Audio
- [ ] Optimizar uso de memoria para modelos grandes
- [ ] Análisis estadístico detallado de resultados
- [ ] Implementación del modelo seleccionado en myMind

## 📞 Contacto

**Juan José Gómez Arenas**  
Pontificia Universidad Javeriana  
Proyecto myMind

---

**Nota:** Este proyecto forma parte del desarrollo de la aplicación myMind para mejorar las capacidades de transcripción automática de voz en español.
