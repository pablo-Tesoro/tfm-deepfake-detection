# Guía de la memoria (Fase 5)

> Mapa entre **lo que exige la guía del TFM (UCM)**, las secciones de la memoria y
> el material ya generado por el pipeline. Sirve para redactar sin dejarse nada y
> para repartir bien las 20 caras.

---

## 1. Requisitos de la guía y dónde se cubren

| Requisito de la guía | Dónde se cubre | Evidencia disponible |
|---|---|---|
| Dataset público, no trillado, sin derechos restrictivos | §2 Datos | `docs/03_etica_privacidad.md` |
| Análisis descriptivo (gráfico en lo posible) | §3 EDA | `01_distribucion_clases.png`, notebook 01 |
| Transformaciones adecuadas | §4 Preprocesamiento | notebook 01, `docs/02_arquitectura.md` |
| Varias técnicas de modelización con bondades y debilidades | §5 Modelización | `tabla_comparativa.csv`, `comparativa_backbones.csv` |
| **Aportar más que un AutoML** | §5 Modelización | comparativa contra el AutoML de referencia |
| Discusión: explicabilidad e interpretabilidad | §7 XAI | figuras Grad-CAM y curva temporal |
| Informe entendible por un perfil de **Negocio** | Todo el documento | `matriz_confusion.png`, umbral por coste |
| **Productivización** (muy valorada) | §8 | capturas de VERIFAKE |
| Conclusiones y mejoras futuras | §9 | — |
| Bibliografía breve (≈ media cara) | §10 | — |
| Anexos con código (no cuentan para las 20 caras) | Anexos | repositorio |

## 2. Estructura propuesta y reparto de páginas

Total: **20 caras** (sin portada, índice ni anexos).

| § | Sección | Caras | Contenido esencial |
|---|---|---|---|
| 1 | Introducción y contexto de negocio | 2 | El problema del fraude de identidad; caso ancla (videoidentificación bancaria, SEPBLAC); objetivos |
| 2 | Descripción del conjunto de datos | 1,5 | FaceForensics++, volumen, 4 métodos, licencia y ética |
| 3 | Análisis exploratorio (EDA) | 2 | Distribución de clases y desbalance; propiedades de los vídeos; tasa de detección facial |
| 4 | Preprocesamiento y transformación | 2 | Muestreo de fotogramas, recorte facial, pipeline fusionado y por qué |
| 5 | Modelización analítica avanzada | 4 | Transfer learning; arquitectura CNN+LSTM; baseline y AutoML; comparativa de backbones |
| 6 | Evaluación y resultados | 3 | Métricas; cross-manipulation; curva de aprendizaje; por método |
| 7 | Explicabilidad (enfoque de negocio) | 2 | Grad-CAM; curva temporal; traducción a decisiones operativas |
| 8 | Productivización | 1,5 | VERIFAKE: flujo, decisión KYC, capturas |
| 9 | Conclusiones y líneas futuras | 1,5 | Qué se demostró; limitaciones; mejoras |
| 10 | Bibliografía | 0,5 | Referencias clave |

> Formato recomendado por la guía: Verdana o Arial, tamaño 10-11.

## 3. Figuras disponibles y su ubicación

Todas en `reports/figures/`, generadas por `run_all.py`:

| Figura / tabla | Sección | Mensaje que transmite |
|---|---|---|
| `01_distribucion_clases.png` | §3 | Volumen y desbalance Real/Fake |
| `tabla_comparativa.csv` | §5 | El híbrido supera al baseline y al AutoML |
| `matriz_confusion.png` | §6 y §7 | Lectura en clave de coste; efecto del umbral |
| `curva_aprendizaje.png` | §6 | El rendimiento crece con el volumen de datos |
| `comparativa_backbones.csv` | §5 | EfficientNet vs ResNet: precisión frente a coste |
| `metricas_por_metodo.csv`, `auc_por_metodo.png` | §6 | Qué manipulaciones cuestan más detectar |
| Grad-CAM y curva temporal (notebook 03) | §7 | Por qué y en qué momento sospecha el modelo |
| Capturas de VERIFAKE | §8 | La solución en uso |

## 4. Equilibrio técnico / negocio

La guía pide que el documento lo entienda un perfil no técnico. Criterio práctico:

- **En la memoria:** el *qué* y el *para qué*. Analogías, figuras, tablas resumen,
  consecuencias de negocio. Cada resultado numérico acompañado de una frase que
  explique qué implica.
- **En los anexos:** el *cómo* detallado. Código, arquitecturas, hiperparámetros,
  salidas completas de los notebooks.

Regla útil: si un párrafo no se entiende sin saber qué es un tensor, va al anexo.

## 5. Vídeo de presentación

- **5 minutos máximo**, MP4, menos de 50 MB, con **voz en off** (no hace falta
  aparecer en imagen). Debe **describir el proyecto**, no ser un *elevator pitch*.
- Guion sugerido: problema y caso de negocio (1 min) → enfoque técnico (1,5 min) →
  demostración de VERIFAKE con un vídeo auténtico y uno manipulado (1,5 min) →
  resultados y lecciones aprendidas (1 min).
- Para la demo, usa vídeos del **conjunto de test** (nunca de entrenamiento) y, si
  es posible, alguno ajeno al dataset.

## 6. Checklist final de entrega

- [ ] ¿La memoria ocupa como máximo 20 caras (sin portada, índice ni anexos)?
- [ ] ¿Están revisados los derechos de uso de los datos?
- [ ] ¿El código está accesible por enlace, con permisos para Carlos Ortega y
      Santiago Mota?
- [ ] ¿El proyecto es reproducible (`requirements-lock.txt`, semillas)?
- [ ] ¿Hay sección de conclusiones y de mejoras futuras?
- [ ] ¿El vídeo dura menos de 5 minutos y describe el proyecto?
- [ ] ¿Hay bibliografía breve (≈ media cara)?
- [ ] ¿El fichero se nombra `Nombre_Apellido1_Apellido2_...`?
