# Alcance y Caso de Negocio — TFM Detección de Deepfakes

> Documento de Fase 0. Define **qué** se va a hacer, **para quién** y **por qué
> importa**. Es la brújula del proyecto: toda decisión técnica posterior debe
> poder justificarse contra este documento. Revísalo y personaliza lo marcado
> con `[...]`.

---

## 1. Problema y contexto

La generación de vídeos manipulados (*deepfakes*) se ha vuelto accesible y muy
realista. Esto crea un riesgo directo para cualquier proceso que confíe en el
vídeo como prueba de identidad o de un hecho. El problema no es solo "detectar
vídeos falsos en abstracto", sino hacerlo en un punto del proceso donde una
decisión equivocada tiene un coste real.

## 2. Caso de negocio ancla

**Sector:** banca / fintech.
**Proceso:** *onboarding* digital mediante **videoidentificación** (alta de
clientes en remoto). En España, la identificación no presencial por vídeo está
regulada por el SEPBLAC en el marco de la prevención del blanqueo de capitales.

**¿Qué decide el modelo?** Ante el vídeo de verificación de un nuevo cliente,
estimar si el rostro ha sido manipulado sintéticamente (suplantación de identidad).

**¿Por qué este caso y no "desinformación en redes" genérico?**

- Tiene un **dueño claro** del problema (el equipo de riesgo/fraude del banco).
- Tiene una **asimetría de coste** muy marcada y fácil de explicar a Negocio:
  - **Falso negativo** (dejar pasar un deepfake): se da de alta a un impostor →
    fraude de identidad, pérdidas, sanción regulatoria. **Coste muy alto.**
  - **Falso positivo** (marcar como falso a un cliente real): fricción, el cliente
    pasa a verificación manual. **Coste moderado** (molestia + coste operativo).
- Esa asimetría justifica que **no optimicemos el F1 a ciegas**, sino que ajustemos
  el umbral de decisión para **priorizar el *recall* de la clase "fake"**. Este es
  exactamente el tipo de razonamiento "de negocio" que pide la guía.

> Alternativas consideradas (mencionar brevemente en la memoria, sección de
> contexto): moderación de contenido en redes sociales, verificación periodística
> de vídeos, peritaje en seguros. Se elige banca por la claridad del coste y la
> relevancia regulatoria. `[Confirma si prefieres otro caso ancla.]`

## 3. Objetivos

**Objetivo general.** Desarrollar un sistema de *Deep Learning* que clasifique
secuencias de vídeo como reales o manipuladas y que **explique visualmente** su
decisión, presentado mediante una herramienta web utilizable por un perfil no
técnico.

**Objetivos específicos.**

1. Construir un *pipeline* reproducible de tratamiento de vídeo masivo (extracción
   facial y muestreo de frames) viable con recursos limitados.
2. Comparar varias técnicas de modelización (baseline a nivel de frame, híbrido
   CNN+LSTM y un AutoML de referencia), justificando bondades y debilidades.
3. Evaluar la **generalización** del modelo a un método de manipulación no visto
   durante el entrenamiento (experimento *cross-manipulation*).
4. Traducir los resultados a métricas de negocio (coste de FP/FN, umbral operativo)
   e interpretarlos con mapas de calor (Grad-CAM).
5. Productivizar la solución en una app interactiva (Gradio).

## 4. Alcance (qué entra y qué NO)

**Dentro del alcance:**

- Detección a nivel de clip (vídeo) sobre FaceForensics++ (c23).
- Manipulaciones de rostro de los 4 métodos del dataset.
- Una arquitectura híbrida + baselines + comparativa.
- Explicabilidad visual y app de demostración.

**Fuera del alcance** (declararlo evita falsas expectativas y es buena práctica):

- Detección en tiempo real a escala de producción / despliegue en la nube real.
- Audio *deepfakes* o manipulación de voz (solo vídeo/imagen facial).
- Generalización a *deepfakes* "en libertad" de cualquier origen (se acota a los
  métodos del dataset; la generalización se estudia, no se garantiza).
- Reentrenamiento continuo / MLOps avanzado.

## 5. Datos

- **Fuente:** FaceForensics++ — https://github.com/ondyari/FaceForensics
- **Acceso:** formulario de Google → script de descarga. **Solicitar cuanto antes.**
- **Volumen:** 1000 vídeos reales + 4000 manipulados (suficientemente grande para
  cumplir la preferencia de la guía por conjuntos masivos).
- **Licencia y ética:** uso académico; no redistribuir; citar a Rössler et al.
  (2019). Considerar en la memoria la **privacidad** de las personas grabadas y el
  **uso dual** de la tecnología (detección vs. evasión).

> Nota sobre "datos sintéticos": la guía desaconseja los datos sintéticos porque
> sus conclusiones suelen ser limitadas. **Aclarar en la memoria** que aquí el
> contenido manipulado es sintético, pero el dataset es un *benchmark* real y
> establecido de un fenómeno real; no es el caso que la guía quiere evitar.

## 6. Métricas de éxito

**Técnicas:** Accuracy, Precision, Recall, F1-Score y AUC-ROC, reportadas por
método de manipulación y en el escenario cross-manipulation.

**De negocio:** matriz de confusión leída en clave de coste, y elección de un
**umbral de operación** que minimice el coste esperado dado el caso ancla
(penalizando más los falsos negativos).

**Criterio de "aporta más que un AutoML":** el híbrido debe **superar de forma
medible** al AutoML de referencia y/o al baseline a nivel de frame; si no lo hace,
se analiza y discute por qué (también es un resultado válido y honesto).

## 7. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Coste computacional del vídeo | Pipeline en 2 etapas + embeddings cacheados + subconjunto + c23 |
| Latencia de acceso al dataset | Solicitar el formulario **hoy**; avanzar el scaffold mientras llega |
| Sobreajuste al método de generación | Experimento cross-manipulation explícito |
| Desbalance de clases | Análisis en el EDA y técnicas de balanceo / pesos en la pérdida |
| Memoria demasiado técnica para "Negocio" | Detalle técnico a anexos; memoria visual y orientada a decisión |

## 8. Entregables

1. Memoria de máx. 20 caras (orientación de negocio).
2. Vídeo MP4 de máx. 5 min (< 50 MB), voz en off descriptiva.
3. Anexos: este repositorio de código + estudios detallados de EDA y modelos.

---

### Checklist de cierre de Fase 0

- [ ] Solicitud de acceso a FaceForensics++ enviada (formulario de Google).
- [ ] Repositorio creado (GitHub/Drive) con permisos para Carlos Ortega y Santiago Mota.
- [ ] Caso de negocio ancla confirmado y personalizado.
- [ ] Entorno reproducible verificado (`set_seed` + `load_config` funcionan).
- [ ] Decisión de framework confirmada (por defecto: PyTorch).
