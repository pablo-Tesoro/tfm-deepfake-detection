# Consideraciones éticas, legales y de privacidad

> Trabajar con rostros humanos y con tecnología de doble uso exige una reflexión
> explícita. Este documento alimenta la sección correspondiente de la memoria y
> responde al punto de la guía sobre **derechos de uso de los datos**.

---

## 1. Licencia y derechos de uso del dataset

**FaceForensics++** no es de dominio público ni de licencia abierta:

- El acceso requiere **solicitud mediante formulario** y aceptación de los
  **términos de uso** (*FaceForensics Terms of Use*).
- El uso está restringido a fines **académicos y de investigación**.
- **No está permitida la redistribución** de los vídeos. Por eso el repositorio
  excluye por completo `data/` mediante `.gitignore`: se comparte el código, nunca
  los datos.
- Es obligatorio **citar** el trabajo original:

  > Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., y Nießner, M.
  > (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images*. ICCV.

Este uso encaja con el requisito de la guía del TFM sobre conjuntos sin
restricciones de uso incompatibles: se trata de un *benchmark* académico empleado
en un trabajo académico, sin redistribución.

## 2. Privacidad de las personas grabadas

Los vídeos contienen **rostros de personas reales** (originalmente de YouTube), lo
que constituye un dato personal de categoría biométrica cuando se procesa para
identificar o caracterizar a alguien.

Medidas adoptadas y matices relevantes:

- El sistema **no realiza identificación**: no reconoce *quién* aparece, solo
  estima si el rostro ha sido manipulado. No se construye ninguna base de datos
  biométrica ni se asocian identidades.
- Los datos **no salen del entorno de trabajo** del alumno (Drive personal) ni se
  publican; los *embeddings* derivados tampoco se comparten.
- Los rostros solo se muestran en la memoria o el vídeo en ejemplos puntuales
  procedentes del propio dataset, con finalidad ilustrativa.
- En un despliegue real (caso KYC), el tratamiento estaría sujeto al **RGPD** y
  requeriría base jurídica, información al interesado, minimización y plazos de
  conservación definidos. Queda fuera del alcance de este prototipo académico.

## 3. Uso dual de la tecnología

Un detector de deepfakes es tecnología de **doble uso**: el mismo conocimiento que
permite detectar artefactos puede emplearse para **generarlos mejor**, entrenando
generadores que evadan precisamente esas señales (dinámica adversaria conocida).

Postura adoptada en este trabajo:

- No se desarrolla ni publica ninguna capacidad **generativa**; el proyecto es
  exclusivamente **defensivo**.
- Las explicaciones (Grad-CAM) se presentan como herramienta de **auditoría y
  trazabilidad** para el analista, no como guía para evadir el detector.
- Se asume y documenta que el rendimiento **se degrada frente a métodos no vistos**
  (lo cuantifica el experimento *cross-manipulation*), lo que desaconseja
  presentarlo como solución definitiva.

## 4. Limitaciones y riesgos de un uso irresponsable

Declararlas es parte de un diseño honesto:

- **No es una herramienta certificada.** Es un prototipo académico; la app lo
  indica explícitamente en su pie.
- **Generalización limitada.** Está entrenado sobre cuatro métodos concretos de
  manipulación; ante técnicas nuevas o vídeos "en libertad" su fiabilidad cae.
- **Riesgo de falsos positivos sobre personas reales.** Acusar erróneamente a un
  cliente legítimo tiene consecuencias reputacionales y legales; por eso el diseño
  incorpora una **zona de revisión manual** en lugar de un veredicto binario
  automático.
- **Posibles sesgos demográficos.** El rendimiento puede variar según tono de piel,
  edad, sexo o condiciones de iluminación si el dataset no está equilibrado. No se
  ha auditado este aspecto; se señala como línea futura necesaria antes de
  cualquier uso real.
- **Supervisión humana.** En el caso de negocio, el sistema se plantea como apoyo
  a la decisión de un analista, nunca como decisión automatizada sin intervención
  (relevante también por el artículo 22 del RGPD sobre decisiones automatizadas).

## 5. Resumen para la memoria

Una síntesis de un párrafo, lista para adaptar:

> El proyecto emplea FaceForensics++ bajo licencia académica, sin redistribuir los
> vídeos y citando la fuente original. Aunque el sistema procesa rostros reales, no
> realiza identificación biométrica ni almacena identidades, y los datos permanecen
> en el entorno privado del autor. Se reconoce el carácter de doble uso de la
> tecnología: el trabajo es estrictamente defensivo y no aporta capacidades
> generativas. Finalmente, se asumen limitaciones relevantes —generalización
> restringida a los métodos vistos, posibles sesgos demográficos no auditados y
> coste de los falsos positivos—, razón por la cual la solución se plantea como
> apoyo a un analista humano, con una zona explícita de revisión manual, y no como
> un veredicto automático.
