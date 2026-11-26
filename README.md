# 🚰 JUNTA ANALYTICS

**Sistema avanzado de análisis y detección de anomalías para medidores de agua**

Una aplicación completa que combina análisis de datos, visualización interactiva y detección inteligente de anomalías para optimizar la gestión de recursos hídricos en juntas de agua comunitarias.

---

## 🎯 **Características Principales**

### 📊 **Dashboard Interactivo**
- **Visualizaciones en tiempo real** de consumo y patrones climáticos
- **KPIs dinámicos** con métricas clave del sistema
- **Gráficos correlacionales** entre lluvia y consumo
- **Interfaz responsive** optimizada para diferentes dispositivos

### 🔍 **Detector de Anomalías V3 (Híbrido)**
- **83.3% de precisión** en detección de anomalías reales
- **Análisis específico por medidor** usando historial individual
- **Enfoque de 3 niveles**:
  - ✅ **Reglas básicas**: Lecturas negativas, consumo cero, extremos
  - 📊 **Análisis estadístico**: Z-scores y ratios específicos del medidor
  - 🤖 **Modelos ML**: Isolation Forest + One-Class SVM
- **Explicaciones claras** del por qué una lectura es anómala

### 📈 **Análisis de Datos Integrado**
- **Integración automática** de datos de consumo y clima
- **Procesamiento de 10,000+ lecturas** de 362 medidores
- **Correlaciones lluvia-consumo** para análisis predictivo
- **Estadísticas por vecindario** y medidor individual

### 🛠️ **API RESTful Completa**
- **`/api/detect-anomaly`**: Detección de anomalías en tiempo real
- **`/api/dashboard-data`**: Datos completos para dashboard
- **`/api/water-meters`**: Gestión de medidores
- **`/api/neighborhoods`**: Estadísticas por vecindario
- **`/health`**: Monitoreo del sistema

---

## 🏗️ **Arquitectura del Sistema**

### **Backend (Flask)**
```
app.py                 # Aplicación principal Flask
├── config.py          # Configuración centralizada
├── database/          # Gestión de base de datos PostgreSQL
├── ml_models/         # Modelos de Machine Learning
│   ├── anomaly_detector_v3.py  # Detector híbrido avanzado
│   └── model_manager.py        # Gestión de modelos
├── utils/             # Utilidades de procesamiento
│   ├── data_processing.py      # Limpieza y transformación
│   └── visualization.py       # Generación de gráficos
└── static/js/         # Frontend interactivo
```

### **Notebooks de Análisis**
```
notebooks/
├── 01_data_integration.ipynb    # Integración de datos
├── 02_rainfall_consumption.ipynb # Análisis correlacional
└── 04_test_new_anomaly_detector.ipynb # Pruebas del detector V3
```

---

## 🚀 **Instalación y Configuración**

### **Prerrequisitos**
- Python 3.8+
- PostgreSQL 12+
- Node.js (opcional, para desarrollo frontend)

### **Instalación**
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/junta-analytics.git
cd junta-analytics

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de base de datos
```

### **Configuración de Base de Datos**
```bash
# Crear base de datos PostgreSQL
createdb junta_jeru_backend

# Configurar en .env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=junta_jeru_backend
DB_USER=tu_usuario
DB_PASSWORD=tu_password
```

### **Ejecutar Aplicación**
```bash
# Modo desarrollo
python app.py

# La aplicación estará disponible en http://localhost:8000
```

---

## 📊 **Uso del Sistema**

### **1. Dashboard Principal**
Accede a `http://localhost:8000` para ver:
- **KPIs en tiempo real** del sistema
- **Gráficos interactivos** de consumo y clima
- **Estadísticas por vecindario**
- **Tendencias y correlaciones**

### **2. Detección de Anomalías**
```bash
# API para detectar anomalías
curl -X POST http://localhost:8000/api/detect-anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "water_meter_id": 479,
    "current_reading": 1050,
    "previous_reading": 1000,
    "days_billed": 30
  }'
```

**Respuesta esperada:**
```json
{
  "is_anomaly": false,
  "score": 0.0,
  "confidence": 0.0,
  "reason": "Consumo normal dentro de parámetros esperados",
  "total_consumed": 50,
  "consumption_per_day": 1.67,
  "detection_method": "none"
}
```

### **3. Análisis de Datos**
Ejecuta los notebooks en orden:
1. **`01_data_integration.ipynb`**: Integra datos de consumo y clima
2. **`02_rainfall_consumption.ipynb`**: Analiza correlaciones
3. **`04_test_new_anomaly_detector.ipynb`**: Prueba el detector de anomalías

---

## 🧠 **Detector de Anomalías V3**

### **¿Cómo Funciona?**

El detector utiliza un **enfoque híbrido de 3 niveles** para máxima precisión:

#### **Nivel 1: Reglas Básicas** 🚨
```python
# Detección inmediata de casos obvios
if lectura_actual < lectura_anterior:
    return "Lectura negativa: error en medición"

if consumo == 0:
    return "Consumo cero: medidor roto o casa vacía"

if consumo_por_día > 100:
    return "Consumo extremo: posible fuga mayor"
```

#### **Nivel 2: Análisis Estadístico** 📊
```python
# Específico para cada medidor
z_score = (consumo_actual - promedio_medidor) / desviación_medidor
ratio = consumo_actual / consumo_típico_medidor

if z_score > 2.5 or ratio > 5:
    return f"Anomalía estadística: {z_score:.1f}σ del patrón del medidor"
```

#### **Nivel 3: Modelos ML** 🤖
```python
# Para patrones complejos
isolation_forest_score = modelo_if.predict(características)
svm_score = modelo_svm.predict(características)
score_combinado = (if_score + svm_score) / 2
```

### **Resultados de Rendimiento**
- ✅ **83.3% de precisión** en casos de prueba
- ✅ **100% de detección** en casos críticos (lecturas negativas, consumo cero)
- ✅ **Explicaciones claras** con razones específicas
- ✅ **Personalización** para cada uno de los 362 medidores

---

## 📈 **Casos de Uso Reales**

### **Detección de Fugas**
```
Medidor 479: Consumo típico 4 unidades/mes
Lectura actual: 20 unidades/mes
→ Detectado: "Anomalía estadística: 6.5σ del patrón del medidor (ratio: 4.8x)"
```

### **Medidores Averiados**
```
Lectura anterior: 1000
Lectura actual: 1000 (sin cambio)
→ Detectado: "Consumo cero: posible medidor roto o casa vacía"
```

### **Errores de Lectura**
```
Lectura anterior: 1000
Lectura actual: 980 (menor)
→ Detectado: "Lectura negativa: lectura actual menor que anterior"
```

---

## 🛡️ **Monitoreo y Salud del Sistema**

### **Health Check**
```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "database": "connected",
  "data": {
    "merged_data": true,
    "consumption_data": true
  },
  "models": {
    "anomaly_detector": true
  }
}
```

### **Información de Modelos**
```bash
curl http://localhost:8000/api/model-info
```

---

## 🔧 **Tecnologías Utilizadas**

### **Backend**
- **Flask 3.0** - Framework web ligero y flexible
- **PostgreSQL** - Base de datos robusta para datos de consumo
- **pandas** - Procesamiento y análisis de datos
- **scikit-learn** - Modelos de Machine Learning
- **plotly** - Visualizaciones interactivas

### **Machine Learning**
- **Isolation Forest** - Detección de anomalías no supervisada
- **One-Class SVM** - Clasificación de normalidad
- **Análisis estadístico** - Z-scores y detección de outliers
- **MLflow** - Tracking y gestión de experimentos

### **Frontend**
- **HTML5/CSS3** - Interfaz moderna y responsive
- **JavaScript ES6** - Interactividad del dashboard
- **Chart.js** - Gráficos dinámicos
- **Bootstrap** - Framework CSS responsive

---

## 📝 **Estructura de Datos**

### **Tabla: Consumo Individual**
```sql
water_meter_id    INT     -- ID único del medidor
total_consumed    INT     -- Consumo total en unidades
days_billed      INT     -- Días del período facturado
period_start     DATE    -- Inicio del período
period_end       DATE    -- Fin del período
neighborhood_id  INT     -- ID del vecindario
```

### **Tabla: Datos Climáticos**
```sql
period_dt        DATE    -- Fecha del período
avg_rainfall     FLOAT   -- Precipitación promedio
max_rainfall     FLOAT   -- Precipitación máxima
total_rainfall   FLOAT   -- Precipitación total
```

---

## 🔄 **Flujo de Trabajo**

### **1. Ingesta de Datos**
```
Datos de medidores → Limpieza → Base de datos PostgreSQL
Datos climáticos → Procesamiento → Integración temporal
```

### **2. Entrenamiento de Modelos**
```
Datos históricos → Estadísticas por medidor → Entrenamiento ML → Modelo guardado
```

### **3. Detección en Tiempo Real**
```
Nueva lectura → API /detect-anomaly → Análisis híbrido → Resultado + explicación
```

### **4. Visualización**
```
Datos procesados → Dashboard → Gráficos interactivos → KPIs en tiempo real
```

---

## 🤝 **Contribución**

### **Para Desarrolladores**
1. Fork del repositorio
2. Crear branch de feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit de cambios: `git commit -m 'Añadir nueva funcionalidad'`
4. Push al branch: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

### **Estándares de Código**
```bash
# Formateo de código
black .

# Linting
flake8 .

# Tests
pytest
```

---

## Pull requests

Un PR se fusiona solo con la aprobación de la revisión de Claude
(`.github/workflows/claude-code-review.yml`, que revisa contra este README y `REVIEW.md`). Antes de
fusionar:

1. Espere a que terminen todos los checks, incluido `claude-review`.
2. Lea el último comentario de la revisión. Aprueba cuando el comentario existe y no deja hallazgos
   bloqueantes ni observaciones sin respuesta. Un check en verde no basta: sin el secreto, el job pasa
   sin revisar.
3. Atienda cada hallazgo u observación: corríjalo o responda en el PR por qué no aplica.
4. Después de una corrección, pida otra revisión con un comentario `/review` en el PR (un push no la
   dispara) y vuelva al paso 1.
5. Si la revisión no corre o falla dos veces, un mantenedor decide y deja la decisión escrita en el PR.

Esto vale para todos los PR, también los que solo cambian documentación.

---

## 📄 **Licencia**

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📞 **Contacto y Soporte**

- **Autor**: Luis Pillaga
- **Email**: [lpillaga@gmail.com]
- **GitHub**: [https://github.com/tu-usuario]

### **Reportar Issues**
Si encuentras bugs o tienes sugerencias, por favor abre un issue en GitHub con:
- Descripción detallada del problema
- Pasos para reproducir
- Logs de error (si aplica)
- Versión del sistema

---

## 🎯 **Roadmap**

### **Próximas Funcionalidades**
- [ ] **Alertas automáticas** por email/SMS para anomalías críticas
- [ ] **API móvil** para lecturas en campo
- [ ] **Predicción de consumo** basada en tendencias históricas
- [ ] **Reportes automatizados** mensuales/anuales
- [ ] **Integración IoT** para lecturas automáticas
- [ ] **Dashboard multi-inquilino** para múltiples juntas de agua

### **Mejoras Técnicas**
- [ ] **Containerización** con Docker
- [ ] **CI/CD pipeline** con GitHub Actions
- [ ] **Caching** con Redis para mejor rendimiento
- [ ] **Monitoreo** con Prometheus/Grafana
- [ ] **Tests automatizados** con mayor cobertura

---

*Desarrollado con ❤️ para optimizar la gestión de recursos hídricos en comunidades.*