# 🧾 **Software Requirements Specification (SRS)**

### **Project Title:** Real-Time Aircraft Engine Monitoring and Predictive Maintenance System

---

## **1. Introduction**

### **1.1 Purpose**

This SRS defines the complete software, functional, and non-functional requirements for the **Real-Time Aircraft Engine Monitoring and Predictive Maintenance System (REMPMS)**.
The system aims to:

* Monitor aircraft engine performance using real-time sensor telemetry.
* Predict the **Remaining Useful Life (RUL)** of engines using machine learning.
* Provide early warning alerts for anomalies or potential failures.
* Demonstrate modern **DevOps practices** such as containerization, CI/CD, automation, and monitoring.

The document serves as a technical and functional guide for developers, DevOps engineers, testers, and academic evaluators involved in the design, development, and deployment of this system.

---

### **1.2 Scope**

The REMPMS platform integrates:

* **Real-time data streaming** via Apache Kafka (KRaft mode, no Zookeeper dependency).
* **Machine learning models** for predictive maintenance and RUL estimation.
* **Microservices architecture** using Docker containers for modularity and scalability.
* **Jenkins CI/CD pipelines** for continuous integration, testing, and deployment.
* **Interactive visualization** using a Streamlit-based dashboard.

**Key outcomes:**

* Real-time data ingestion and analytics.
* Predictive insights to optimize maintenance schedules.
* A clean, industrial-grade UI for intuitive decision-making.
* Fully automated DevOps workflows.

---

### **1.3 Definitions, Acronyms, and Abbreviations**

| Term  | Definition                                     |
| ----- | ---------------------------------------------- |
| RUL   | Remaining Useful Life                          |
| ML    | Machine Learning                               |
| CI/CD | Continuous Integration / Continuous Deployment |
| KRaft | Kafka Raft Metadata mode                       |
| API   | Application Programming Interface              |
| UI    | User Interface                                 |
| Redis | In-memory data store used for caching          |

---

### **1.4 References**

* NASA C-MAPSS dataset documentation.
* Apache Kafka 3.7 (KRaft mode) official documentation.
* Jenkins CI/CD Pipeline User Guide.
* Streamlit Framework Documentation.
* Docker and Docker Compose official guides.
* Redis and PostgreSQL documentation.

---

## **2. Overall Description**

### **2.1 Product Perspective**

The system is composed of modular services communicating via message streams. It follows a **containerized microservices architecture** to ensure scalability, fault isolation, and portability.

**High-Level Architecture:**

```
Data Simulator → Kafka Broker (KRaft) → Consumer Service (ML Processing) → PostgreSQL
                                                         ↓
                                                Redis Cache
                                                         ↓
                                                Streamlit Dashboard
```

All components are managed through **Docker Compose** and integrated into a **Jenkins CI/CD pipeline** for build, test, and deploy automation.

---

### **2.2 Product Features**

* **Real-time Engine Telemetry Ingestion:** Continuous sensor data stream from simulated engine sensors.
* **Predictive Analytics:** Machine learning model predicts Remaining Useful Life (RUL).
* **Anomaly Detection:** Identify irregular engine behavior using statistical or ML-based models.
* **Dashboard Visualization:** Real-time performance charts, anomaly alerts, and engine health reports.
* **Automated Deployment:** Jenkins pipeline automates container builds, testing, and deployment.
* **High Availability:** KRaft-mode Kafka ensures lightweight, fault-tolerant message handling.

---

### **2.3 User Classes and Characteristics**

| User Type                          | Description                                                  | Privileges                                                         |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| **Maintenance Engineer**           | Monitors engine health and responds to alerts.               | Read-only access to dashboard, export reports.                     |
| **System Admin / DevOps Engineer** | Manages CI/CD pipeline, container orchestration, monitoring. | Full access to Jenkins, Kafka, and container systems.              |
| **Data Scientist**                 | Trains and updates predictive models.                        | Access to data pipelines, ML service logs, and retraining scripts. |

---

### **2.4 Operating Environment**

| Component            | Technology                         |
| -------------------- | ---------------------------------- |
| OS                   | Linux / Windows (Docker supported) |
| Programming Language | Python 3.11+                       |
| Data Streaming       | Apache Kafka (KRaft mode)          |
| Database             | PostgreSQL 15                      |
| Cache                | Redis 7                            |
| Visualization        | Streamlit                          |
| ML Libraries         | Scikit-learn, NumPy, Pandas        |
| Containerization     | Docker, Docker Compose             |
| CI/CD                | Jenkins                            |
| Version Control      | Git + GitHub                       |

---

### **2.5 Design and Implementation Constraints**

* The system must use **containerized services** to ensure portability.
* The **Kafka setup must not use Zookeeper** — KRaft mode only.
* All components must be deployable using **Docker Compose**.
* System must handle **simulated high-frequency telemetry data**.
* ML models must be retrainable via automated Jenkins jobs.

---

### **2.6 Assumptions and Dependencies**

* Reliable system clock synchronization across services.
* Kafka and database volumes persist data between container restarts.
* Jenkins host has Docker access.
* Model files are pre-trained or updated on schedule.

---

## **3. System Features**

### **3.1 Feature 1: Real-Time Data Ingestion**

**Description:**
A data simulation service generates telemetry readings (temperature, pressure, vibration, fuel flow, etc.) every few seconds, publishing them to Kafka topics.

**Functional Requirements:**

1. System shall simulate at least 10 sensors per engine.
2. Kafka producer shall stream data at configurable intervals (1–10 seconds).
3. Data schema shall include timestamp, engine ID, and sensor metrics.
4. Data shall be serialized in JSON for interoperability.

---

### **3.2 Feature 2: Data Processing and Storage**

**Description:**
Kafka consumers receive telemetry data, perform cleaning and preprocessing, and store it in PostgreSQL.

**Functional Requirements:**

1. System shall use Kafka consumers for parallel message processing.
2. Preprocessed data shall be stored in PostgreSQL with appropriate indexing.
3. Redis shall be used for caching frequently accessed recent readings.
4. Failures in Kafka consumers must be logged and retried automatically.

---

### **3.3 Feature 3: Predictive Maintenance (ML Pipeline)**

**Description:**
A machine learning model predicts Remaining Useful Life (RUL) and detects anomalies using recent sensor readings.

**Functional Requirements:**

1. The ML model shall be trained using NASA C-MAPSS-based synthetic data.
2. The model shall predict RUL in real-time for incoming data.
3. Predictions and anomalies shall be published to a results topic or cached in Redis.
4. Jenkins pipelines shall support automated model retraining based on new data.

---

### **3.4 Feature 4: Streamlit Dashboard**

**Description:**
An interactive dashboard displays live sensor data, predicted RUL, and alerts.

**Functional Requirements:**

1. Dashboard shall update every 5 seconds (configurable refresh rate).
2. Display:

   * Live sensor metrics (temperature, pressure, etc.)
   * RUL prediction (numeric + trend graph)
   * Anomaly detection status
   * Alert notifications and logs
3. Provide an export option (PDF/CSV) for reports.
4. System shall show “Connection Lost” state if Kafka or DB disconnects.

---

### **3.5 Feature 5: DevOps CI/CD Integration**

**Description:**
A fully automated CI/CD pipeline handles builds, testing, deployment, and retraining workflows.

**Functional Requirements:**

1. Jenkins shall build and test all microservices on each commit.
2. Successful builds trigger Docker image creation and push to registry.
3. Deployment shall be triggered automatically using Docker Compose.
4. Model retraining pipelines shall run periodically or manually via Jenkins.
5. Logs and test results must be archived automatically.

---

## **4. External Interface Requirements**

### **4.1 User Interface**

* Web-based dashboard (Streamlit).
* Minimal dark-theme UI with clean typography and card-based layout.
* Responsive design suitable for laptops and control screens.

### **4.2 Hardware Interfaces**

* Runs on any Docker-capable system (8GB RAM recommended).

### **4.3 Software Interfaces**

* Kafka broker accessible via `PLAINTEXT://kafka:9092`.
* PostgreSQL accessible via internal Docker network.
* REST API endpoints (FastAPI optional) for dashboard-backend communication.

### **4.4 Communications Interfaces**

* Internal Docker network communication between containers.
* HTTP (port 8501) for dashboard.
* TCP/IP for Kafka and database.

---

## **5. Non-Functional Requirements**

| Category            | Requirement                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Performance**     | Must handle 100+ telemetry messages per second with minimal delay.                  |
| **Scalability**     | System should allow scaling Kafka consumers and ML services independently.          |
| **Availability**    | Kafka (KRaft mode) ensures high availability without external dependencies.         |
| **Reliability**     | Data integrity maintained through persistent Kafka and PostgreSQL volumes.          |
| **Security**        | Only authorized users can access Jenkins and database; sensitive configs in `.env`. |
| **Maintainability** | Modular architecture allows independent service updates.                            |
| **Portability**     | Dockerized deployment ensures cross-platform portability.                           |
| **Usability**       | Dashboard must be intuitive, minimal, and visually consistent.                      |

---


## **6. Appendices**

* Sample sensor data schema.
* Jenkinsfile snippets for build-test-deploy stages.
* Docker Compose configuration with KRaft-mode Kafka setup.
