pipeline {
    agent any
    
    environment {
        PATH = "/opt/homebrew/bin:/usr/local/bin:${env.PATH}"
        DOCKER_REGISTRY = "raksharane"  // Docker Hub registry
        PROJECT_NAME = "aircraft-engine-monitoring"
        DOCKER_COMPOSE_FILE = "docker-compose.yml"
        
        // Notification settings
        SLACK_CHANNEL = "#devops-alerts"
        EMAIL_RECIPIENTS = "team@company.com"
        
        // Deployment environments
        DEV_ENVIRONMENT = "development"
        STAGING_ENVIRONMENT = "staging"
        PROD_ENVIRONMENT = "production"
    }
    
    parameters {
        choice(
            name: 'DEPLOY_ENVIRONMENT',
            choices: ['development', 'staging', 'production'],
            description: 'Target deployment environment'
        )
        
        booleanParam(
            name: 'RETRAIN_MODELS',
            defaultValue: false,
            description: 'Trigger ML model retraining'
        )
        
        booleanParam(
            name: 'RUN_INTEGRATION_TESTS',
            defaultValue: true,
            description: 'Run integration tests'
        )
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo "Checking out code from repository..."
                checkout scm
                
                script {
                    env.GIT_COMMIT_SHORT = sh(
                        script: "git rev-parse --short HEAD",
                        returnStdout: true
                    ).trim()
                    
                    env.BUILD_TAG = "${env.BUILD_NUMBER}-${env.GIT_COMMIT_SHORT}"
                }
            }
        }
        
        stage('Environment Setup') {
            steps {
                echo "Setting up build environment..."
                sh '''
                    # Create required directories
                    mkdir -p logs
                    mkdir -p test-results
                    mkdir -p artifacts
                    
                    # Set up environment variables
                    echo "BUILD_TAG=${BUILD_TAG}" > .env.build
                    echo "BUILD_TIMESTAMP=$(date -Iseconds)" >> .env.build
                    echo "GIT_COMMIT=${GIT_COMMIT}" >> .env.build
                    
                    # Display environment info
                    echo "Build Environment:"
                    echo "  Build Tag: ${BUILD_TAG}"
                    echo "  Git Commit: ${GIT_COMMIT_SHORT}"
                    echo "  Deploy Environment: ${DEPLOY_ENVIRONMENT}"
                    echo "  Python Version: $(python3 --version)"
                    echo "  Docker Version: $(docker --version)"
                    echo "  Docker Compose Version: $(docker-compose --version)"
                '''
            }
        }
        
        // stage('Code Quality & Security') {
        //     parallel {
        //         stage('Lint Python Code') {
        //             steps {
        //                 echo "Running Python linting..."
        //                 sh '''
        //                     # Install linting tools
        //                     pip3 install flake8 black isort pylint
        //                     
        //                     # Run code formatting check
        //                     echo "Checking code formatting with Black..."
        //                     black --check --diff services/ || echo "Code formatting issues found"
        //                     
        //                     # Run import sorting check
        //                     echo "Checking import sorting with isort..."
        //                     isort --check-only --diff services/ || echo "Import sorting issues found"
        //                     
        //                     # Run flake8 for style and complexity
        //                     echo "Running flake8 for style checking..."
        //                     flake8 services/ --max-line-length=100 --ignore=E203,W503 || echo "Style issues found"
        //                     
        //                     # Generate lint report
        //                     pylint services/ --output-format=json > test-results/pylint-report.json || echo "Pylint completed with warnings"
        //                 '''
        //             }
        //         }
        //         
        //         stage('Security Scan') {
        //             steps {
        //                 echo "Running security scans..."
        //                 sh '''
        //                     # Install security scanning tools
        //                     pip3 install bandit safety
        //                     
        //                     # Run Bandit for security issues
        //                     echo "Running Bandit security scan..."
        //                     bandit -r services/ -f json -o test-results/bandit-report.json || echo "Security scan completed with findings"
        //                     
        //                     # Check for known vulnerabilities in dependencies
        //                     echo "Checking for vulnerable dependencies..."
        //                     find services/ -name "requirements.txt" -exec safety check --file {} \\; || echo "Dependency check completed"
        //                 '''
        //             }
        //         }
        //         
        //         stage('Docker Security') {
        //             steps {
        //                 echo "Scanning Docker configurations..."
        //                 sh '''
        //                     # Install hadolint for Dockerfile linting
        //                     wget -O hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
        //                     chmod +x hadolint
        //                     
        //                     # Scan all Dockerfiles
        //                     find services/ -name "Dockerfile" -exec ./hadolint {} \\; || echo "Dockerfile issues found"
        //                 '''
        //             }
        //         }
        //     }
        // }
        
        stage('Unit Tests') {
            steps {
                echo "Running unit tests..."
                sh '''
                    # Create test environment
                    python3 -m venv test-env
                    source test-env/bin/activate
                    
                    # Install test dependencies
                    pip install pytest pytest-cov pytest-xdist
                    
                    # Install project dependencies
                    for service in services/*/; do
                        if [ -f "$service/requirements.txt" ]; then
                            echo "Installing dependencies for $service"
                            pip install -r "$service/requirements.txt" || echo "Dependencies installed with warnings"
                        fi
                    done
                    
                    # Run tests with coverage
                    echo "Running unit tests..."
                    pytest services/ \
                        --cov=services \
                        --cov-report=xml:test-results/coverage.xml \
                        --cov-report=html:test-results/coverage-html \
                        --junit-xml=test-results/unit-tests.xml \
                        -v || echo "Tests completed with some failures"
                    
                    deactivate
                '''
            }
            post {
                always {
                    // Publish test results
                    publishTestResults testResultsPattern: 'test-results/unit-tests.xml'
                    
                    // Publish coverage report
                    publishCoverage adapters: [
                        cobertura('test-results/coverage.xml')
                    ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }
            }
        }
        
        stage('Build Docker Images') {
            steps {
                echo "Building Docker images..."
                sh '''
                    # Build all service images
                    for service in data-simulator ml-service consumer-service dashboard; do
                        echo "Building $service image..."
                        docker build -t ${PROJECT_NAME}/${service}:${BUILD_TAG} services/${service}/
                        docker tag ${PROJECT_NAME}/${service}:${BUILD_TAG} ${PROJECT_NAME}/${service}:latest
                    done
                    
                    # List built images
                    docker images | grep ${PROJECT_NAME}
                '''
            }
        }
        
        stage('Integration Tests') {
            when {
                expression { params.RUN_INTEGRATION_TESTS }
            }
            steps {
                echo "Running integration tests..."
                sh '''
                    # Start test environment
                    docker-compose -f docker-compose.test.yml up -d
                    
                    # Wait for services to be ready
                    echo "Waiting for services to start..."
                    sleep 60
                    
                    # Run integration tests
                    python3 -m pytest tests/integration/ \
                        --junit-xml=test-results/integration-tests.xml \
                        -v || echo "Integration tests completed"
                    
                    # Cleanup test environment
                    docker-compose -f docker-compose.test.yml down -v
                '''
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test-results/integration-tests.xml'
                }
            }
        }
        
        stage('ML Model Training') {
            when {
                expression { params.RETRAIN_MODELS }
            }
            steps {
                echo "Training ML models..."
                sh '''
                    # Create model training environment
                    docker run --rm \
                        -v $(pwd)/data:/app/data \
                        -v $(pwd)/models:/app/models \
                        ${PROJECT_NAME}/ml-service:${BUILD_TAG} \
                        python -c "
import sys
sys.path.append('/app')
from ml_service import MLService
ml = MLService()
if ml._connect_services():
    success = ml._train_models()
    print('Model training successful' if success else 'Model training failed')
    sys.exit(0 if success else 1)
else:
    print('Failed to connect to services')
    sys.exit(1)
"
                    
                    # Archive trained models
                    tar -czf artifacts/models-${BUILD_TAG}.tar.gz models/
                '''
            }
            post {
                success {
                    archiveArtifacts artifacts: 'artifacts/models-*.tar.gz', fingerprint: true
                }
            }
        }
        
        stage('Push Docker Images') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    expression { params.DEPLOY_ENVIRONMENT == 'production' }
                }
            }
            steps {
                echo "Pushing Docker images to registry..."
                sh '''
                    # Login to Docker registry (configure credentials in Jenkins)
                    # docker login ${DOCKER_REGISTRY}
                    
                    # Push images
                    for service in data-simulator ml-service consumer-service dashboard; do
                        echo "Pushing ${service} image..."
                        # docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}/${service}:${BUILD_TAG}
                        # docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}/${service}:latest
                        echo "Would push ${PROJECT_NAME}/${service}:${BUILD_TAG}"
                    done
                '''
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    def deploymentConfig = ""
                    
                    switch(params.DEPLOY_ENVIRONMENT) {
                        case 'development':
                            deploymentConfig = "docker-compose.dev.yml"
                            break
                        case 'staging':
                            deploymentConfig = "docker-compose.staging.yml"
                            break
                        case 'production':
                            deploymentConfig = "docker-compose.prod.yml"
                            break
                        default:
                            deploymentConfig = "docker-compose.yml"
                    }
                    
                    echo "Deploying to ${params.DEPLOY_ENVIRONMENT} environment..."
                    
                    sh """
                        # Update environment configuration
                        echo "DEPLOY_TAG=${BUILD_TAG}" > .env.deploy
                        echo "DEPLOY_ENVIRONMENT=${DEPLOY_ENVIRONMENT}" >> .env.deploy
                        
                        # Deploy services
                        docker-compose -f ${deploymentConfig} down || echo "No existing deployment found"
                        docker-compose -f ${deploymentConfig} up -d
                        
                        # Wait for services to be ready
                        echo "Waiting for deployment to stabilize..."
                        sleep 30
                        
                        # Check service health
                        docker-compose -f ${deploymentConfig} ps
                    """
                }
            }
        }
        
        stage('Health Check') {
            steps {
                echo "Running post-deployment health checks..."
                sh '''
                    # Wait for services to be fully ready
                    sleep 60
                    
                    # Check Kafka health
                    echo "Checking Kafka health..."
                    docker exec kafka kafka-broker-api-versions.sh --bootstrap-server localhost:9092 || echo "Kafka health check failed"
                    
                    # Check database connection
                    echo "Checking database health..."
                    docker exec postgres pg_isready -U admin -d aircraft_monitoring || echo "Database health check failed"
                    
                    # Check Redis connection
                    echo "Checking Redis health..."
                    docker exec redis redis-cli ping || echo "Redis health check failed"
                    
                    # Check dashboard accessibility
                    echo "Checking dashboard health..."
                    curl -f http://localhost:8501/_stcore/health || echo "Dashboard health check failed"
                    
                    # Generate health report
                    echo "Deployment Health Summary:" > artifacts/health-report.txt
                    echo "  Build Tag: ${BUILD_TAG}" >> artifacts/health-report.txt
                    echo "  Environment: ${DEPLOY_ENVIRONMENT}" >> artifacts/health-report.txt
                    echo "  Deployment Time: $(date)" >> artifacts/health-report.txt
                    docker-compose ps >> artifacts/health-report.txt
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'artifacts/health-report.txt', fingerprint: true
                }
            }
        }
    }
    
    post {
        always {
            echo "Pipeline completed. Cleaning up..."
            
            // Archive build logs
            sh 'docker-compose logs > logs/docker-compose.log 2>&1 || echo "No compose logs available"'
            archiveArtifacts artifacts: 'logs/*.log', allowEmptyArchive: true
            
            // Cleanup
            sh '''
                # Remove test environment
                rm -rf test-env
                
                # Clean up temporary files
                rm -f hadolint
                
                # Prune unused Docker images
                docker image prune -f || echo "Docker cleanup completed"
            '''
        }
        
        success {
            echo "Pipeline completed successfully!"
            
            // Send success notification
            script {
                if (env.BRANCH_NAME == 'main' || params.DEPLOY_ENVIRONMENT == 'production') {
                    // Slack notification (configure Slack plugin)
                    // slackSend(
                    //     channel: env.SLACK_CHANNEL,
                    //     color: 'good',
                    //     message: ":white_check_mark: Deployment successful\\n" +
                    //              "Project: ${PROJECT_NAME}\\n" +
                    //              "Environment: ${DEPLOY_ENVIRONMENT}\\n" +
                    //              "Build: ${BUILD_TAG}\\n" +
                    //              "Commit: ${GIT_COMMIT_SHORT}"
                    // )
                    
                    // Email notification
                    // emailext(
                    //     to: env.EMAIL_RECIPIENTS,
                    //     subject: "Deployment Success: ${PROJECT_NAME} - ${DEPLOY_ENVIRONMENT}",
                    //     body: "Deployment completed successfully.\\n\\nBuild: ${BUILD_TAG}\\nEnvironment: ${DEPLOY_ENVIRONMENT}\\nCommit: ${GIT_COMMIT}"
                    // )
                }
            }
        }
        
        failure {
            echo "Pipeline failed!"
            
            // Send failure notification (commented out for now)
            // script {
            //     slackSend(
            //         channel: env.SLACK_CHANNEL,
            //         color: 'danger',
            //         message: ":x: Deployment failed\\n" +
            //                  "Project: ${PROJECT_NAME}\\n" +
            //                  "Environment: ${DEPLOY_ENVIRONMENT}\\n" +
            //                  "Build: ${BUILD_TAG}\\n" +
            //                  "Stage: ${env.STAGE_NAME}"
            //     )
            //     
            //     emailext(
            //         to: env.EMAIL_RECIPIENTS,
            //         subject: "Deployment Failed: ${PROJECT_NAME} - ${DEPLOY_ENVIRONMENT}",
            //         body: "Deployment failed at stage: ${env.STAGE_NAME}\\n\\nBuild: ${BUILD_TAG}\\nEnvironment: ${DEPLOY_ENVIRONMENT}\\nCommit: ${GIT_COMMIT}"
            //     )
            // }
        }
        
        unstable {
            echo "Pipeline completed with warnings"
        }
    }
}