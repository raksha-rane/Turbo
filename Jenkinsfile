pipeline {
    agent any
    
    environment {
        PATH = "/opt/homebrew/bin:/usr/local/bin:${env.PATH}"
        DOCKER_REGISTRY = "raksharane"
        DOCKER_CREDENTIALS_ID = "dockerhub-credentials"
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo "Checking out code from GitHub..."
                checkout scm
            }
        }
        
        stage('Build Docker Images') {
            steps {
                echo "Building Docker images..."
                script {
                    def services = ['data-simulator', 'ml-service', 'consumer-service', 'dashboard']
                    def version = env.BUILD_NUMBER
                    
                    services.each { service ->
                        echo "Building ${service}..."
                        sh """
                            docker build \
                                -t ${DOCKER_REGISTRY}/aircraft-${service}:latest \
                                -t ${DOCKER_REGISTRY}/aircraft-${service}:${version} \
                                -f services/${service}/Dockerfile \
                                services/${service}/
                        """
                    }
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                echo "Pushing images to Docker Hub..."
                script {
                    docker.withRegistry('https://index.docker.io/v1/', DOCKER_CREDENTIALS_ID) {
                        def services = ['data-simulator', 'ml-service', 'consumer-service', 'dashboard']
                        def version = env.BUILD_NUMBER
                        
                        services.each { service ->
                            echo "Pushing ${service}..."
                            sh """
                                docker push ${DOCKER_REGISTRY}/aircraft-${service}:latest
                                docker push ${DOCKER_REGISTRY}/aircraft-${service}:${version}
                            """
                        }
                    }
                }
            }
        }
    }
    
    post {
        success {
            echo "✅ Pipeline completed successfully!"
            echo "Images pushed to Docker Hub: https://hub.docker.com/u/${DOCKER_REGISTRY}"
        }
        
        failure {
            echo "❌ Pipeline failed!"
        }
        
        always {
            echo "Cleaning up..."
            sh 'docker image prune -f || true'
        }
    }
}
