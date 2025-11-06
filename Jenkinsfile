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
                    def buildErrors = []
                    
                    services.each { service ->
                        echo "Building ${service}..."
                        try {
                            sh """
                                docker build \
                                    -t ${DOCKER_REGISTRY}/aircraft-${service}:latest \
                                    -t ${DOCKER_REGISTRY}/aircraft-${service}:${version} \
                                    -f services/${service}/Dockerfile \
                                    services/${service}/
                            """
                            echo "✅ ${service} built successfully"
                        } catch (Exception e) {
                            buildErrors.add("${service}: ${e.message}")
                            echo "❌ Failed to build ${service}"
                        }
                    }
                    
                    if (buildErrors.size() > 0) {
                        error("Build failed for: ${buildErrors.join(', ')}")
                    }
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                echo "Pushing images to Docker Hub..."
                withCredentials([usernamePassword(credentialsId: DOCKER_CREDENTIALS_ID, usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    script {
                        def services = ['data-simulator', 'ml-service', 'consumer-service', 'dashboard']
                        def version = env.BUILD_NUMBER
                        def pushErrors = []
                        
                        try {
                            // Login to Docker Hub
                            sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                            
                            services.each { service ->
                                echo "Pushing ${service}..."
                                try {
                                    sh """
                                        docker push ${DOCKER_REGISTRY}/aircraft-${service}:latest
                                        docker push ${DOCKER_REGISTRY}/aircraft-${service}:${version}
                                    """
                                    echo "✅ ${service} pushed successfully"
                                } catch (Exception e) {
                                    pushErrors.add("${service}: ${e.message}")
                                    echo "❌ Failed to push ${service}"
                                }
                            }
                        } finally {
                            // Always logout
                            sh 'docker logout || true'
                        }
                        
                        if (pushErrors.size() > 0) {
                            error("Push failed for: ${pushErrors.join(', ')}")
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
            echo "Build Number: ${env.BUILD_NUMBER}"
        }
        
        failure {
            echo "❌ Pipeline failed!"
            echo "Check the console output above for details"
        }
        
        always {
            echo "Cleaning up..."
            sh '''
                # Remove dangling images
                docker image prune -f || true
                
                # Remove build cache (optional - uncomment if disk space is an issue)
                # docker builder prune -f || true
            '''
        }
    }
}
