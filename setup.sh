# Setup script for Aircraft Engine Monitoring System

#!/bin/bash

set -e

echo "🚀 Setting up Aircraft Engine Monitoring System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose found: $(docker-compose --version)"
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEM_GB" -lt 8 ]; then
            print_warning "System has less than 8GB RAM. Performance may be affected."
        else
            print_status "Memory check passed: ${MEM_GB}GB available"
        fi
    fi
    
    # Check disk space
    if command -v df &> /dev/null; then
        DISK_GB=$(df -BG . | awk 'NR==2{print int($4)}')
        if [ "$DISK_GB" -lt 10 ]; then
            print_warning "Less than 10GB disk space available. Consider cleaning up."
        else
            print_status "Disk space check passed: ${DISK_GB}GB available"
        fi
    fi
}

# Create necessary directories
create_directories() {
    print_step "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p models
    mkdir -p artifacts
    mkdir -p test-results
    
    print_status "Directories created successfully"
}

# Build Docker images
build_images() {
    print_step "Building Docker images..."
    
    echo "Building data simulator..."
    docker build -t aircraft-monitoring/data-simulator:latest services/data-simulator/
    
    echo "Building ML service..."
    docker build -t aircraft-monitoring/ml-service:latest services/ml-service/
    
    echo "Building consumer service..."
    docker build -t aircraft-monitoring/consumer-service:latest services/consumer-service/
    
    echo "Building dashboard..."
    docker build -t aircraft-monitoring/dashboard:latest services/dashboard/
    
    print_status "All Docker images built successfully"
}

# Start services
start_services() {
    print_step "Starting services..."
    
    # Stop any existing services
    docker-compose down -v 2>/dev/null || true
    
    # Start infrastructure services first
    print_status "Starting infrastructure services (Kafka, PostgreSQL, Redis)..."
    docker-compose up -d kafka postgres redis
    
    # Wait for infrastructure services to be ready
    print_status "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Check infrastructure health
    echo "Checking Kafka..."
    docker-compose exec -T kafka kafka-broker-api-versions.sh --bootstrap-server localhost:9092 || {
        print_error "Kafka is not ready"
        exit 1
    }
    
    echo "Checking PostgreSQL..."
    docker-compose exec -T postgres pg_isready -U admin -d aircraft_monitoring || {
        print_error "PostgreSQL is not ready"
        exit 1
    }
    
    echo "Checking Redis..."
    docker-compose exec -T redis redis-cli ping || {
        print_error "Redis is not ready"
        exit 1
    }
    
    print_status "Infrastructure services are ready"
    
    # Start application services
    print_status "Starting application services..."
    docker-compose up -d
    
    # Wait for all services to start
    sleep 20
    
    print_status "All services started successfully"
}

# Create Kafka topics
create_kafka_topics() {
    print_step "Creating Kafka topics..."
    
    # Create telemetry topic
    docker-compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 \
        --create --if-not-exists --topic engine-telemetry --partitions 3 --replication-factor 1
    
    # Create predictions topic
    docker-compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 \
        --create --if-not-exists --topic engine-predictions --partitions 3 --replication-factor 1
    
    # Create alerts topic
    docker-compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 \
        --create --if-not-exists --topic engine-alerts --partitions 3 --replication-factor 1
    
    # List topics
    print_status "Created Kafka topics:"
    docker-compose exec -T kafka kafka-topics.sh --bootstrap-server localhost:9092 --list
}

# Verify deployment
verify_deployment() {
    print_step "Verifying deployment..."
    
    # Check service status
    print_status "Service status:"
    docker-compose ps
    
    # Check logs for errors
    if docker-compose logs --tail=50 | grep -i error; then
        print_warning "Some errors found in logs. Please check individual service logs."
    fi
    
    # Test dashboard accessibility
    echo "Testing dashboard accessibility..."
    sleep 10
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        print_status "Dashboard is accessible at http://localhost:8501"
    else
        print_warning "Dashboard may not be ready yet. Please wait a few more minutes."
    fi
    
    print_status "Deployment verification completed"
}

# Show system information
show_system_info() {
    print_step "System Information"
    
    echo ""
    echo -e "${GREEN}🎉 Aircraft Engine Monitoring System Setup Complete!${NC}"
    echo ""
    echo "📊 Access Points:"
    echo "  • Dashboard:    http://localhost:8501"
    echo "  • Kafka:        localhost:9092"
    echo "  • PostgreSQL:   localhost:5432 (admin/secure_password)"
    echo "  • Redis:        localhost:6379"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  • View logs:           docker-compose logs -f"
    echo "  • Check status:        docker-compose ps"
    echo "  • Stop system:         docker-compose down"
    echo "  • Restart system:      docker-compose restart"
    echo "  • View service logs:   docker-compose logs <service-name>"
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Open dashboard at http://localhost:8501"
    echo "  2. Monitor engine data streams"
    echo "  3. Check system alerts and predictions"
    echo "  4. Review logs for any issues"
    echo ""
    echo "🆘 Troubleshooting:"
    echo "  • If services fail to start, check Docker resources"
    echo "  • If dashboard is not accessible, wait 2-3 minutes for full startup"
    echo "  • Check individual service logs: docker-compose logs <service>"
    echo ""
}

# Cleanup function
cleanup() {
    print_step "Cleaning up temporary files..."
    rm -f .env.build 2>/dev/null || true
    print_status "Cleanup completed"
}

# Main setup process
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║        Aircraft Engine Monitoring System Setup              ║"
    echo "║                     Version 1.0                             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run setup steps
    check_prerequisites
    create_directories
    build_images
    start_services
    create_kafka_topics
    verify_deployment
    show_system_info
    
    print_status "Setup completed successfully! 🎉"
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "clean")
        print_step "Cleaning up system..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_status "System cleaned up"
        ;;
    "restart")
        print_step "Restarting system..."
        docker-compose restart
        verify_deployment
        print_status "System restarted"
        ;;
    "status")
        print_step "System status:"
        docker-compose ps
        echo ""
        print_step "Resource usage:"
        docker stats --no-stream
        ;;
    "logs")
        docker-compose logs --tail=100 -f
        ;;
    *)
        echo "Usage: $0 {setup|clean|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  setup   - Full system setup (default)"
        echo "  clean   - Clean up all containers and volumes"
        echo "  restart - Restart all services"
        echo "  status  - Show system status"
        echo "  logs    - Show live logs"
        exit 1
        ;;
esac