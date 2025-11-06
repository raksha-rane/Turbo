#!/bin/bash

# Quick validation script to check if services are running

echo "Checking Docker services..."
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi
echo "✅ Docker is running"

# Check containers
echo ""
echo "Container Status:"
echo "=================="

containers=("kafka" "postgres" "redis" "data-simulator" "ml-service" "consumer-service" "dashboard")

for container in "${containers[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        status=$(docker inspect --format='{{.State.Status}}' $container 2>/dev/null)
        health=$(docker inspect --format='{{.State.Health.Status}}' $container 2>/dev/null)
        
        if [ "$status" = "running" ]; then
            if [ "$health" = "healthy" ] || [ "$health" = "<no value>" ]; then
                echo "✅ $container: running"
            elif [ "$health" = "starting" ]; then
                echo "⏳ $container: starting..."
            else
                echo "⚠️  $container: $health"
            fi
        else
            echo "❌ $container: $status"
        fi
    else
        echo "❌ $container: not found"
    fi
done

echo ""
echo "Quick Service Tests:"
echo "===================="

# Test Kafka
if docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
    echo "✅ Kafka: responding"
else
    echo "❌ Kafka: not responding"
fi

# Test PostgreSQL
if docker exec postgres pg_isready -U admin >/dev/null 2>&1; then
    echo "✅ PostgreSQL: responding"
else
    echo "❌ PostgreSQL: not responding"
fi

# Test Redis
if docker exec redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis: responding"
else
    echo "❌ Redis: not responding"
fi

# Test Dashboard
if curl -s http://localhost:8501 >/dev/null 2>&1; then
    echo "✅ Dashboard: accessible at http://localhost:8501"
else
    echo "⏳ Dashboard: not yet accessible"
fi

echo ""
echo "To view logs: docker logs <service-name>"
echo "To restart: docker-compose restart"
echo "To stop: docker-compose down"
