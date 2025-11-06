#!/bin/bash

# Demo script for Aircraft Engine Monitoring System

echo "🚀 Starting Aircraft Engine Monitoring System Demo"
echo "=================================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Start the system
echo "🔧 Starting all services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to initialize (this may take 2-3 minutes)..."
sleep 30

echo ""
echo "🏥 Checking service health..."

# Check Kafka
if docker-compose exec -T kafka kafka-broker-api-versions.sh --bootstrap-server localhost:9092 >/dev/null 2>&1; then
    echo "✅ Kafka is healthy"
else
    echo "⚠️  Kafka is still starting up..."
fi

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U admin -d aircraft_monitoring >/dev/null 2>&1; then
    echo "✅ PostgreSQL is healthy"
else
    echo "⚠️  PostgreSQL is still starting up..."
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "⚠️  Redis is still starting up..."
fi

echo ""
echo "🎯 System Status:"
docker-compose ps

echo ""
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "⏱️  Please wait 1-2 more minutes for the ML models to train and data to start flowing"

echo ""
echo "🔍 To monitor the system:"
echo "   View all logs:        docker-compose logs -f"
echo "   Data simulator logs:  docker-compose logs -f data-simulator"
echo "   ML service logs:      docker-compose logs -f ml-service"
echo "   Dashboard logs:       docker-compose logs -f dashboard"

echo ""
echo "⏹️  To stop the system:"
echo "   docker-compose down"

echo ""
echo "🎉 Demo setup complete! The system is now running."
echo "   Open http://localhost:8501 in your browser to see the dashboard."