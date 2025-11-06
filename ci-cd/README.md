# CI/CD Configuration

This directory contains Jenkins configuration files for the Aircraft Engine Monitoring project.

## Files

### `plugins.txt`
List of required Jenkins plugins for this project. Install these plugins in your Jenkins instance:

```bash
# To install plugins in Jenkins
cat plugins.txt | while read plugin; do
    jenkins-cli install-plugin $plugin
done
```

**Key Plugins:**
- **Docker Pipeline** - Build and push Docker images
- **GitHub Integration** - Connect to GitHub repositories
- **Pipeline** - Declarative pipeline support
- **Credentials Binding** - Secure credential management

### `jenkins.yaml`
Jenkins Configuration as Code (JCasC) file. This can automatically configure Jenkins if you use the Configuration as Code plugin.

**To use:**
1. Install the "Configuration as Code" plugin in Jenkins
2. Go to: Manage Jenkins → Configuration as Code
3. Upload this jenkins.yaml file
4. Jenkins will auto-configure based on this file

## Jenkins Setup Guide

### 1. Install Jenkins

**Option A: Using Docker (Recommended for Demo)**
```bash
docker run -d \
  --name jenkins \
  -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts
```

**Option B: Using Homebrew (Mac)**
```bash
brew install jenkins-lts
brew services start jenkins-lts
```

### 2. Initial Setup

1. Open http://localhost:8080
2. Get initial password:
   ```bash
   # Docker
   docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
   
   # Homebrew
   cat /Users/Shared/Jenkins/Home/secrets/initialAdminPassword
   ```
3. Install suggested plugins
4. Create admin user

### 3. Install Required Plugins

Go to **Manage Jenkins** → **Manage Plugins** → **Available**

Install from `plugins.txt`:
- Docker Pipeline
- GitHub Integration
- Pipeline
- Credentials Binding
- Blue Ocean (optional, better UI)

### 4. Configure Credentials

Go to **Manage Jenkins** → **Manage Credentials** → **Global**

**Add GitHub Credentials:**
- Kind: Username with password
- Username: Your GitHub username
- Password: GitHub Personal Access Token
- ID: `github-credentials`

**Add Docker Hub Credentials:**
- Kind: Username with password
- Username: Your Docker Hub username
- Password: Your Docker Hub password
- ID: `dockerhub-credentials`

### 5. Create Pipeline Job

1. **New Item** → Enter name: `aircraft-engine-monitoring-pipeline`
2. Select **Pipeline** → OK
3. **Pipeline** section:
   - Definition: "Pipeline script from SCM"
   - SCM: Git
   - Repository URL: Your GitHub repo URL
   - Credentials: Select github-credentials
   - Branch: `*/main`
   - Script Path: `Jenkinsfile`
4. **Save**

### 6. Configure GitHub Webhook (Optional)

For automatic builds on push:

1. Go to your GitHub repository
2. **Settings** → **Webhooks** → **Add webhook**
3. Payload URL: `http://YOUR_JENKINS_URL:8080/github-webhook/`
4. Content type: `application/json`
5. Events: "Just the push event"
6. Save

## Pipeline Stages

The main `Jenkinsfile` includes these stages:

1. **Checkout** - Pull code from GitHub
2. **Environment Setup** - Configure Python and dependencies
3. **Unit Tests** - Run service unit tests
4. **Integration Tests** - Test service interactions
5. **Code Quality** - Run Pylint, Flake8, Black
6. **Security Scan** - Check for vulnerabilities
7. **Build Docker Images** - Build all microservices
8. **Push to Registry** - Upload to Docker Hub
9. **Deploy** - Deploy to environment
10. **Health Check** - Verify deployment

## Troubleshooting

### Jenkins Can't Connect to Docker
```bash
# Give Jenkins permission to Docker socket
sudo chmod 666 /var/run/docker.sock
```

### Pipeline Fails on Build
```bash
# Check Docker is running
docker ps

# Check Jenkins has Docker access
docker exec jenkins docker ps
```

### GitHub Webhook Not Triggering
- Ensure Jenkins is accessible from internet (use ngrok for local testing)
- Check webhook delivery in GitHub repository settings
- Verify webhook URL is correct

## Manual Build Trigger

To manually trigger a build:
1. Go to Jenkins dashboard
2. Click on your pipeline job
3. Click **Build Now**
4. Watch the build progress in **Build History**

## Viewing Build Logs

1. Click on build number in Build History
2. Click **Console Output**
3. View real-time logs

## CI/CD Best Practices Implemented

✅ **Automated Testing** - Tests run on every commit
✅ **Code Quality Checks** - Pylint and Flake8 enforcement
✅ **Security Scanning** - Dependency vulnerability checks
✅ **Docker Image Versioning** - Tagged with build numbers
✅ **Automated Deployment** - Deploy on successful builds
✅ **Health Checks** - Verify deployment success
✅ **Rollback Capability** - Previous images available

## Environment Variables

Configure these in Jenkins pipeline or credentials:

```bash
DOCKER_HUB_CREDENTIALS  # Docker Hub login (configured as credential)
GITHUB_CREDENTIALS      # GitHub access (configured as credential)
DOCKER_REGISTRY         # Docker registry URL (in Jenkinsfile)
PROJECT_NAME            # Project name (in Jenkinsfile)
```

## Additional Resources

- [Jenkins Pipeline Documentation](https://www.jenkins.io/doc/book/pipeline/)
- [Docker Pipeline Plugin](https://plugins.jenkins.io/docker-workflow/)
- [Jenkins Configuration as Code](https://github.com/jenkinsci/configuration-as-code-plugin)

---

**Note**: The main pipeline configuration is in the root `Jenkinsfile`. This directory contains only reference and configuration files.
