def img
pipeline {
    agent {
        kubernetes {
            defaultContainer 'jnlp'
            yamlFile 'podTemplate.yaml'
        }
    }

    stages {
        stage('deliver') {
            stages {
                stage('build-image') {
                    steps {
                        container('docker') {
                            script {
                                def pJson = readJSON file: 'package.json'
                                img = docker.build("${pJson.group}/${pJson.name}:${pJson.version}", "-f Dockerfile .")
                                //onesait-things/platoon-ntl:1.0.0
                            }
                        }
                    }
                }
                stage('deliver-image') {
                    steps {
                        container('docker') {
                            script {
                                def pJson = readJSON file: 'package.json'
                                docker.withRegistry('https://production-registry.devops.onesait.com', 'registry-onesait') {
                                    switch ("${gitlabBranch}") {
                                        case "develop":
                                            img.push("${pJson.version}.RELEASE")
                                            break
                                        case "master":
                                            img.push("latest")
                                            img.push("${pJson.version}")
                                            break
                                    }
                                    sh 'docker images'
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
