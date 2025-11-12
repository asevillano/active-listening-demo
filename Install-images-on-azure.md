## How install STT and PII Detection images on Azure

Just for demo purposes (since it doesn’t make sense for a real project), the following document explains how to deploy the STT and PII Detection images in Azure Container.

### Deploy STT container on Azure:
1) Create Container Registry (ACR):

```az acr create --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-ACR-NAME> --sku Basic```

2) Initiate session in ACR:

```az acr login --name <YOUR-ACR-NAME>```

3) Download the image for ACR:

```docker pull mcr.microsoft.com/azure-cognitive-services/speechservices/speech-to-text:latest```

4) Tag the image for ACR:

```docker tag mcr.microsoft.com/azure-cognitive-services/speechservices/speech-to-text:latest <YOUR-ACR-NAME>.azurecr.io/speech-to-text:latest```

5) Upload the image to ACR:

```docker push <YOUR-ACR-NAME>.azurecr.io/speech-to-text:latest```

6) Deploy in Azure Container using the image in ACR:

- Enable admin:
```az acr update -n <YOUR-ACR-NAME> --admin-enabled true```
- Get the ACR username:
```az acr credential show --name <YOUR-ACR-NAME> --query "username" -o tsv```
- Get the ACR password:
```az acr credential show --name <YOUR-ACR-NAME> --query "passwords[0].value" -o tsv```
- Deploy:

```az container create --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-STT-CONTAINER-NAME> --image mcr.microsoft.com/azure-cognitive-services/speechservices/speech-to-text:latest --cpu 4 --memory 8 --ports 5000 --os-type Linux --ip-address Public --registry-login-server <YOUR-ACR-NAME>.azurecr.io --registry-username <YOUR-ACR-USERNAME> --registry-password <ACR-PASSWORD> --environment-variables Eula=accept ApiKey=<YOUR-SPEECH-APIKEY> Billing=https://<REGION-OF-YOUR-SPEECH-SERVICE>.api.cognitive.microsoft.com/```

7) Get public IP and test the container:

```az container show --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-STT-CONTAINER-NAME> --query "{IP:ipAddress.ip}" --output table```

Test it: 

```http://<YOUR-STT-CONTAINER-NAME>.<YOUR-REGION>.azurecontainer.io:5000/```

Or

```curl http://<IP_PUBLICA>:5000/ready```

### Deploy TTS container on Azure:
1) Create Container Registry (ACR):

```az acr create --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-ACR-NAME> --sku Basic```

2) Initiate session in ACR:

```az acr login --name <YOUR-ACR-NAME>```

3) Download the image for ACR:

```docker pull mcr.microsoft.com/azure-cognitive-services/speechservices/neural-text-to-speech:latest```

4) Tag the image for ACR:

```docker tag mcr.microsoft.com/azure-cognitive-services/speechservices/neural-text-to-speech:latest <YOUR-ACR-NAME>.azurecr.io/neural-text-to-speech:latest```

5) Upload the image to ACR:

```docker push <YOUR-ACR-NAME>.azurecr.io/neural-text-to-speech:latest```

6) Deploy in Azure Container using the image in ACR:

- Enable admin:
```az acr update -n <YOUR-ACR-NAME> --admin-enabled true```
- Get the ACR username:
```az acr credential show --name <YOUR-ACR-NAME> --query "username" -o tsv```
- Get the ACR password:
```az acr credential show --name <YOUR-ACR-NAME> --query "passwords[0].value" -o tsv```
- Deploy:

```az container create --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-TTS-CONTAINER-NAME> --image mcr.microsoft.com/azure-cognitive-services/speechservices/neural-text-to-speech:latest --cpu 6 --memory 12 --ports 5000 --os-type Linux --ip-address Public --registry-login-server <YOUR-ACR-NAME>.azurecr.io --registry-username <YOUR-ACR-USERNAME> --registry-password <ACR-PASSWORD> --environment-variables Eula=accept ApiKey=<YOUR-SPEECH-APIKEY> Billing=https://<REGION-OF-YOUR-SPEECH-SERVICE>.api.cognitive.microsoft.com/```

7) Get public IP and test the container:

```az container show --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-TTS-CONTAINER-NAME> --query "{IP:ipAddress.ip}" --output table```

Test it: 

```http://<YOUR-TTS-CONTAINER-NAME>.<YOUR-REGION>.azurecontainer.io:5000/```

Or

```curl http://<IP_PUBLICA>:5000/ready```

### Deploy PII Detector Container on Azure:

1) Create Container Registry (ACR)

```az acr create --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-ACR-NAME> --sku Basic```

2) Initiate session in ACR

```az acr login --name <YOUR-ACR-NAME>```

3) Download the image for ACR:

```docker pull mcr.microsoft.com/azure-cognitive-services/textanalytics/pii:latest```

4) Tag the image for ACR:

```docker tag mcr.microsoft.com/azure-cognitive-services/textanalytics/pii:latest <YOUR-ACR-NAME>.azurecr.io/textanalytics:latest```

5) Upload the image to ACR:

```docker push <YOUR-ACR-NAME>.azurecr.io/textanalytics:latest```

6) Deploy in Azure Container using the image in ACR:

- Enable admin:
```az acr update -n <YOUR-ACR-NAME> --admin-enabled true```
- Get the ACR username:
```az acr credential show --name <YOUR-ACR-NAME> --query "username" -o tsv```
- Get the ACR password:
```az acr credential show --name <YOUR-ACR-NAME> --query "passwords[0].value" -o tsv```
- Deploy:

```az container create --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-PII-CONTAINER-NAME> --image <YOUR-ACR-NAME>.azurecr.io/textanalytics:latest --cpu 2 --memory 4 --ports 5000 --os-type Linux --ip-address Public --registry-login-server <YOUR-ACR-NAME>.azurecr.io --registry-username <YOUR-ACR-USERNAME> --registry-password <ACR-PASSWORD> --environment-variables Eula=accept ApiKey=<YOUR-AILANGUAGE-APIKEY> Billing=https://<YOUR-AILANGUAGE-RESOURCE-NAME>.cognitiveservices.azure.com/```

7) Get public IP and test the container:

```az container show --resource-group <YOUR-RESOURCE-GROUP-NAME> --name <YOUR-PII-CONTAINER-NAME> --query "{IP:ipAddress.ip}" --output table```

Test it:

```http://<YOUR-PII-CONTAINER-NAME>.<YOUR-REGION>.azurecontainer.io:5000/```
or

```curl http://<IP_PUBLICA>:5000/ready```

