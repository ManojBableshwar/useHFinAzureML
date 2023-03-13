
This sample shows how to fine tune the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model to detect emotions [using emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) and deploy it to an endpoint for real time inference.  

We use the AzureML CLI, but the same can be done using Python SDK

## 0. Setup CLI environment
Learn more: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public
```
az account set -s <your_subscription>
az configure --defaults group=<your_resource_group> workspace=<your_workspace> location=<your_workspace_location>
```

## 1. Create AzureML Training environment 
Learn more: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?tabs=cli 

```
az ml environment create --file finetune/environment/transformers-training.yml
```

## 2. Submit the fine-tuning job

Key inputs to the [job config](./finetune/job/job.yml)
* [Trining code](./finetune/job/finetune-evaluate.py) and [training job config](./finetune/job/job.yml)
* base_model_name takes a fill mask model as input. Tested only for bert-base-uncased. 
* Docker image with transformers installed is the environment created in previous step.
* The training script has the emotions dataset with label-id mapping hard coded, todo - refactor as a parameter
* GPU compute cluster `sample-finetune-cluster`. Change to your custer name as needed. 

``` 
pipeline_job_name=$(az ml job create --file finetune/job/job.yml --query name -o tsv)
echo $pipeline_job_name

```

## 3. Register the fine-tuned model to workspace
Fetch the fine-tuning job name and look for `trained_model` output in the fine-tuning job to create the model. This links the model to the job that trained (fine-tuned in this case) it. 

```
train_job_name=$(az ml job list --parent-job-name $pipeline_job_name --query [0].name | sed 's/\"//g') 
echo "training job name: $train_job_name"
az ml model create --name bert-emotion --version 1 --type custom_model --path azureml://jobs/$train_job_name/outputs/trained_model
```
## 4. Create environment for inference
Ideally, we should be able to use the training environment, but looks like the base image needs some AzureML specific inference packages, specifically `azureml-inference-server-http`

```
az ml environment create --file inference/online-endpoint/transformers-inference/transformers-inference.yml 
```

## 5. Create online-endpoint and deploy fine-tuned model

* Endpoint names need to unique in entire Azure region. Hence append Unix time in epoch format to endpoint name.
* [Inference code](./inference/online-endpoint/deployment/score.py) loads transformers pipeline using the finetuned model. To do, figure out how to load the model to transformers pipeline in init, and not repeat it for each scorig request i.e. `run()`

```
version=$(date +%s)
az ml online-endpoint create --name bert-emotion-$version
az ml online-deployment create --file --endpoint-name bert-emotion-$version --all-traffic
```

## 6. Invoke the inference endpoint

```
az ml online-endpoint invoke --name bert-emotion-$version --request-file ./inference/sample-inference.json 
```

Sample output for the [sample input](./inference/sample-inference.json):

```
"[{\"label\": \"joy\", \"score\": 0.9931063055992126}, {\"label\": \"joy\", \"score\": 0.9842185378074646}, {\"label\": \"sadness\", \"score\": 0.8363726139068604}]"
```



