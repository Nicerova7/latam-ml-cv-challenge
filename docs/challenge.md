##📘 Challenge Documentation

### Part I

In this part, I included the __data__ folder with the full dataset: train, test, and val. I also included data_train.yaml, which is a modified version of the original data.yaml, updating the paths to:

    train: ../../data/train/images
    val: ../../data/valid/images
    test: ../../data/test/images

This was necessary because, on Windows, those directories behave as if they were root-level paths even when placed inside a subfolder.

For the same reason, I rewrote the __yolo_label_paths()__ function. The original version mixed \ and / in the paths, which caused issues when loading the labels.

With these configurations, I was able to analyze and train the model. Here is a short summary of what I found:

    During training I evaluated the model using the built-in YOLO metrics.
    The final scores were mAP50 = 0.262 and mAP50-95 = 0.176, which are expected for a dataset with strong class imbalance and several small objects.

    While reviewing the per-class results, I noticed a very uneven distribution of performance:

    forklift and person reached the highest scores.
    This matched what I saw during the dataset exploration: they are the most common classes and appear in clear, consistent poses across many images.

    Several classes such as traffic light, van, road sign, and gloves had extremely low or even zero mAP.
    After checking the dataset statistics, it was clear that these classes had very few samples, and some appeared only 1–2 times. The model simply didn’t have enough examples to learn meaningful patterns.

    I also checked object sizes during exploration. Many minority classes corresponded to tiny objects, which explains the low recall even further. With the training resolution used (640×640), some of these objects became almost indistinguishable.

Finally, I added a visual example of a low-resolution image, and I adjusted line_width and color_mode in the inference bounding-box visualization to make the detections easier to see.

__**Important**__: When uploading everything to GitHub, I did not include the entire data folder, only the minimal subset required to run the tests.

### Part II

For this part, I focused on creating a clean FastAPI service.
Instead of loading the PyTorch .pt model directly, I decided to use the exported ONNX model. This is closer to a real deployment scenario because ONNX reduces framework dependencies and makes the inference pipeline easier to integrate in cloud environments.

I reorganized the API structure by adding a small config module (challenge/config.py) to store constants like IMGSZ. This keeps the API file simpler and aligns with standard practices and the loading section now is: 


    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    

##### Improved health check

The original challenge template only returned "model_loaded".
I extended it so it actually reports the state of both the ONNX model and the classes file adding:


        return {
            "status": "ok",
            "model": "loaded" if session is not None else "unloaded",
            "classes": "loaded" if len(class_names) > 0 else "unloaded",
        }
    

##### Prediction endpoint

For the /predict endpoint, I used a minimal preprocessing pipeline based on OpenCV and ONNXRuntime:

* resize
* normalize
* transpose
* forward pass
* simple postprocessing + NMS

During post-processing, the bounding boxes produced by the ONNX model can sometimes fall slightly outside image boundaries.
To prevent this, and to ensure the __tests__ would not fail due to negative coordinates, I added a bounding-box sanitization step:

    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(0, min(x2, orig_w - 1))
    y2 = max(0, min(y2, orig_h - 1))

This guarantees that all detections stay within valid image coordinates.

Finally, to ensure full compatibility with the test suite, I kept all output keys, shapes, and field names exactly as expected (class, confidence, bbox).

### Part III

In this part, I deployed the object-detection API built with FastAPI and ONNX Runtime.
The goal was to expose the model as an HTTP endpoint and host it on a cloud provider, as requested in the challenge instructions.

#### Steps performed
1. Containerization. I created a production-ready Docker image using:
    - python:3.12-slim
    - System dependencies required by OpenCV (libglib2.0-0, libgl1, libopencv-core-dev)
    - Project dependencies installed via requirements.txt

This allowed me to run the API inside a minimal and reproducible environment (that's why I am using 3.12-slim).

2. Deployment on Google Cloud Run. For deployment, I followed these steps:
    - Created a new GCP project and enabled Artifact Registry + Cloud Run.
    - Built and pushed the container image to Artifact Registry.
    - Deployed the Cloud Run service with:
        * --allow-unauthenticated
        * autoscaling enabled (min 0, max 1)
        * us-central1 region

3. After deployment, the service provided a public HTTPS URL (https://yolo11-api-111036914172.us-central1.run.app)
This URL was added in line 26 of the Makefile, as required by the challenge.

Cloud Run was selected because it supports containerized inference easily and it was recommended for this challenge. The API remained available during the entire evaluation period.

### Part IV

The final part of the challenge required setting up a proper Continuous Integration and Continuous Delivery pipeline.
I implemented both workflows using GitHub Actions, following the GitFlow conventions.

#### Continuous Integration (ci.yml)

The CI workflow validates every change pushed to develop and main. In these steps:

- Checks out the repository.
- Uses a Python 3.12-slim environment (matching the Dockerfile runtime).
- Installs the additional system-level packages required for OpenCV (libglib2.0-0, libgl1, etc.).
- Installs:
    * requirements.txt
    * requirements-test.txt

- Runs all API and dataset-related tests using 
        
    * make api-test
    

__**Important**__: Since the full dataset cannot be stored in the repository due to size, I included a minimal subset of 6 test images and labels, enough for the CI tests to execute correctly.

Finally, the tests executed successfully, you can see the run here: https://github.com/Nicerova7/latam-ml-cv-challenge/actions/runs/19907634205


#### Continuous Delivery (cd.yml)

The CD workflow only triggers on pushes to the main branch or manual executions (workflow_dispatch)

What the CD pipeline does:
- Authenticates to Google Cloud using the GCP_SA_KEY GitHub secret.
- Configures Docker to push images to Artifact Registry.
- Builds the container image using the repository root as context.
- Pushes the image to:

    $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest

- Deploys the image to Cloud Run using:

    google-github-actions/deploy-cloudrun@v2


Result

Every push to main automatically:
* builds a new API version,
* uploads it to Artifact Registry,
* deploys it to Cloud Run.

This ensures consistent, automated, and reproducible deployments without manual intervention.