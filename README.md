# CVAT-custom-yolov8-segmentation-auto-annotation

This GitHub repository contains the files to use a custom YoloV8 segmentation model serverless with Nuclio on CVAT with GPU support.

First, you need to set up all the serverless functionalities, install Nuclio as a function in Nuclio, etc. See the documentation [here](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/). The files in this repository can be used to deploy the Nuclio functions that run the custom YoloV8 segmentation model. The files will create a CVAT Nuclio project to contain the functions within a Docker container. Commands should be run only after CVAT has been installed using Docker Compose because it runs the Nuclio dashboard which manages all serverless functions.

Besides the custom YoloV8 segmentation model, you need the `function-gpu.yaml` file for setting up Nuclio and a `main.py` script that runs in Nuclio and handles the data from the model.

Once everything is set up, navigate to your CVAT folder from which you can deploy the Nuclio function. If you have a GPU, you can do it like this:

```bash
./serverless/deploy_gpu.sh /home/felix/cvat/serverless/pytorch/ultralytics/custom_yolov8_GPU
```

I believe the scripts should also work if you don't have a GPU.