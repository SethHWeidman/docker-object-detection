# docker-object-detection
A repo containing the starting files for a Docker image that runs object detection using TensorFlow

## Description

Following [here](https://github.com/floydhub/dl-docker), the Dockerfile installs Ubuntu 16.04 and all the dependencies needed to do Deep Learning.

The extra steps come from the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

## Instructions

Pull the Docker image using `docker pull sethweidman/docker-object-detection`.

Once pulled, run using `docker run -it -p 8898:8898 <IMAGE ID> jupyter notebook --allow-root`. This will launch a Jupyter Notebook inside the container and make available port 8898 (not 8888, to avoid conflicting with any other Jupyter Notebooks you might have running).

Then, you can navigate to `http://localhost:8898` to see a Jupyter notebook with the TensorFlow object detection tutorial notebook.

Finally, you can navigate to the `models/research/object_detection` folder and run the cells in the `object_detection_tutorial.ipynb` notebook.
