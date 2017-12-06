# docker-object-detection
A repo containing the starting files for a Docker image that runs object detection using TensorFlow

## Description

Following [here](https://github.com/floydhub/dl-docker), the Dockerfile installs Ubuntu 16.04 and all the dependencies needed to do Deep Learning.

The extra steps come from the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

## Instructions

Pull the Docker image using `docker pull sethweidman/docker-object-detection`.

Once pulled, run using `docker run -it -p 5000:5000 <IMAGE ID> python detection_app.py`. This will launch a Flask app inside the Docker container.

Then, from this folder, run:

`python3 request_image.py test.jpg`

Which will do object detection on the image you pass in.

--




Install Flask!
