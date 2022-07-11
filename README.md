# svfpy

Python project and library for calculating the SVF (Sky View Factor) from a given DSM (Digital Surface Model)


## Using Local Python Installation

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Using Docker Containner

```
docker build -t svfpy .
docker run -it -v ~/dev/svfpy:/opt/svfpy svfpy bash
```