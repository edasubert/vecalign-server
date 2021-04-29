# Vecalign server
Server implementation of Vecalign sentence aligner (https://github.com/thompsonb/vecalign) using FastAPI. 

## Run
Either build and run the image:
``` 
docker build -t vecalign-server:latest .
docker run -p80:80 vecalign-server:latest
```
Or use docker hub:
```
docker run -p80:80 eduardsubert/vecalign-server:latest
```

## Request
Send POST requests to `http://localhost/`.
Documentation is available at `http://localhost/docs` (the server needs to be running).
