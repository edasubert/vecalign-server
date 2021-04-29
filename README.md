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
There are two endpoints available: 
* `http://localhost/align_text` intended for tinkering and testing; pure text interface; discards files after every run
* `http://localhost/align_files_in_place` intended for processing; requires mounted volume with files; preserves files

Documentation is available at `http://localhost/docs` (the server needs to be running).
