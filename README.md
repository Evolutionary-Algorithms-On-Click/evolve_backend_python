# How to setup locally?

1. Clone the repository
```git
git clone https://github.com/Evolutionary-Algorithms-On-Click/evolve-backend-python
```
2. Go to the backend directory

3. Create a virtual environment
```bash
python -m venv venv
```

4. Activate the virtual environment

<br>
macOS / Linux

```bash
source venv/bin/activate
``` 
    
windows


```bash
.\venv\Scripts\activate 
```

> [!NOTE]
> To deactivate the virtual environment run 
> ```bash
> deactivate
> ```

5. Install the dependencies

```bash
pip install -r requirements.txt
```
Update the requirements
```
pip freeze > requirements.txt
```

6. Start the FastAPI app

```bash
uvicorn main:app --reload
```

7. To run in production.

In Windows.
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

In Linux.
```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 -w 4
```

