# Put the app online (free) so it runs anywhere, anytime

Before deploying, put the REAL dataset in place so the live site doesn't show the synthetic sample:

    python download_dataset.py        # or copy student-mat.csv into the data/ folder

Check that `data/student-mat.csv` exists. Then pick ONE option below.

---------------------------------------------------------------
## Option A - Render.com (easiest; gives you a public URL)

1. Create a free account at https://github.com and make a new repository.
   Upload ALL files of this folder (including data/student-mat.csv).
2. Create a free account at https://render.com (sign in with GitHub).
3. Click New + -> Web Service -> pick your repository.
4. Render reads `render.yaml` automatically. If it asks, use:
     Build command: pip install -r requirements.txt
     Start command: gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT
5. Click Deploy. After 3-5 minutes you get a link like https://student-performance-predictor.onrender.com

Note: the free plan "sleeps" after ~15 minutes without visitors; the first visit afterwards
takes ~30-60 seconds to wake up.

---------------------------------------------------------------
## Option B - Hugging Face Spaces (free, does not sleep as aggressively)

1. Create an account at https://huggingface.co -> New Space -> SDK: Docker -> Public.
2. Add this block at the very top of README.md in the Space:

       ---
       title: Student Performance Predictor
       sdk: docker
       app_port: 7860
       ---

3. Upload all files of this folder (including data/student-mat.csv) to the Space.
   The included Dockerfile builds and starts the app automatically.
4. Your app appears at https://huggingface.co/spaces/<your-name>/<space-name>

---------------------------------------------------------------
## Option C - Run on your own computer and share on your Wi-Fi

    set HOST=0.0.0.0        (Windows PowerShell:  $env:HOST="0.0.0.0")
    python app.py

Others on the same network open http://<your-computer-IP>:5000
(The app only works while your computer is on.)

---------------------------------------------------------------
## Good to know

- READ_ONLY=1 (already set in render.yaml and the Dockerfile) hides Upload/Reset so visitors
  cannot replace your dataset. Remove it only for private use.
- The trained models live in memory and are shared by all visitors. If one visitor clicks
  "Train & evaluate" with different options, the live page changes for everyone. That is fine for a
  demo or class project, but a multi-user product would need per-user sessions.
- Free hosts have limited memory and time. Training takes a few seconds on startup.
