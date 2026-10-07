# Student Performance Predictor (web app)

Predicts a student's final grade **G3 (0-20)** from study time, absences, family background and more.
Models: **Linear Regression, Decision Tree Regressor, Random Forest Regressor**.
Covers: categorical encoding (one-hot), regression evaluation (MAE, RMSE, R2, 5-fold CV), feature importance.

## Quick start

Requires Python 3.9+.

**Windows:** double-click `run.bat`
**Mac / Linux:** `./run.sh`

Or manually:

    pip install -r requirements.txt
    python app.py

Then open **http://127.0.0.1:5000** in your browser.

## Use the REAL UCI dataset

The app starts with a built-in *synthetic* sample (same columns as UCI) so it works immediately.
For your project, use the real data - any one of these:

1. Run `python download_dataset.py` (needs internet) and restart the app, **or**
2. Download from https://archive.ics.uci.edu/dataset/320/student+performance, put `student-mat.csv`
   in the `data/` folder and restart, **or**
3. Use the **Upload CSV** button in the web page.

The page shows a "Real data" / "Sample data" badge so you always know which you are using.

## Using the app

1. **Dataset** - see the source, grade distribution and a preview; upload your own CSV.
2. **Train models** - choose whether to include G1/G2, exclude dropouts (G3 = 0), and the test split.
3. **Feature importance** - per-model bars and an actual-vs-predicted chart.
4. **Predict** - fill in a student's details and get all three models' predictions.

### Tip for your report
G1 and G2 (earlier grades) almost determine G3, so R2 is high when they are included. Run once with and
once without them and compare - without them you see the effect of study time, absences, failures, etc.

## Files

    app.py               Flask backend (training, evaluation, API)
    templates/index.html Web UI (no external libraries needed)
    sample_data.py       Synthetic fallback dataset
    download_dataset.py  Fetches the real UCI data
    requirements.txt     flask, pandas, numpy, scikit-learn
    run.bat / run.sh     One-click launchers

To put the app online so it runs anywhere, see **DEPLOY.md**.
