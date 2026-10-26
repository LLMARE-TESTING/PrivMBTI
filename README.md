# Privacy Persona

The repository of paper *"PrivMBTI: Integrating Compact Types with Rich Personas for LLM-Based Privacy Simulation"*.  

It includes privacy behavior simulation prompting and evaluation scripts, four experiment tasks, and a locally hosted PrivMBTI test website.

## Repository Structure

```text
.
├── PrivMBTI_web/                      # Local Privacy MBTI test application
│   ├── index.html                     # Single-page UI: participant details, test, questionnaire, and results
│   ├── server.js                      # Node.js server and result-saving API
│   ├── questions_35.json              # Main 35-question persona test
│   ├── questions_20.json              # Alternative 20-question test set
│   ├── user_questionnaire/            # Follow-up questionnaires assigned by participant number
│   ├── 4-dimensional_personality.py   # Calculates four-dimensional persona scores and traits
│   ├── aggregate_persona_responses.py # Aggregates persona response data
│   ├── enrich-answers.js              # Helper script for enriching answer data
│   ├── data/                          # Locally generated participant results
│   └── index_files/                   # Static assets used by the web page
├── scripts/                           # Prompt variants and evaluation scripts
│   ├── prompt_M0.py ... prompt_M4_M5.py  # Simulation scripts
│   └── evaluator.py                   # Evaluation scripts
├── tasks/                             # Four Experiment tasks
│   ├── assessment_task.json
│   ├── contextual_task.json
│   ├── knowledge_task.json
│   └── setting_task.json
└── README.md
```



## Open the PrivMBTI Test Website

### Prerequisites

- Node.js, used to run the local web server
- Python 3, used to calculate persona scores
- Python packages: `numpy` and `pandas`

If the Python dependencies are not installed, run this command from the repository root:

```powershell
python -m pip install numpy pandas
```

### Start the Website

Open PowerShell in the repository root and run:

```powershell
cd PrivMBTI_web
node server.js
```

The service is ready when the terminal displays:

```text
Privacy MBTI local server running at http://localhost:3000
```

Open the following address in a browser:

```text
http://localhost:3000
```

Enter a participant number from **1 to 53** and a name to start the test. The website automatically loads the corresponding follow-up questionnaire and saves submitted results under `PrivMBTI_web/data/`.

### Troubleshooting

- **Do not open `index.html` by double-clicking it.** Start the application with `node server.js` and visit `http://localhost:3000` so that question JSON files, questionnaires, and the result-saving API work correctly.
- **Port already in use:** Start the server on a different port, for example: `$env:PORT=3001; node server.js`. Then visit `http://localhost:3001`.
- **No persona result is generated:** Confirm that `python --version` works and that `numpy` and `pandas` are installed.


