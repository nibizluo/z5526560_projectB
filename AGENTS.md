# AGENTS.md

## Codex Instructions

This is my Project B workspace:


projects/z5526560_projectB

work inside projecs/z5526560_projectB

Project B is the FINS3645 MarketBridge Funds, Sentiment and App Project. It
covers Stations 3 and 4 of the Data Factory Floor. The work uses the clean
equity, crypto and news data prepared in Project A to build out-of-sample
investment funds, news-sentiment analytics and a deployed Streamlit investment
app.

Before making major changes, read the assignment and project context first. Do
not assume missing requirements or results. If a file, dataset or output cannot
be found, flag it out in your response in the AI chat box.

## Read First

Before helping with Project B, read:

- PROJECT_BRIEF.md
- README.md
- SUBMISSION_CHECKLIST.md
- files in context/ to understand file contexts, project contexts and also how to verify AI outputs
- relevant outputs from Project A
- existing files in src/
- existing files in scripts/
- scripts/check_handin.py

Do not begin large implementation work until the assignment requirements,
existing code and available Project A outputs are understood.

## Project Scope

Project B covers:

- Station 3: portfolio funds, out-of-sample backtesting, sentiment analysis and
  sentiment fusion
- Station 4: the Streamlit investment app and product implementation
- the Part B report
- deployment and final submission preparation

The required minimum for this project is a combined equity and crypto fund using at least two
optimisation methods.

The project may also include equity-only funds, crypto-only funds, extra
optimisation methods, transaction-cost analysis, original sentiment extensions
and useful app features. Only add extensions that improve the analysis or app
and can be evaluated with evidence.

## Project Boundaries

Keep Project B work inside this folder.

Do not:

- edit files outside this project B folder
- rerun or duplicate the full project A cleaning pipeline, if there is a need, flag it out to me first before proceeding
- use future information in portfolio weights, backtests or sentiment signals
- invent missing results, citations, statistics or sources
- add features that are unrelated to the assignment
- submit AI-generated interpretation without checking and rewriting it in my own
  words, if I were to submit, tell me before hand if I submit anything that is not in my own words.


## Working Locations

Use these folders consistently:

- results/data/ for app-readable derived datasets
- results/tables/ for report tables
- results/figures/ for figures
- scripts/for runnable scripts
- src/ for reusable helper functions
- ai/ for prompt logs and AI-use notes
- report/ for report files

Keep the workflow reproducible. Main analysis should run through scripts and
helper functions. The Streamlit app should load precomputed outputs instead of
rerunning heavy backtests or sentiment analysis. NEVER compute anything live.

Use relative project paths where possible. Do not rename required outputs without
checking PROJECT_BRIEF.md.

## Required Outputs

Confirm all requirements against the brief, including these exact filenames:

- results/data/fund_returns.csv
- results/data/fund_weights.csv
- results/data/sector_sentiment_index.csv
- results/tables/performance_metrics.csv
- report/report.pdf

Additional outputs can be created if they have clear, descriptive names.

Every figure and table should be self-contained, readable and suitable as
evidence for the report. Include all the axis labelling, units and title of the graphs.

## AI Workflow

AI may help with:

- planning
- coding
- debugging
- explaining errors
- checking calculations
- producing workings
- preparing charts and tables
- reviewing results
- drafting report sections
- editing writing
- testing the project
- preparing the repository and submission

Before major changes, explain the plan first. The plan should state:

- which files will be changed
- what will be implemented
- what assumptions will be used
- what outputs should be created
- how the work will be tested

Do not make large unexplained edits.

After editing files, summarise:

- which files changed
- what each changed file does
- what outputs should be created
- what command I should run next
- what I should manually check

## Accuracy Rules

The AI must:

- never invent a citation, statistic, result or source
- flag any claim it cannot verify instead of stating it confidently
- distinguish verified results from assumptions or estimates
- show the formula, inputs and working for important numbers
- identify which project file a reported result came from
- never claim that code, tests or the app passed unless they were actually run
  and inspected
- remind me to check AI outputs before using them in the report or submission

If information cannot be verified from the project files, say so clearly instead
of guessing.

My final analysis and interpretation of report must be checked and written in my
own words.

## Testing And Verification

After major changes:

- run the affected script
- inspect errors and warnings
- confirm the expected outputs were created
- inspect important CSV columns and sample rows
- check dates, missing values and numerical values
- check for look-ahead bias
- inspect figures visually
- confirm unrelated files were not changed unexpectedly
- run it with test_smoke and check_handin for every edits you done and make sure that no errors are present

After changing the Streamlit app:

- start the app locally
- confirm it loads
- test all pages, tabs and interactive features
- confirm required outputs load correctly
- confirm the app remains light and uses precomputed results

Do not state that something works unless it has been tested.

## Report Rules

The report should focus on:

- funds and backtest design
- out-of-sample fund performance
- fund fact sheets
- the sector sentiment index
- sentiment fusion
- extensions and innovations
- the Streamlit investor journey
- limitations and critical reflection
- three concrete recommendations

Every important claim must be supported by a project result, table or figure.

Check every statistic against the relevant output file before including it.
Check every citation against the original source.

Keep the editable report files in report/. The submitted PDF must be:



Before submission:

- fix every genuine [FAIL] from the hand-in checker
- complete README.md
- complete SUBMISSION_CHECKLIST.md
- confirm the report PDF opens correctly
- confirm required results are committed
- confirm raw data and secrets are excluded
- confirm the Streamlit app is live
- confirm the GitHub repository is public at hand-in
- preserve the required folder structure
- create the final Moodle zip

The final deployment step requires my own GitHub and Streamlit login.
