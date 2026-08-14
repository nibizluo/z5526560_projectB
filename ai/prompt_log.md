# Prompt log - Inspecting and planning Project B

## What I wanted
I wanted the AI assistant to inspect the Project B folder before
making any code changes. I was trying to understand what the assigment
requires, what the starter code already has, what outputs should be created,
and what is the best implementation method. I also wanted the Ai to check 
if my idea, MarketBridge for Project A was feasible.

## Prompt(s)
You are helping me with FINS3645 Project B.
Before editing any files, please inspect the project
folder and read the main files:
- PROJECT_BRIEF.md
- README.md
- SUBMISSION_CHECKLIST.md
- scripts/run_part_b.py
- src/data_access.py
- src/portfolios.py
- src/sentiment.py
- src/fusion.py
- streamlit_app.py

Do not edit any files yet.
Please explain Project B in simple terms:
- what the assignment requires
- what outputs must be created
- what the starter code already does
- what is missing
- what files needs to be edited
- what order should i implement my idea in
- what mistakes i need to avoid, or anything that might risk me getting a bad grade

My app is called MarketBridge, a continuation from Part A.
My preferred Part B direction is:
- use minimum variance strategy with a crypto exposure cap since found out that crypto was very volatile from Part A.
- add a lagged sector sentiment tilt using headline data
- add a transaction cost transparency test to compare gross and net performance of the stocks
- use the streamlit app to show fund performance, risk, sentiment and transaction cost 

Important to take note:
- do not create look-ahead bias
- use only past data when forming portfolio weights
- lag sentiment before using it
- the streamlit app should read results from the results/ folder
- the app should not recompute backtest or run sentiment scoring live

Again, do not edit the files yet. Just inspect and explain.

## What the assistant produced
The assistant explained that Project B requires systematic fund construction,
that uses market return data, out of sample backtests with no look ahead bias,
a news sentiment index from headline data, a fusion strategy that uses sentiment in 
the equity portfolio, a streamlit app that shows fund performance, risk, sentiment, 
an investor facing factsheets.

It explained that the current part B script is a structure which is expected, 
The assistant suggested an implementation order: first build the clean return panels and basic fund backtests, 
then add guarded minimum variance, transaction costs, sentiment scoring, sentiment tilt, 
figures, the app, and finally the report.

## What was wrong or risky
The response was useful, but it suggested many possible extensions at once, including
maximum Sharpe, transaction costs, sentiment tilt, app features and reporting. The main risk is
trying to implement too many things at once, too many innovations before making sure that the
core of the backtest works, if the base funds returns and weights are inaccurate, the sentiment and transaction
cost would also be unreliable. 
There is also a risk that sentiment and transaction cost ideas could create look-ahead bias if
they are implemented incorrectly. 

## What I changed and why
I decided to implement Project B in stages instead of asking Ai to build everything at once
The first implementation step will focus only on the core portfolio backtest: combined equal weight
combined maximum sharpe, and guarded minimum variance with a 20% crypto cap. Sentiment, transaction cost
fusion, and app deployment will be added later after basic fund_returns, fund_weights, and performance_metrics
outputs are working. 
This will help with any hidden errors and makes it easy to verify before building the next layer.

---

## Implementing first stage of Project B

### What I wanted
A function to compute simple daily returns per ticker from the equity panel.

### Prompt(s)
"Please implement the first stage of FINS3645 Project B, basic fund construction
and walk-forward backtest.

Work only inside z5526560_projectB

Do not implement VADER, fusion, transaction costs or streamlit app changes yet
. This is only for the core portfolio backtest.

My app is called MarketBridge.

Goal:
Create a reproducible Part B pipeline that builds equity, crypto and combined return panels, then
backtests some systematic funds out-of sample using only past data.

Implement these fund methods:
- Combined equal weight (Long only, equal weight across all included assets.
- Combined Minimum variance (Long only, weights chosen to minimise portfolio risk using historical covariance matrix, weights sum to 1, no short selling)
- Combined maximum Sharpe (Long only, use historical mean returns and covariance matrix, use a simple risk free rate)
- Guarded minimum variance (Based on combined minimum variance, total crypto allocated cap to 20%, Long only,)

For each fund calculate:
- cumulative return
- annualised return
- annualised volatility
- sharpe ratio
- maximum drawdown
- average weight in equities
- average weight in crypto
- number of rebalance dates

Important:
- keep the code simple and explainable.
- sdd comments for the optimisation and walk-forward logic.
- if an optimisation fails, fall back to equal weight and record/warn clearly.
- do not create look-ahead bias.
- do not add sentiment or transaction costs yet.

After editing, summarise:

- which files changed
- what functions were added
- what outputs were created
- what assumptions were used
- how look-ahead bias was avoided
- how the 20% crypto cap was implemented
- what command I should run next
"
### What the assistant produced
The assistant implemented the core backtest stage and edited the src/features.py, src/portfolios.py, and scripts/run_part_b.py
It added return construction from adjusted closed price, equity, crypto and combined return panels, long-only equal weights, 
minimum variance, and maximum sharpe methods.

The assistant also produced the growth, drawdowns  and Sharpe comparison figures.
The run produced 10 funds across equity-only, crypto-only, combined, and guarded combined variants. 
The smoke test also passed, confirming that imports and data loading were working.

### What was wrong or risky
The main risk was look ahead bias. If the AI had estimated portfolio weights using the full dataset or
included the current live return row in the estimation window, the backtest would be unrealistic. I checked the 
summary and confirmed that weights were formed using only the previous 252 return observations at each rebalance date.
I also crosschecked with the AI to see if it had any look ahead bias.

The Guarded minimum variance fund included a 20% crypto cap, but the generated
results showed that the cap was not binding in this sample. The fund's maximum crypto allocation 
was about 2.9%, and average crypto weight about 0.6%, this means the cap didnt work as intended.

Later formula review found another issue: the reported Sharpe ratio in
performance_metrics.csv was calculated as CAGR over annualised volatility.
That is a return-to-risk ratio, but it is not the standard Sharpe ratio in
the project brief. The brief allows a zero risk-free-rate assumption, so the
better formula is sqrt(periods_per_year) times mean daily return over
daily return standard deviation. The optimiser was already using mean return
over volatility for the maximum-Sharpe objective, but the reported Sharpe metric
used a different formula.

### What I changed and why
I kept the core backtest implementation because the required files were created successfully and the smoke test passed.
However, I reviewed the outputs before moving to the next stage and decided to treat the results carefully in the report.

After checking the brief again, I changed the reported Sharpe calculation in
src/portfolios.py to use the standard zero-risk-free Sharpe formula:
sqrt(periods_per_year) * mean(daily_return) / std(daily_return). I kept the
annualised return as the compounded annual growth rate because that is a separate
metric. I asked the assistant to double check my logic if it fits the project brief.

I decided to continue the implementation in stages. The next step will be the transaction-cost transparency test, 
before adding sentiment and streamlit features. This is because transaction costs use the existing fund returns and weights, 
so it is a robustness check after the core backtest.

I was confused as for why the figures for growth of 1 started at 2020-09 as previously
all the figures were created from 2020-01, but after asking AI i realised that it is
due to backtesting, so if i were to look at the first 252 days of the data, the earliest was
2020-01-01 plus 252 days which was around 2020-09. 


## Implementing the second stage of MarketBridge- transaction cost transparency test

### What I wanted
I wanted to proceed on with the next stage of the project, implementing my innnovation 
which was to add a transparency feature of the app, which is the transaction costs.
I assumed that the transaction costs will be $0.001 per dollar.

### Prompts
Please implement the next Project B innovation -  a transaction cost transparency test

Do not implement VADER, fusion or streamlit deploy yet. This step is only 
for turnover and transaction cost analysis.

Goal:
Add a transaction cost transparency test that compares
gross fund performance with net performance after accounting for 
estimated trading costs. this is supposed to be an innovation as it allows
investors to see how trading costs affects realised returns.

Assume a simple transaction cost rate of $0.001 per 1 dollar traded.
Apply transaction costs only on rebalance dates.

The transaction cost metrics table should include:
- fund_name
- gross_cumulative_return
- net_cumulative_return
- gross_sharpe_ratio
- net_sharpe_ratio
- gross_max_drawdown
- net_max_drawdown
- average_turnover
- total turnover
- transaction_cost_rate

Please keep this as a robustness and product transparency feature, not as 
a new trading strategy.
Do not claim that 0.10% is the real world cost, it is assumed for simplicity sake only.

After editing, summarise:

which files changed
what new outputs were created
what transaction-cost assumption was used
how turnover was calculated
which funds were most affected by transaction costs
whether common-sample figures/metrics were created
what command I should run next

### What the assistant produced
The assistant implemented the transaction cost transparency test inside the folder.
It also produced a funds_returns.csv in data folder which compares gross fund returns and 
net fund returns.

The results showed that the funds most affected by transaction costs is Crypto Minimum variance,
combined maximum sharpe and equity maximum sharpe.
This makes sense as these strategies has higher turnover which means more costs.

### What was wrong/risky
The transaction logic worked, but the first gross vs net graph was not very useful for the report.
It plotted too many funds on one figure, and the gross and net lines were very close together, so the 
transaction cost effect was hard to see. The crypto funds also has highest volatility, which made the 
equity and combined funds looked almost flat.
It was more of a presentation issue, where the assistant tried to cramp everything into 
one figure, but it was too crowded to get a meaningful value out of it.

### What i changed and why
I kept the transaction cost calculation, but I decided not to use the figure as it was difficult to interpret.
Instead I asked the assistant to create more report friendly figures, such as bar plot showing estimated
transaction cost drag. These figures make the main point clearer, which was higher turnover are more exposed to 
transaction costs. This change improves the clarity of the figures without changing the main point.

## Implementing the third stage: sector sentiment index and lagged sentiment fusion
### What I want
I wanted the assistant to implement the text and sentiment component of the Project in one stage
The goal was to create a sector level sentiment index from the headline data and test if a lagged sentiment
signal could improve existing guarded minimum variance fund.

### Prompts
Please implement the next stage: sector sentiment index and lagged sentiment fusion strategy

Goal: 
Add the text/ sentiment component required for Project B and connect it to the existing MarketBridge fund analysis.

This stage should have:
- a daily sector sentiment index using headline data
- a lagged sentiment signal that avoids look-ahead bias
- a sentiment-tilted version of the guarded ,minimum variance fund
- a before-versus-after comparison between:
  - guarded minimum variance
  - guarded minimum variance + sentiment tilt

Things to take note:
- do not use same day sentiment to trade on the same day
- sentiment used for portfolio weights must be lagged
- after each rebalance date, use only sentiment information available before this date
- do not recompute sentiment index within streamlit
- keep the code simple and explainable

This stage should consists of three parts:
- Sector sentiment index- output should include columns like dates, sector, headline_count, mean_compound, sentiment_index. Figures should have a sector sentiment index.
- Lagged sentiment signal- create a lagged sentiment signal for portfolio use. Make sure that no same day or future sentiment is used when forming weights.
- Sentiment tilted guarded minimum variance fund- this should start from the existing guarded minimum variance logic and apply a small sector-level sentiment tilt.

After editing, summarise:

which files changed
what new outputs were created
what sentiment VADER was used
what command I should run next

### What the assistant produced
The assistant implemented the sentiment and fusion part inside ProjectB folder, it created
a sector sentiment index using the headline data and VADER sentiment score.
It also created a lagged sentiment signal, so the portfolio only uses dates before the rebalance dates,it produced the necessary 
sentiment and fusion outputs.

### What was risky/wrong
The main risk was look ahead bias. If the assistant used the same day sentiment to make same day portfolio changes, the backtest
would not be realistic. To avoid this, the sentiment was lagged at least a day before this.

The fusion growth figure was also not very useful because the two lines are once again very closely aligned like the previous prompt.This means
that the sentiment did not have enough influence on the returns.

### What I changed and why
I kept the sentiment and fusion implementation because the outputs were created successfully and the checks passed. 
However,  changed how I planned to explain the result.

Instead of saying that sentiment improved the fund, I will explain it as a test of whether MarketBridge can combine 
headline sentiment with portfolio construction. The result shows that the method worked technically, but the simple 
lagged sentiment tilt did not improve performance in this sample

For the report, I will focus more on the sentiment index and fusion metrics comparison instead of 
relying too much on the growth chart, because the growth chart is hard to interpret

## Streamlit deployment stage
## What I want
I wanted the assistant to build the streamlit app, to compare systematic funds, based on performance,risk,
crypto exposure, transaction cost impact, sector sentiment and fusion comparison.
### Prompts
Please implement the streamlit app for Project B for MarketBridge.

Do not change the backtest, sentiment, transaction-cost, or fusion logic unless there is a clear bug
The streamlit app should read precomputed CSV files and figures from the results folder only
It should NOT recompute portfolio optimisation, VADER sentiment, transaction costs, or backtests live

The app should help a retail investor compare systematic funds using:
- Performance
- Risk
- Crypto exposure
- Transaction cost impact
- Sector sentiment
- Fusion comparison

Page title: MarketBridge: Fund Performance, Risk and News Sentiment Dashboard

1. Overview: Briefly explains that MarketBridge compares systematic equity, crypto and combined funds. Explain where the results come from.
2. Fund performance: Show performance metric from common sample, display key metrics such as cumulative return, annualised return, sharpe ratio and max drawdown. Show graphs for $1 growth and drawdowns
3. Risk and allocation: Show fund weights summary, average equity and average crypto weight. Highlight that guarded minimum variance has a 20% cap, and explain the reason why.
4. Transaction cost transparency: Show transaction cost metrics, turnover by funds, explain that the transaction costs is based on an assumed rate of 0.01% per dollar traded, not real world.
5. Sector sentiment: Show sector sentiment index, explain that VADER sentiment was aggregated by sector and date.
6. Sentiment fusion: Show fusion comparison, fusion metrics comparison, clearly explain that the lagged sentiment did not help improve as much performance as expected.

Keep the app simple and professional.
Do not make it too crowded.
Do not crash if figures are missing.
Format percentages nicely, nicer to read.

After editing, summarise:

which files changed
what new outputs were created
what command I should run next

### What the assistant produced
The assistant implemented and deployed the streamlit app for MarketBridge. The app reads the precomputed results
from the results folder and display the main output,including fund performance, growth of $1, drawdowns, transaction
cost metrics, and fusion comparison.
The app also included sections for comparing results, viewing risk metrics, checking transaction cost impact, and explaining
the sentiment fusion result. It did not recompute the VADER sentiment, portfolio optimisation, or transaction cost calculations live which was good.
The app seems functional

### What was wrong/risky
The app was working, but it is missing out on two important features that the project brief requirements. 
First the app did not let users set their own allocation across the funds. For example, the user could not build a custom
portfolio such as 40% combined maximum sharpe, 40% guarded minimum variance, 20% crypto minimum variance. 

Second, the app did not include a portfolio weights over time figure. It only showed average equity and crypto weights over time.
The figure of Daily sector headline sentiment index for 21 days was way too messy, there are 6 lines intercrossing with one another which was hard to see the trend.

### What I changed and why
I needed to fix the two missing parts before starting on my report. I asked the assistant to add a
custom fund allocation tool and a weights over time figure with the existing fund_returns.csv and fund_weights.csv.
I also asked the assistant instead of cramming 6 different sectors into the sentiment index, separate the six of them
into separate graphs so that the trend is much more observable.

## Finance lexicon sentiment extension
### What I want
I wanted to create a new finance lexicon based on words that the VADER lexicon missed out to see if the new lexicon had a
positive impact on the portfolio returns.
### Prompt
Please implement the innovation extension: finance lexicon sentiment extension

Goal:
Extend the existing sentiment analysis by adding a small finance specific lexicon.
The current sentiment index uses VADER, but some finance headline words may
not be captured by VADER. This stage aims to identify common finance related words
in the headlines that are missing from VADER, assign finance sentiment scores to the
selected terms, recalculate the sentiment index, and compare the new finance adjusted 
sentiment index with the original VADER sentiment index.

Important:
Do no replace the original VADER sentiment outputs. Keep the original VADER
sentiment index as the baseline, and create new finance adjusted outputs separetely.

This stage should have:
- a table of common headline terms that are not in the VADER lexicon
- a small custom finance lexicon with assigned sentiment score
- a finance adjusted sector sentiment index
- a comparison between the original VADER sentiment index and the finance adjusted sentiment index.
- a finance adjusted sentiment tilt fund to compare with the existing sentiment tilt fund

Things to take note:
- keep the lexicon small and explainable
- use values between -1 and +1
- do not overfit the lexicon just to improve performance
- do not claim the finance lexicon is objectively correct
- preserve the existing VADER sentiment files
- make sure any sentiment signal used for portfolio weights is lagged
- do not use same day or future sentiment when forming weights
- do not recompute sentiment inside Streamlit
- keep the code simple and explainable

This stage should consist of four parts:
1. Missing finance terms - Filter headline words that are not in the VADER lexicon.
Rank them by frequency and save a candidate terms table. Remove obvious stopwords, tickers,
company names and irrelevant words.
2. Custom finance lexicon - Create a small finance specific lexicon from selected candidate words.
Assign scores between -1 and +1. Positive words may include upgrade, beat, growth. Negative words may include
downgrade, slump, loss.
3. Finance adjusted sentiment index - Recalculate headline sentiment by combining VADER compound scores
with the custom finance lexicon adjustment. Then aggregate the finance adjusted scores by sector and date.
4. Compare old and new sentiment - Compare the original VADER sector sentiment index with the new finance
adjusted sentiment index

Create one extra fund: Guarded Minimum Variance wih Finance lexicon sentiment tilt
Compare it with Guarded Minimum Variance, Guarded Minimum variance with sentiment tilt.

After editing, summarise:

which files changed
what new outputs were created
how missing words were identified
how finance sentiment scores were assigned
how the finance-adjusted sentiment index differs from VADER
whether the finance-adjusted sentiment improved or worsened performance, if implemented
whether look-ahead bias was avoided
what command I should run next

### What the assistant produced
The assistant implemented a finance lexicon sentiment extension inside the Project B folder.
It kept the original VADER sentiment index as the baseline and created new finance adjusted sentiment outputs separately.

It first identifies headline word that were not already in the VADER lexicon, then created 
a small custom finance lexicon with 48 selected terms. These terms were given sentiment scores between -1 and +1. 
The assistant then recalculated the sector sentiment index using a finance adjusted score, which combined the
original VADER compound score with a small finance word adjustment.

The assistant also compared the original VADER sentiment index with the new finance adjusted sentiment index. The two sentiment
indexes were very similar with a correlation of 0.9959. It also created a new optional fund called Guarded Minimum Variance + Finance lexicon
sentiment tilt to test if the finance sentiment signal improved the fusion result.

### What was wrong/risky
The main risk was overfitting the finance lexicon to improve the backtest result. If the lexicon was designed only to make the
fund perform better, then the result would not be reliable. To avoid this, the custom lexicon used conservative scores and only 
included terms that appeared in the headline data.

Another risk was that the finance sentiment scores were partly subjective. Some words in financial headlines can be ambiguos.
For example, a word may sound negative in normal situations but may have a different meaning in a financial terms. This means that
the custom lexicon should be treated as an experimental extension and not an objectively correct sentiment model.

### What I changed and why
I kept the finance lexicon extension although it didnt show alot of improvement but it worked as an innovation and robustness test. 
I wouldnt present it as a performance improvement, rather if it helped with the robustness of the sentiment model. If adding the finance
specific vocabulary changes the sentiment signal compared with plain VADER.

## Adding finance lexicon extension to the Streamlit app
### What I want
I wanted the Streamlit app to show the finance lexicon extension as part of the
investor dashboard, because the finance lexicon had already been implemented in
the results outputs but was not shown at all in the app.

### Prompt
Please add the finance lexicon sentiment extension to the MarketBridge Streamlit app.

Goal:
The finance lexicon extension has already been created in the results outputs,
but it is not  shown in the Streamlit app. Add a simple Finance Lexicon
tab so investors can see the extension as a robustness and innovation feature.

The new app section should show:
- the custom finance lexicon terms and conservative scores
- the VADER versus finance-adjusted sentiment comparison by sector
- the finance lexicon fusion comparison
- any existing finance lexicon figures if available

Do not change the backtest, sentiment calculation, transaction-cost logic, fusion
logic, or generated results.
Do not recompute VADER, finance-adjusted sentiment, portfolio optimisation,
transaction costs, or backtests inside Streamlit.
The app should read precomputed files from the results folder only.

Important:
- Add the finance lexicon as a separate extension tab, so it doesnt confuse the
  baseline results.
- Do not make the app crowded.
- Do not crash if finance lexicon files or figures are missing.
- Format tables and percentages cleanly.
- Clearly state that the finance lexicon tilt did not outperform the original
  Guarded Minimum Variance fund. Hence, this should not be treated as a fund improvement strategy.

After editing:
- summarise which files changed
- summarise what was added to the app
- confirm whether the app still reads precomputed results only
- confirm whether any baseline output or backtest logic was changed
- tell me what command I should run next


### What the assistant produced
The assistant checked the Streamlit app and confirmed that the finance lexicon
outputs existed in the results folder, but the app was only showing the baseline
VADER sentiment tab and the normal sentiment fusion comparison.

The assistant then added a separate Finance Lexicon tab to the Streamlit app.
This tab reads only precomputed files from the results folder. It displays the
custom finance lexicon terms, the comparison between the original VADER sentiment
index and the finance-adjusted sentiment index, and the finance lexicon fusion
comparison. It also has the relevant figures and table including the correlation between
old and new sentiment tilt.

### What was wrong/risky
The main risk was accidentally making the app recompute sentiment scores or
portfolio backtests live. 

Another risk was making the normal sentiment fusion section confusing by mixing
the finance lexicon extension into the required VADER fusion comparison. To avoid
this, the app keeps the normal Sentiment Fusion tab separate and uses the new
Finance Lexicon tab only for the extension. The distinction is to keep it separated
and clear that the Finance Lexicon is different from the original VADER lexicon.

### What I changed and why
I kept the finance lexicon extension as a separate app tab so it is clear that it
is an innovation and robustness test, not a replacement for the baseline VADER
sentiment model.

## Final report and app review
### What I wanted
I asked AI to review my Part B draft with the project brief and the generated code 
outputs. Make sure that there is no live optimising and satisfy the project requirement.
I wanted it to check if there is any major content gaps or structural gaps

### Prompt(s)
Please inspect the Project B brief thoroughly, then read through my latest
3645 Part B draft together with the code, generated outputs, app files and
verify against project brief

Focus on:
- whether the report meets the Project B brief
- whether every required exhibit is included and interpreted
- whether the sentiment signal is lagged correctly
- whether the app recomputes anything live or only reads precomputed results
- whether the fund fact sheet requirement is properly covered
- whether current holdings/latest target weights are included
- whether table and figure numbering is consistent
- whether the AI workflow pack and prompt logs are sufficient
- whether there are any remaining structural, coding, logic or submission issues

You can ignore final deployment, GitHub upload and zipping for now. Focus more
on the content, structure, reproducibility and logic of the project.

Please do not rewrite the whole project. Review it carefully and tell me what
is still wrong, what has been fixed, and what I should prioritise before final
hand-in.

### What the assistant produced
The assistant reviewed the Project B brief, the report draft, the source code,
Streamlit app, generated result tables, figures, and submission files.

The assistant identified several content and structure issues in the
report. The first major issue was that the report initially showed average
equity and crypto weights but did not clearly show current holdings or latest
target weights from the most recent rebalance, even though the project brief
explicitly mentioned to include it. The second issue was that the report did not 
clearly explain how sector-days with no headlines were treated before the sentiment
signal was lagged, the project brief also stated to include what happened to missing headlines
should be mentioned in the report. The assistant also pointed out that some table numbering became 
inconsistent after I added the latest target holdings table

### What was wrong/risky
The biggest risk was that the report could appear to meet the performance-table
requirement but still miss the fund fact-sheet requirement because average
weights are not the same as current holdings. This matters because the brief
defines current holdings as the target weights from the most recent rebalance.
That was what happened in my draft report, I mentioned the allocation of cryto
and equity to each funds, but that was the average weights rather than the final 
weight at the last rebalance date.

There was also a risk that the Streamlit app might recompute analysis live. I 
asked the AI to double check if there is any live computation going on, the only
thing that was close to it was when the user allocate custom weights and the risks
and returns adjusted accordingly. The assistant confirmed that there was no live optimisation.

### What I changed and why
I added a latest target holdings table which includes the latest rebalance date and latest 
equity and crypto weights. This fixed the current holdings gap in the fund fact-sheet discussion.

I added a sentence in the sentiment methodology explaining that sector-days with
no headlines were treated as neutral before the sentiment signal was lagged and
smoothed.

