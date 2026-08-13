# AI Notes - Project B

## How I Used AI

I used AI throughout Project B as a coding, checking and review assistant,
not to replace my own intepretations of the data that was produced in the code.
The main uses were the structure of the report, I was not sure which part should come
first, so I used AI to help me structure my report, help me to get everything done in sections
before I move on to the next section. This helps me to keep my work organised and have a better
flow of ideas. The assistant also helped me to build the Streamlit app and the coding part of the
project, and to review the final submission order. Whenever I encounter financial terms I am not sure 
of, I would first Google and if I still cant understand, I asked the assistant to provide an example 
of it, terms such as lagging, backtest, common sample.

I also directed the work in stages. First, I prompted the assistant to start with the main core of the project,
building the out of sample backtest and making sure that the fund returns, weights and performance metrics were 
generated correctly. After that, I added the transaction cost model, the VADER sentiment index, the sentiment fusion test
the finance lexicon innovation and lastly to build the Streamlit app features. This helped me to acoid
mixing too many tasks together, and made it easier to verify each components first before moving on.

## How I Checked AI Output

I checked AI outputs against the project brief, such as when it says I was missing a 
latest rebalance date and individual fund fact,  I had to manually check it to see if it was
legitimate. For the backtest, I checked the portfolio weights were formed only
from past returns and that the first live backtest dates made sense after the 252 days estimation window.

For the Streamlit app, I checked that the figures that was displayed on the app was the same as the results
that was in Pycharm. The app also used pictures and numbers from the results generated in Pycharm, 
I needed to cross check everything to make sure it reflects the correct values and figures.

One issue that I caught was the Sharpe ratio calculation, The first version of the output did
not match the standard daily return Sharpe approach, it used the cumulative returns as the returns
used in Sharpe ratios rather than the average returns. So i checked the calculation
and changed it to fit the conventional Sharpe ratio.

Another issue was the finance lexicon extension. Ai initially helped add the finance adjusted sentiment
signal, but I made sure to not overfit the result. And that made the results only slightly
better than the plain VADER tilt, which was not the result that I wanted but then 
I present it as robustness test rather than it improves the fund materially.

I made the final decisions for the intepretation of the results, report structure, and what to include
in the final submission. Sometimes the assistant wants to include in figures that is already covered in the table
inside the report, this will make the appendix messy to read, I also decided not to present sentiment or finance lexicon as 
performance strategies since the results did not support.

Overall AI aided me most by helping me break a large project into smaller, manageable stages.
Instead of trying to build everything at once, I used AI to plan the order of work and then checked the output
to match what I want before I continue. This made the project less overwhelming. However, I learned that AI 
output cannot be trusted without checking, I still had to compare the results against the brief, csv outputs
and see if the calculations make sense. The biggest benefit was that the AI helped me to understand problems 
in a much simpler way, able to explain the results while I am stil the one that is in charged of what to put in my report.



