# Evaluation

For simplcity, the scoring metric used for the phase 1 tasks is the [accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).
Although for future phases, the scoring metric for each task can vary across tasks. 

Overall, at the end of the competition, the scores will be aggregated using Performance Profiles [1, 2], and the a winner will be chosen based on the highest area under their performance profile curve. 
In other words, the winner will be determined using the Area Under the Performance Profile (AUP) score. 
First, a lower-is-better performance metric (LPM) is computed for a given method on each competition task. 
Next, we compute performance profile curves [1, 2] for each method across all the tasks---namely, we plot ρs(τ) for each method s∈S on all tasks P. 
Since we must obtain a final ranking, we compute the area under the curve, ρs(τ), for each method s∈S. 
The Area Under the Performance Profile (AUP) ρs(τ) will be used as the metric to determine the leaderboard and final winners. 

![AUP Score](https://raw.githubusercontent.com/nick11roberts/nick11roberts.github.io/master/img/aup.png "AUP Score")


## References

[1] http://www.argmin.net/2018/03/26/performance-profiles/

[2] Dolan, E., Moré, J. Benchmarking optimization software with performance profiles. Math. Program. 91, 201–213 (2002). https://doi.org/10.1007/s101070100263
