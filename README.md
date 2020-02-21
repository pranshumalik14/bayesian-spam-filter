# bayesian-spam-filter

For every word, ![w_i](eqns/w_i.png), a smoothed estimate of its probability of appearing in either email class (spam or ham), 
![p_word_given_class](eqns/p_word_given_class.png), is calculated by iterating over the training files in `data/spam/` and  `data/ham/`.

The probability of an email belonging to either class can be calculated using Bayes' rule as ![p_class_given_email](eqns/p_class_given_email.png). 

We can express the probability of the email appearing in a particular class as ![p_email_given_class](eqns/p_email_given_class.png), where ![f_i](eqns/f_i.png) is the frequency of the word ![w_i](eqns/w_i.png) in the given email. 

To classify the email as by picking the the class which has a higher probability than the other. For spam, this will happen when
![decision](eqns/log_probability_comparison.png).

Since we are less tolerant of spam, the filter needs to lower the "threshold" to classify an email as spam. Therefore, we introduce a decision factor, ![zeta](eqns/zeta.png), that takes a value ranging from
0 to 1. We can now classify the email as spam if ![decision](eqns/zeta_decision_factor.png) and ham otherwise. 
To evaluate performance of the filter over different values of ![zeta](eqns/zeta.png), we define the following errors,   
 - Type 1 error: fraction of spam emails misclassified as ham
 - Type 2 error: fraction of of ham emails misclassified as spam

The error trade-off curve can be seen below:
![ErrorTradeOffCurve](error_tradeoff_curve.png)

The optimal value of the decision factor is when both error curves meet and are sufficiently low, which happens at ![zeta](eqns/zeta.png) = 0.88. At this value, the filter correctly classifies 42 out of 49 spam emails and 44 out of 51 ham emails in the testing dataset which can be found in `data/testing/`.
